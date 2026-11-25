"""Train a GNN on the public MFCAD++ dataset: label every face of a part with its machining feature.

MFCAD++ (Colligan et al., Hierarchical CADNet) = 59,665 STEP files with 25 face classes
(0 chamfer ... 23 round, 24 stock). The class of each face is stored in the face's name inside
the STEP file, which brep_to_graph.step_to_graph(path, names=True) reads back.

Step 1 (slow, once): STEP files -> graphs, built on all CPU cores and cached in data/mfcad_cache/.
Step 2: train a GNN and a node-only baseline on the official train split, report on the test split.

Run:  python train_mfcad.py [epochs] [train_n] [val_n] [test_n]
      defaults: 30 epochs, 6000 / 1000 / 1500 parts; use 0 for "all parts of that split"
      full benchmark: python train_mfcad.py 30 0 0 0
Needs data/MFCAD_dataset/MFCAD++_dataset/step/{train,val,test}/*.step (unzipped download).

Optional environment variables (handy on Colab, point them at Google Drive so nothing is lost
when the session ends):
  MFCAD_CACHE  folder for the graph caches        (default data/mfcad_cache)
  MFCAD_OUT    folder for weights and results     (default data)
Graphs are built in shards of 2000 parts, each saved as soon as it is done, so a stopped run
continues where it left off when started again.
Out:  MFCAD_OUT/mfcad_gnn.pt       GNN weights (used by predict_step.py)
      MFCAD_OUT/mfcad_results.pt   both models' weights, training curves and test confusion matrices
"""
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared code: brep_gnn/

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from brep_to_graph import step_to_graph
from gnn_common import FaceGNN, FaceMLP, evaluate, one_hot, report, train

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.join(DATA_DIR, "MFCAD_dataset", "MFCAD++_dataset")
CACHE = os.environ.get("MFCAD_CACHE") or os.path.join(DATA_DIR, "mfcad_cache")
OUT_DIR = os.environ.get("MFCAD_OUT") or DATA_DIR
WEIGHTS = os.path.join(OUT_DIR, "mfcad_gnn.pt")
RESULTS = os.path.join(OUT_DIR, "mfcad_results.pt")
SHARD = 2000                        # parts per saved piece of the graph cache
WORKERS = max(2, (os.cpu_count() or 2) - 1)
SEED = 0
HIDDEN, LAYERS = 128, 4

LABELS = ["Chamfer", "Through hole", "Triangular passage", "Rectangular passage", "6-sided passage",
          "Triangular through slot", "Rectangular through slot", "Circular through slot",
          "Rectangular through step", "2-sided through step", "Slanted through step", "O-ring",
          "Blind hole", "Triangular pocket", "Rectangular pocket", "6-sided pocket",
          "Circular end pocket", "Rectangular blind slot", "Vertical circular end blind slot",
          "Horizontal circular end blind slot", "Triangular blind step", "Circular blind step",
          "Rectangular blind step", "Round", "Stock"]
TYPES = ("plane", "cylinder", "cone", "sphere", "torus", "bspline", "other")
CONVEXITY = ("convex", "concave", "smooth")


def node_features(n):
    direction = n.get("direction", (0.0, 0.0, 0.0))
    return (one_hot(n["type"], TYPES)
            + [n["area"] / 500, n.get("radius", 0.0) / 10]
            + [abs(c) for c in direction]
            + [c / 50 for c in n["centroid"]])


def edge_features(e):
    return [e["length"] / 50, e["angle"] / 180] + one_hot(e["convexity"], CONVEXITY)


def step_to_data(path):
    """One STEP file -> a PyG graph with a label per face (None if the file cannot be used)."""
    try:
        G = step_to_graph(path, names=True)
        nodes = [G.nodes[i] for i in range(G.number_of_nodes())]
        y = [int(n["name"]) for n in nodes]
        src, dst, attr = [], [], []
        for a, b, e in G.edges(data=True):
            f = edge_features(e)
            src += [a, b]
            dst += [b, a]
            attr += [f, f]
        if not src:
            return None
        return Data(x=torch.tensor([node_features(n) for n in nodes], dtype=torch.float),
                    y=torch.tensor(y), edge_index=torch.tensor([src, dst]),
                    edge_attr=torch.tensor(attr, dtype=torch.float))
    except Exception:
        return None


def split_files(split, count):
    """Folder and file names of `count` random parts of a split (0 = all); same pick every time."""
    folder = os.path.join(ROOT, "step", split)
    files = sorted(os.listdir(folder))
    if count:
        files = random.Random(SEED).sample(files, min(count, len(files)))
    return folder, files


def load_split(split, count):
    """Graphs of `count` random parts of a split (0 = all), built once and cached.
    Every graph carries `name`, the STEP file it came from (unusable files are skipped)."""
    folder, files = split_files(split, count)
    cache_file = os.path.join(CACHE, f"{split}_{len(files)}.pt")
    if os.path.exists(cache_file):
        graphs = torch.load(cache_file, weights_only=False)
        if graphs and not hasattr(graphs[0], "name") and len(graphs) == len(files):
            for g, f in zip(graphs, files):           # caches from before names were stored
                g.name = f
        return graphs

    shard_dir = os.path.join(CACHE, f"{split}_{len(files)}_parts")
    os.makedirs(shard_dir, exist_ok=True)
    print(f"building {len(files)} {split} graphs from STEP files on {WORKERS} processes ...")
    start, good, skipped = time.time(), [], 0
    with Pool(WORKERS) as pool:
        for s in range(0, len(files), SHARD):
            shard_file = os.path.join(shard_dir, f"shard_{s // SHARD:03d}.pt")
            if os.path.exists(shard_file):            # done in an earlier run
                good += torch.load(shard_file, weights_only=False)
                print(f"  {min(s + SHARD, len(files))}/{len(files)} already built")
                continue
            names = files[s:s + SHARD]
            graphs = pool.map(step_to_data, [os.path.join(folder, f) for f in names], chunksize=20)
            done = []
            for g, f in zip(graphs, names):
                if g is None:
                    skipped += 1
                    continue
                g.name = f
                done.append(g)
            torch.save(done, shard_file + ".tmp")     # write fully, then rename: no half-written shards
            os.replace(shard_file + ".tmp", shard_file)
            good += done
            print(f"  {s + len(names)}/{len(files)} ({time.time() - start:.0f}s)")
    print(f"  {len(good)} usable, {skipped} skipped in this run ({time.time() - start:.0f}s)")
    torch.save(good, cache_file)
    return good


def main(epochs=30, n_train=6000, n_val=1000, n_test=1500):
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    train_set = load_split("train", n_train)
    val_set = load_split("val", n_val)
    test_set = load_split("test", n_test)
    print(f"{len(train_set)} train / {len(val_set)} validation / {len(test_set)} test parts")

    k = len(LABELS)
    counts = torch.bincount(torch.cat([g.y for g in train_set]), minlength=k).float()
    weights = (counts.sum() / (k * counts.clamp(min=1))).to(device)
    print("training faces per class:", {LABELS[i]: c for i, c in enumerate(counts.int().tolist()) if c})

    loaders = [DataLoader(s, batch_size=32, shuffle=(s is train_set)) for s in (train_set, val_set, test_set)]
    n_in, n_edge = train_set[0].x.shape[1], train_set[0].edge_attr.shape[1]

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"epochs": epochs, "seed": SEED, "class_weights": weights.cpu(),
               "parts": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
               "state": {}, "history": {}, "cm": {}}
    models = (("baseline", FaceMLP(n_in, k, HIDDEN), "the baseline (each face alone)"),
              ("gnn", FaceGNN(n_in, n_edge, k, HIDDEN, LAYERS), "the GNN"))
    for key, model, title in models:
        print(f"\ntraining {title}")
        torch.manual_seed(SEED)
        model, history = train(model, loaders[0], loaders[1], epochs, device, key, k, weights, print_every=5)
        results["state"][key] = {n: v.cpu() for n, v in model.state_dict().items()}
        results["history"][key] = history
        results["cm"][key] = evaluate(model, loaders[2], device, k).cpu()
        torch.save(results, RESULTS)                  # saved after each model, in case the session ends

    report("Baseline (no neighbours)", results["cm"]["baseline"], LABELS)
    report("GNN (uses neighbours and links)", results["cm"]["gnn"], LABELS)
    torch.save(results["state"]["gnn"], WEIGHTS)
    print("\nSaved", WEIGHTS, "and", RESULTS)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
