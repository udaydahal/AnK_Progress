"""Train a GNN on the public MFCAD++ dataset: label every face of a part with its machining feature.

MFCAD++ (Colligan et al., Hierarchical CADNet) = 59,665 STEP files with 25 face classes
(0 chamfer ... 23 round, 24 stock). The class of each face is stored in the face's name inside
the STEP file, which brep_to_graph.step_to_graph(path, names=True) reads back.

Step 1 (slow, once): STEP files -> graphs, built on all CPU cores and cached in data/mfcad_cache/.
Step 2: train a GNN and a node-only baseline on the official train split, report on the test split.

Run:  python train_mfcad.py [epochs] [train_n] [val_n] [test_n]
      defaults: 30 epochs, 6000 / 1000 / 1500 parts; use 0 for "all parts of that split"
Needs data/MFCAD_dataset/MFCAD++_dataset/step/{train,val,test}/*.step (unzipped download).
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
CACHE = os.path.join(DATA_DIR, "mfcad_cache")
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


def load_split(split, count):
    """Graphs of `count` random parts of a split (0 = all), built once and cached."""
    folder = os.path.join(ROOT, "step", split)
    files = sorted(os.listdir(folder))
    if count:
        files = random.Random(SEED).sample(files, min(count, len(files)))
    cache_file = os.path.join(CACHE, f"{split}_{len(files)}.pt")
    if os.path.exists(cache_file):
        return torch.load(cache_file, weights_only=False)

    os.makedirs(CACHE, exist_ok=True)
    print(f"building {len(files)} {split} graphs from STEP files ...")
    start = time.time()
    paths = [os.path.join(folder, f) for f in files]
    with Pool(max(1, (os.cpu_count() or 2) - 4)) as pool:
        graphs = []
        for i, g in enumerate(pool.imap(step_to_data, paths, chunksize=20), 1):
            graphs.append(g)
            if i % 1000 == 0:
                print(f"  {i}/{len(paths)} ({time.time() - start:.0f}s)")
    good = [g for g in graphs if g is not None]
    print(f"  {len(good)} usable, {len(graphs) - len(good)} skipped ({time.time() - start:.0f}s)")
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
    counts = torch.bincount(torch.cat([g.y for g in train_set]), minlength=k)
    print("training faces per class:", {LABELS[i]: c for i, c in enumerate(counts.tolist()) if c})

    loaders = [DataLoader(s, batch_size=32, shuffle=(s is train_set)) for s in (train_set, val_set, test_set)]
    n_in, n_edge = train_set[0].x.shape[1], train_set[0].edge_attr.shape[1]

    print("\ntraining the baseline (each face alone)")
    mlp, _ = train(FaceMLP(n_in, k, HIDDEN), loaders[0], loaders[1], epochs, device, "baseline", k, print_every=5)
    print("\ntraining the GNN")
    gnn, _ = train(FaceGNN(n_in, n_edge, k, HIDDEN, LAYERS), loaders[0], loaders[1], epochs, device, "gnn", k, print_every=5)

    report("Baseline (no neighbours)", evaluate(mlp, loaders[2], device, k), LABELS)
    report("GNN (uses neighbours and links)", evaluate(gnn, loaders[2], device, k), LABELS)
    torch.save(gnn.state_dict(), os.path.join(DATA_DIR, "mfcad_gnn.pt"))
    print("\nSaved", os.path.join(DATA_DIR, "mfcad_gnn.pt"))


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
