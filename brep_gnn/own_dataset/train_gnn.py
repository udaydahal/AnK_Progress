"""Train a small GNN that labels every face of a bracket as plain / hole / fillet.

Reads data/dataset.json (made by generate_dataset.py), turns every part into a PyTorch
Geometric graph, trains on 70% of the parts and tests on 15% it has never seen.
A second model that ignores the neighbours (node features only) shows what the graph adds.

Run:  python train_gnn.py [epochs]      (default 150)
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared code: brep_gnn/

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from bracket_cq import DATA_DIR
from gnn_common import FaceGNN, FaceMLP, evaluate, one_hot, report, train

LABELS = ("plain", "hole", "fillet")
TYPES = ("plane", "cylinder", "other")
CONVEXITY = ("convex", "concave", "smooth")
SEED = 0


def node_features(n):
    kind = n["type"] if n["type"] in TYPES else "other"
    direction = n.get("direction", (0.0, 0.0, 0.0))
    return (one_hot(kind, TYPES)
            + [n["area"] / 1000, n.get("radius", 0.0) / 10]
            + [abs(c) for c in direction]      # sign of an axis does not matter for a cylinder
            + [c / 100 for c in n["centroid"]])


def edge_features(e):
    return [e["length"] / 100, e["angle"] / 180] + one_hot(e["convexity"], CONVEXITY)


def to_graph(part):
    x = torch.tensor([node_features(n) for n in part["nodes"]], dtype=torch.float)
    y = torch.tensor([LABELS.index(n["label"]) for n in part["nodes"]])
    src, dst, attr = [], [], []
    for e in part["edges"]:               # every link goes both ways
        f = edge_features(e)
        src += [e["a"], e["b"]]
        dst += [e["b"], e["a"]]
        attr += [f, f]
    return Data(x=x, y=y, edge_index=torch.tensor([src, dst]), edge_attr=torch.tensor(attr, dtype=torch.float))


def main(epochs=150):
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    with open(os.path.join(DATA_DIR, "dataset.json")) as f:
        parts = json.load(f)
    graphs = [to_graph(p) for p in parts]
    random.shuffle(graphs)                           # split by whole part, never by face
    n_train, n_val = int(0.70 * len(graphs)), int(0.15 * len(graphs))
    train_set, val_set, test_set = graphs[:n_train], graphs[n_train:n_train + n_val], graphs[n_train + n_val:]
    print(f"{len(graphs)} parts -> {len(train_set)} train / {len(val_set)} validation / {len(test_set)} test")

    counts = torch.bincount(torch.cat([g.y for g in train_set]), minlength=len(LABELS)).float()
    weights = (counts.sum() / (len(LABELS) * counts)).to(device)   # rare classes count more
    print("training faces per class:", dict(zip(LABELS, counts.int().tolist())))

    loaders = [DataLoader(s, batch_size=32, shuffle=(s is train_set)) for s in (train_set, val_set, test_set)]
    n_in, n_edge = graphs[0].x.shape[1], graphs[0].edge_attr.shape[1]

    k = len(LABELS)
    print("\ntraining the baseline (node features only)")
    mlp = train(FaceMLP(n_in, k), loaders[0], loaders[1], epochs, device, "baseline", k, weights)
    print("\ntraining the GNN")
    gnn = train(FaceGNN(n_in, n_edge, k), loaders[0], loaders[1], epochs, device, "gnn", k, weights)

    report("Baseline (no neighbours)", evaluate(mlp, loaders[2], device, k), LABELS)
    report("GNN (uses neighbours and links)", evaluate(gnn, loaders[2], device, k), LABELS)
    torch.save(gnn.state_dict(), os.path.join(DATA_DIR, "face_gnn.pt"))
    print("\nSaved", os.path.join(DATA_DIR, "face_gnn.pt"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
