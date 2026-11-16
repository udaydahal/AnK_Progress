"""Does knowing what a part LOOKS like help the face classifier? Train three models on MFCAD++:

  3D only   the GNN of train_mfcad.py (face numbers + links between faces)
  2D only   each face alone, judged from its visual vector (SigLIP2 features, see vlm_features.py)
  fused     the same GNN, but every face also gets its visual vector (concatenated to its 15 numbers)

All three use the same parts, epochs and class weights; the test split is only looked at once at the end.

Run:  python train_fused.py [epochs] [train_n] [val_n] [test_n]     (defaults 30 6000 1000 1500)
Needs the graph caches of train_mfcad.py and the feature caches of vlm_features.py (built if missing).
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # shared code: brep_gnn/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mfcad"))

import random

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from gnn_common import FaceGNN, FaceMLP, accuracy, evaluate, mean_iou, report, train
from train_mfcad import HIDDEN, LABELS, LAYERS, SEED, load_split
from vlm_features import build_split


class VisualMLP(FaceMLP):
    """2D only: each face judged alone from its visual vector."""

    def forward(self, data):
        return self.net(data.v)


class FusedGNN(nn.Module):
    """3D + 2D: the GNN, with the visual vector appended to every face's own numbers."""

    def __init__(self, n_geo, n_vis, n_edge, n_classes):
        super().__init__()
        self.gnn = FaceGNN(n_geo + n_vis, n_edge, n_classes, HIDDEN, LAYERS)

    def forward(self, data):
        data = copy.copy(data)
        data.x = torch.cat([data.x, data.v], dim=1)
        return self.gnn(data)


def attach(graphs, feats, mean=None, std=None):
    """Put each part's visual vectors on its graph as graph.v (standardised; 0 for faces no view sees)."""
    assert all(f is not None for f in feats), "some parts could not be rendered"
    for g, f in zip(graphs, feats):
        assert f.shape[0] == g.num_nodes, "views and graph disagree about the number of faces"
    if mean is None:
        seen = torch.cat([f[f[:, -1] > 0, :-1].float() for f in feats])
        mean, std = seen.mean(0), seen.std(0) + 1e-6
    for g, f in zip(graphs, feats):
        f = f.float()
        v = torch.where(f[:, -1:] > 0, (f[:, :-1] - mean) / std, torch.zeros_like(f[:, :-1]))
        g.v = torch.cat([v, f[:, -1:]], dim=1)
    return mean, std


def main(epochs=30, n_train=6000, n_val=1000, n_test=1500):
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    sizes = {"train": n_train, "val": n_val, "test": n_test}
    graphs = {s: load_split(s, n) for s, n in sizes.items()}
    mean, std = attach(graphs["train"], build_split("train", n_train))
    for s in ("val", "test"):
        attach(graphs[s], build_split(s, sizes[s]), mean, std)
    train_set, val_set, test_set = graphs["train"], graphs["val"], graphs["test"]
    print(f"{len(train_set)} train / {len(val_set)} validation / {len(test_set)} test parts")

    k = len(LABELS)
    counts = torch.bincount(torch.cat([g.y for g in train_set]), minlength=k).float()
    weights = (counts.sum() / (k * counts.clamp(min=1))).to(device)
    loaders = [DataLoader(s, batch_size=32, shuffle=(s is train_set)) for s in (train_set, val_set, test_set)]
    n_geo, n_edge, n_vis = train_set[0].x.shape[1], train_set[0].edge_attr.shape[1], train_set[0].v.shape[1]

    models = {
        "3D only (GNN)": FaceGNN(n_geo, n_edge, k, HIDDEN, LAYERS),
        "2D only (each face alone)": VisualMLP(n_vis, k, HIDDEN),
        "fused 2D + 3D (GNN)": FusedGNN(n_geo, n_vis, n_edge, k),
    }
    results = {}
    for name, model in models.items():
        print(f"\ntraining: {name}")
        torch.manual_seed(SEED)
        model, _ = train(model, loaders[0], loaders[1], epochs, device, name.split(" (")[0], k, weights, print_every=5)
        cm = evaluate(model, loaders[2], device, k)
        report(name, cm, LABELS)
        results[name] = (accuracy(cm), mean_iou(cm))

    print("\n" + "=" * 60)
    print(f"{'model':<30}{'test accuracy':>16}{'mean IoU':>12}")
    for name, (acc, iou) in results.items():
        print(f"{name:<30}{100 * acc:>15.2f}%{100 * iou:>11.2f}%")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
