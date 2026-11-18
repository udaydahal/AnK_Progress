"""Does knowing what a part LOOKS like help the face classifier? Train three models on MFCAD++:

  3D only   the GNN of train_mfcad.py (face numbers + links between faces)
  2D only   each face alone, judged from its visual vector (SigLIP2 features, see vlm_features.py)
  fused     the same GNN; every face's visual vector is first squeezed to VIS_DIM numbers (with
            dropout) and then appended to its 15 geometry numbers

All three use the same parts, epochs and class weights; the test split is only looked at once per run.
Each model is trained `runs` times with different seeds, so the table shows mean +- spread.

Run:  python train_fused.py [epochs] [train_n] [val_n] [test_n] [runs]     (defaults 30 6000 1000 1500 3)
Needs the graph caches of train_mfcad.py and the feature caches of vlm_features.py (built if missing).
Out:  data/fused_results.pt  (per run: loss / validation curves, test confusion matrices, trained weights)
"""
import copy
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                        # shared code: brep_gnn/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mfcad"))

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from gnn_common import FaceGNN, FaceMLP, accuracy, evaluate, mean_iou, report, train
from train_mfcad import HIDDEN, LABELS, LAYERS, SEED, load_split
from vlm_features import build_split

VIS_DIM = 32        # the fused GNN sees the 769 visual numbers squeezed to this many
VIS_DROPOUT = 0.3   # share of visual numbers randomly hidden during training (fights memorising)
MODELS = ["3D only (GNN)", "2D only (each face alone)", "fused 2D + 3D (GNN)"]


class VisualMLP(FaceMLP):
    """2D only: each face judged alone from its visual vector."""

    def forward(self, data):
        return self.net(data.v)


class FusedGNN(nn.Module):
    """3D + 2D: the visual vector is squeezed to VIS_DIM numbers, then appended to every face's own numbers."""

    def __init__(self, n_geo, n_vis, n_edge, n_classes):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Dropout(VIS_DROPOUT), nn.Linear(n_vis, VIS_DIM), nn.ReLU())
        self.gnn = FaceGNN(n_geo + VIS_DIM, n_edge, n_classes, HIDDEN, LAYERS)

    def forward(self, data):
        data = copy.copy(data)
        data.x = torch.cat([data.x, self.squeeze(data.v)], dim=1)
        return self.gnn(data)


def build_model(name, n_geo, n_vis, n_edge, k):
    if name == MODELS[0]:
        return FaceGNN(n_geo, n_edge, k, HIDDEN, LAYERS)
    if name == MODELS[1]:
        return VisualMLP(n_vis, k, HIDDEN)
    return FusedGNN(n_geo, n_vis, n_edge, k)


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


def main(epochs=30, n_train=6000, n_val=1000, n_test=1500, runs=3):
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

    saved = {"runs": [], "epochs": epochs, "sizes": sizes, "vis_mean": mean, "vis_std": std,
             "config": {"vis_dim": VIS_DIM, "vis_dropout": VIS_DROPOUT}}
    for run in range(runs):
        seed = SEED + run
        out = {"seed": seed, "history": {}, "cm": {}, "state": {}}
        for name in MODELS:
            print(f"\nrun {run + 1}/{runs} (seed {seed}), training: {name}")
            random.seed(seed)
            torch.manual_seed(seed)                       # same start weights and shuffling for this seed
            model = build_model(name, n_geo, n_vis, n_edge, k)
            model, history = train(model, loaders[0], loaders[1], epochs, device, name.split(" (")[0], k,
                                   weights, print_every=10)
            cm = evaluate(model, loaders[2], device, k)
            print(f"  test accuracy {100 * accuracy(cm):.2f}%, mean IoU {100 * mean_iou(cm):.2f}%")
            out["history"][name], out["cm"][name] = history, cm
            out["state"][name] = {key: val.cpu() for key, val in model.state_dict().items()}
        saved["runs"].append(out)

    report(f"{MODELS[2]}, run 1", saved["runs"][0]["cm"][MODELS[2]], LABELS)
    print("\n" + "=" * 72)
    print(f"{'model':<30}{'test accuracy':>20}{'mean IoU':>20}   ({runs} runs, mean +- std)")
    for name in MODELS:
        acc = np.array([100 * accuracy(r["cm"][name]) for r in saved["runs"]])
        iou = np.array([100 * mean_iou(r["cm"][name]) for r in saved["runs"]])
        print(f"{name:<30}{acc.mean():>12.2f} +- {acc.std():.2f}%{iou.mean():>12.2f} +- {iou.std():.2f}%")
    path = os.path.join(HERE, "data", "fused_results.pt")
    torch.save(saved, path)
    print("\nSaved", path)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
