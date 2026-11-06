"""Predict machining features on any STEP file using the trained MFCAD++ GNN.

Loads the saved weights (data/mfcad_gnn.pt), converts the STEP file to a face
adjacency graph, and predicts one of 25 machining-feature classes for every face.

Run:  python predict_step.py                         (uses the default test file)
      python predict_step.py path/to/your_part.step  (any STEP file)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib
import networkx as nx
from torch_geometric.data import Data

from brep_to_graph import step_to_graph

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from gnn_common import FaceGNN
from train_mfcad import LABELS, TYPES, CONVEXITY, HIDDEN, LAYERS, node_features, edge_features

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_FILE = os.path.join(DATA_DIR, "test_file", "Parça 259.step")
WEIGHTS = os.path.join(DATA_DIR, "mfcad_gnn.pt")

N_IN = len(TYPES) + 2 + 3 + 3      # 15 node features
N_EDGE = 2 + len(CONVEXITY)          # 5 edge features
K = len(LABELS)                       # 25 classes


def step_to_data(path):
    G = step_to_graph(path)
    nodes = [G.nodes[i] for i in range(G.number_of_nodes())]
    src, dst, attr = [], [], []
    for a, b, e in G.edges(data=True):
        f = edge_features(e)
        src += [a, b]
        dst += [b, a]
        attr += [f, f]
    if not src:
        raise ValueError("STEP file produced a graph with no edges")
    return Data(
        x=torch.tensor([node_features(n) for n in nodes], dtype=torch.float),
        edge_index=torch.tensor([src, dst]),
        edge_attr=torch.tensor(attr, dtype=torch.float),
    ), nodes


def predict(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading weights from {WEIGHTS}")
    model = FaceGNN(N_IN, N_EDGE, K, HIDDEN, LAYERS).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    print(f"reading {path}")
    data, nodes = step_to_data(path)
    data = data.to(device)

    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu()
        confs = probs.max(dim=1).values.cpu()

    n = data.num_nodes
    avg_conf = confs.mean().item()
    high = (confs >= 0.90).sum().item()
    mid = ((confs >= 0.50) & (confs < 0.90)).sum().item()
    low = (confs < 0.50).sum().item()

    print()
    print("=" * 74)
    print(f"  PREDICTION REPORT: {os.path.basename(path)}")
    print("=" * 74)
    print(f"  Device      : {device}")
    print(f"  Weights     : {os.path.basename(WEIGHTS)}")
    print(f"  Total faces : {n}")
    print(f"  Total edges : {data.edge_index.shape[1] // 2}")
    print(f"  Avg confidence : {100*avg_conf:.1f}%")
    print(f"  High (>=90%)   : {high}/{n} faces")
    print(f"  Medium (50-90%): {mid}/{n} faces")
    print(f"  Low (<50%)     : {low}/{n} faces")
    print("=" * 74)

    print(f"\n{'Face':>5}  {'Surface type':<12}  {'Predicted class':<35}  {'Confidence':>10}")
    print("-" * 70)
    for i in range(n):
        flag = " *" if confs[i] < 0.50 else ""
        print(f"{i:5d}  {nodes[i]['type']:<12}  {LABELS[preds[i]]:<35}  {100*confs[i]:5.1f}%{flag}")
    print("-" * 70)
    print("  (* = low confidence, model is uncertain)\n")

    counts = {}
    conf_by_class = {}
    for i in range(n):
        name = LABELS[preds[i].item()]
        counts[name] = counts.get(name, 0) + 1
        conf_by_class.setdefault(name, []).append(confs[i].item())

    print(f"  {'Predicted feature':<35}  {'Count':>5}  {'Avg conf':>8}")
    print("  " + "-" * 52)
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        avg = 100 * sum(conf_by_class[name]) / len(conf_by_class[name])
        print(f"  {name:<35}  {cnt:5d}  {avg:7.1f}%")
    print("  " + "-" * 52)
    print(f"  {'TOTAL':<35}  {n:5d}  {100*avg_conf:7.1f}%")
    print()

    draw_graph(data, nodes, preds, confs, os.path.basename(path))


def draw_graph(data, nodes, preds, confs, title):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.cpu()
    for col in range(ei.shape[1]):
        a, b = ei[0, col].item(), ei[1, col].item()
        if a < b:
            G.add_edge(a, b)

    cmap = plt.colormaps["tab20"]
    colours = [cmap(preds[i].item() % 20) for i in range(data.num_nodes)]

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
    pos = nx.spring_layout(G, seed=42, k=1.4)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color=colours,
                           edgecolors="white", linewidths=1.5)
    labels = {i: f"F{i}\n{LABELS[preds[i]]}\n{100*confs[i]:.0f}%" for i in range(data.num_nodes)}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=6)
    ax.set_title(f"Predicted machining features: {title}  ({data.num_nodes} faces)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    out = os.path.join(DATA_DIR, "test_file", title.rsplit(".", 1)[0] + "_predictions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGraph saved to {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    if not os.path.isfile(path):
        print(f"error: file not found: {path}")
        sys.exit(1)
    predict(path)
