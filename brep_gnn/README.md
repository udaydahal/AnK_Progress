# BRep + GNN

Goal of the bigger project: teach a GNN (graph neural network) to recognise features
(holes, fillets, pockets, ...) of a CAD part from its BRep shape alone, without SolidWorks.

## Layout
```
brep_gnn/
  brep_to_graph.py    shared: STEP file -> face graph (reads faces, edges, convexity)
  gnn_common.py       shared: GNN model, node-only baseline, training loop, accuracy / IoU report
  requirements.txt
  .venv/              the Python environment (64-bit Python 3.12)

  own_dataset/        our own labelled L-brackets (3 classes: plain, hole, fillet)
    bracket_cq.py       builds one bracket with CadQuery -> STEP
    generate_dataset.py makes 300 random labelled brackets -> data/dataset.json
    train_gnn.py        trains the GNN and a baseline, prints results
    visualize_own.ipynb notebook: confusion matrix, part viewer, "does the graph help?" test
```

## Words
- **BRep** (boundary representation): a solid stored as its skin: **faces** (surfaces), **edges** (where two faces meet), **vertices** (corners). A `.step` file holds one.
- **Face adjacency graph**: each face is a dot (node); two faces that touch along an edge get a line (edge). This is the form a GNN reads.
- Node numbers: surface type (plane, cylinder, ...), area, direction (normal of a plane, axis of a cylinder), radius.
- Link numbers: shared edge length, and **convex** (outside corner), **concave** (inside corner) or **smooth** (tangent blend, e.g. where a fillet meets a flat face).

## Setup (once)
Needs 64-bit Python (the conda Python here is 32-bit and has no OpenCascade wheels). From `brep_gnn/`:
```
C:\Users\UDAY\AppData\Local\Microsoft\WindowsApps\python3.12.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
`requirements.txt` has a comment on how to install the CUDA build of PyTorch (for the GPU).

## Own dataset (run from `brep_gnn/`)
```
.venv\Scripts\python.exe own_dataset\bracket_cq.py        # one bracket -> own_dataset/data/bracket.step
.venv\Scripts\python.exe brep_to_graph.py                 # its face graph: table + own_dataset/data/bracket_graph.png
.venv\Scripts\python.exe own_dataset\generate_dataset.py  # 300 labelled parts -> own_dataset/data/dataset.json
.venv\Scripts\python.exe own_dataset\train_gnn.py [epochs]
```
Then open `own_dataset/visualize_own.ipynb`. Both models reach 100% on these easy brackets; the last section of the notebook hides the easy clues (axis direction, position) to show what the graph adds.

Default bracket: 14 faces (8 planes + 6 cylinders: 2 bend + 4 holes), 32 links (28 convex, 4 smooth).

