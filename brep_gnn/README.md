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

  mfcad/              MFCAD++ public dataset (25 machining feature classes)
    train_mfcad.py      converts STEP files to graphs, trains GNN + baseline on MFCAD++
    visualize_mfcad.ipynb notebook: baseline vs GNN comparison, confusion matrix, per-class bars, part viewer

  views2d/            2D views of the same parts, joined to the graph face by face
    render_views.py     STEP -> 7 views: shaded image, line drawing, face-id map (which face each pixel shows)
    vlm_features.py     frozen SigLIP2 image encoder -> one visual vector per face, cached in views2d/data/
    train_fused.py      MFCAD++: 3D-only GNN vs 2D-only vs fused (GNN + visual vectors)
    visualize_views.ipynb notebook: one part through every step, loss curves, results, seen vs hidden faces
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

## MFCAD++ dataset (run from `brep_gnn/`)
Uses the public MFCAD++ dataset (Colligan et al., 2022): 59,665 synthetic STEP files, 25 face classes
(chamfer, through hole, blind hole, pockets, slots, steps, O-ring, round, stock).

Download the dataset and unzip it to `mfcad/data/MFCAD_dataset/MFCAD++_dataset/`.
```
.venv\Scripts\python.exe mfcad\train_mfcad.py [epochs] [train_n] [val_n] [test_n]
```
Defaults: 30 epochs, 6000 / 1000 / 1500 parts. First run is slow (~10–15 min) because it converts
STEP files to graphs on all CPU cores; the results are cached in `mfcad/data/mfcad_cache/` so later
runs start in seconds. Use `0` for a count to use all parts of that split.

Then open `mfcad/visualize_mfcad.ipynb`:
- Baseline (no neighbours) vs GNN comparison: ~49% vs ~98% accuracy
- Confusion matrix with class names
- Per-class accuracy and IoU
- Part viewer showing true vs predicted labels

The GNN is a GATv2 (4 layers, 4 attention heads, 128 hidden, edge features).
The baseline is an MLP that sees each face alone without any neighbour information.
Training uses inverse-frequency class weights (Stock has ~50x more faces than Round).

## 2D views + fusion (run from `brep_gnn/`)
Each part is rendered from 14 directions (front, back, top, bottom, right, left + 8 corner views). Next to every
picture we keep a **face-id map**: the same size, each pixel holds the number of the face seen there
(0 = background, k = graph node k-1). Faces are numbered by the same function as the graph, so the
link between pixels and graph nodes is exact; nothing is learned or guessed.
```
.venv\Scripts\python.exe views2d\render_views.py [file.step]   # picture sheet -> views2d/data/<name>_views.png
.venv\Scripts\python.exe views2d\render_views.py --check 20    # self-test + 20 MFCAD++ parts
.venv\Scripts\python.exe views2d\vlm_features.py               # visual vector per face (~40 min, once)
.venv\Scripts\python.exe views2d\train_fused.py                # 3 models x 3 runs -> views2d/data/fused_results.pt
```
Then open `views2d/visualize_views.ipynb` (reads the saved results, trains nothing).
`vlm_features.py` runs the frozen SigLIP2 image encoder (`google/siglip2-base-patch16-384`, downloaded on
first run) on the shaded views. Its 24 x 24 grid of patch vectors is smoothly enlarged to 4 x 4 pixel cells and
averaged over each face's pixels: 768 numbers + a "seen" flag per face. With the 8 corner views 99.3% of
MFCAD++ test faces are seen by some view (7 views saw ~70%: pocket and hole walls are edge-on from straight views).
The fused GNN squeezes the 769 visual numbers to 32 (with 30% dropout) before appending them to the 15 geometry numbers.

Results (6000 / 1000 / 1500 parts, 30 epochs, 3 runs each, mean +- std):

| Model | Test accuracy | Mean IoU |
|---|---|---|
| 3D only (GNN) | 97.47 +- 0.06% | 92.02 +- 0.21% |
| 2D only (each face alone) | 81.23 +- 0.17% | 64.77 +- 0.13% |
| Fused 2D + 3D (GNN) | 97.93 +- 0.14% | 93.49 +- 0.11% |

First version (7 views, 16 x 16 pooling, plain concatenation, one run): 3D only 97.32%, 2D only 62.98%,
fused 95.91% (it overfit). The fused model now beats 3D only in all 3 runs, by about +0.5 points accuracy and
+1.5 points mean IoU.

