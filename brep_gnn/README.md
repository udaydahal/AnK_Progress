# BRep to graph (step 1 of BRep + GNN)

Goal of the bigger project: teach a GNN (graph neural network) to recognise features
(holes, fillets, ...) of a CAD part from its BRep shape alone, without SolidWorks.
This folder is step 1: see what a BRep is and turn one into a graph. No AI training yet.

## Words
- **BRep** (boundary representation): a solid stored as its skin: **faces** (surfaces), **edges** (where two faces meet), **vertices** (corners). A `.step` file holds one.
- **Face adjacency graph**: each face is a dot (node); two faces that touch along an edge get a line (edge). This is the form a GNN reads.
- Node numbers: surface type (plane, cylinder, ...), area, direction (normal of a plane, axis of a cylinder), radius.
- Link numbers: shared edge length, and **convex** (outside corner), **concave** (inside corner) or **smooth** (tangent blend, e.g. where a fillet meets a flat face).

## Setup (once)
Needs 64-bit Python (the conda Python here is 32-bit and has no OpenCascade wheels). From this folder:
```
C:\Users\UDAY\AppData\Local\Microsoft\WindowsApps\python3.12.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run
```
.venv\Scripts\python.exe bracket_cq.py       # builds the L-bracket, writes data/bracket.step
.venv\Scripts\python.exe brep_to_graph.py    # reads the STEP, prints the table, writes data/bracket_graph.png
```
Change the numbers in `PARAMS` at the top of `bracket_cq.py` (or call `build_bracket(HOLE_COUNT_BASE=4)`) and run again; the graph changes with the part.

## Files
- `bracket_cq.py`  the bracket built with CadQuery (open-source CAD kernel), exported as STEP
- `brep_to_graph.py`  STEP -> networkx graph (`step_to_graph`), printing and plotting

## What the default bracket gives
14 faces: 8 planes + 6 cylinders (2 bend + 4 holes), 32 links (28 convex, 4 smooth). Adding 2 base holes adds exactly 2 cylinder nodes. With a sharp corner (`FILLET_RADIUS=0`) the inside corner shows up as the only concave link.

## Next steps
2. Train a simple GNN on a public labelled BRep dataset (face-level labels).
3. Generate many labelled brackets with this builder and train on those.
