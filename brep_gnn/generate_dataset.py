"""Make a labelled dataset of random L-brackets for training a GNN.

For every random (but valid) bracket this script:
  1. builds it with CadQuery and saves it as a STEP file,
  2. reads the STEP back and turns it into the face graph,
  3. gives every face a label: "hole", "fillet" or "plain".

The labels come from how the part was built (where the holes were drilled),
NOT from the graph's own link features, so a GNN has to learn them for real.

Run:  python generate_dataset.py [count] [seed]     (default 300 parts, seed 0)
Out:  data/dataset/bracket_0000.step ... and data/dataset.json
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter

from bracket_cq import (DATA_DIR, PARAMS, base_hole_centres, build_bracket, export_step,
                        flange_hole_centres)
from brep_to_graph import step_to_graph

LABELS = ("plain", "hole", "fillet")
HOLE_MATCH_TOL = 0.1  # mm: how close a cylinder's centre must be to a drilled hole's centre


def validate(p):
    """Return a list of problems (empty = the design is fine). Same rules as the SolidWorks version."""
    L, W, T, H, R = p["BASE_LENGTH"], p["BASE_WIDTH"], p["THICKNESS"], p["FLANGE_HEIGHT"], p["FILLET_RADIUS"]
    d, e = p["HOLE_DIA"], p["EDGE_OFFSET"]
    nb, nf = p["HOLE_COUNT_BASE"], p["HOLE_COUNT_FLANGE"]
    bad = []
    if T + R >= L or T + R >= H:
        bad.append("bend too big for the plate")
    if (nb or nf) and e < d / 2 + 1:
        bad.append("edge offset too small")
    x_min = T + R + d / 2 + 1
    if nb:
        if L - e < x_min:
            bad.append("base hole hits the bend")
        if nb == 4 and ((L - e) - (T + R + e) < 2 * d or T + R + e < x_min):
            bad.append("4 base holes do not fit")
        if nb in (2, 4) and W / 2 - e < d / 2 + 1:
            bad.append("base too narrow for the holes")
    if nf:
        if (H - e) - d / 2 < T + R + 1:
            bad.append("flange hole hits the bend")
        if nf == 2 and W / 2 - e < d / 2 + 1:
            bad.append("flange too narrow for 2 holes")
    return bad


def random_params(rng):
    """One random design (may still be invalid; the caller checks with validate)."""
    T = rng.choice([3, 4, 5, 6, 8, 10])
    return dict(
        BASE_LENGTH=rng.randint(50, 150),
        BASE_WIDTH=rng.randint(30, 80),
        THICKNESS=T,
        FLANGE_HEIGHT=rng.randint(40, 120),
        FILLET_RADIUS=0 if rng.random() < 0.2 else rng.randint(2, 12),
        HOLE_DIA=rng.choice([4, 5, 6, 8, 10]),
        EDGE_OFFSET=rng.randint(6, 20),
        HOLE_COUNT_BASE=rng.choice([0, 1, 2, 4]),
        HOLE_COUNT_FLANGE=rng.choice([0, 1, 2]),
    )


def label_faces(G, p):
    """Give every node of G a 'label'. Raises AssertionError if the labels do not add up."""
    T = p["THICKNESS"]
    drilled = [(x, T / 2, z) for x, z in base_hole_centres(p)] + \
              [(T / 2, y, z) for y, z in flange_hole_centres(p)]
    unmatched = list(drilled)

    for _, d in G.nodes(data=True):
        d["label"] = "plain"
        if d["type"] != "cylinder":
            continue
        centre = d["centroid"]
        near = [c for c in unmatched if math.dist(c, centre) < HOLE_MATCH_TOL]
        if near:
            unmatched.remove(near[0])
            d["label"] = "hole"
        else:
            d["label"] = "fillet"

    counts = Counter(d["label"] for _, d in G.nodes(data=True))
    assert not unmatched, f"holes with no matching face: {unmatched}"
    assert counts["hole"] == len(drilled), f"{counts['hole']} hole faces, expected {len(drilled)}"
    expected_fillets = 2 if p["FILLET_RADIUS"] > 0 else 0
    assert counts["fillet"] == expected_fillets, f"{counts['fillet']} fillet faces, expected {expected_fillets}"
    return counts


def graph_to_dict(G):
    return {
        "nodes": [{"id": i, **d} for i, d in G.nodes(data=True)],
        "edges": [{"a": a, "b": b, **d} for a, b, d in G.edges(data=True)],
    }


def make_part(p, step_path):
    export_step(build_bracket(**p), step_path)
    G = step_to_graph(step_path)
    counts = label_faces(G, p)
    return G, counts


def self_test():
    """The default bracket must give 4 hole, 2 fillet, 8 plain faces (what we know by hand)."""
    path = os.path.join(DATA_DIR, "_default.step")
    _, counts = make_part(dict(PARAMS), path)
    os.remove(path)
    assert (counts["hole"], counts["fillet"], counts["plain"]) == (4, 2, 8), counts
    print("self-test: default bracket -> 4 hole, 2 fillet, 8 plain faces  OK")


def main(count=300, seed=0):
    self_test()
    out_dir = os.path.join(DATA_DIR, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    samples, totals, rejected = [], Counter(), 0
    start = time.time()

    while len(samples) < count:
        p = random_params(rng)
        if validate(p):
            rejected += 1
            continue
        name = f"bracket_{len(samples):04d}"
        G, counts = make_part(p, os.path.join(out_dir, name + ".step"))
        totals.update(counts)
        samples.append({"name": name, "params": p, **graph_to_dict(G)})
        if len(samples) % 50 == 0:
            print(f"  {len(samples)}/{count} parts ({time.time() - start:.0f}s)")

    path = os.path.join(DATA_DIR, "dataset.json")
    with open(path, "w") as f:
        json.dump(samples, f)
    faces = sum(totals.values())
    print(f"\n{len(samples)} valid parts ({rejected} invalid designs skipped), {faces} faces in total")
    for label in LABELS:
        print(f"  {label:<7}{totals[label]:>6}  ({100 * totals[label] / faces:.1f}%)")
    print("Saved", path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
