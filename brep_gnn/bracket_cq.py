"""L-bracket built with CadQuery (no SolidWorks) and exported as a STEP file.

Same shape as the SolidWorks version: a base plate and a vertical flange joined
by a bend, with holes in both. The STEP file is a BRep: it stores the part as a
set of faces, edges and vertices.

Run:  python bracket_cq.py        -> data/bracket.step

Axes (mm): X along the base, Y up the flange, Z across the width (centred on 0).
"""
import math
import os

import cadquery as cq

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ----------------------------- PARAMETERS (mm) -----------------------------
PARAMS = dict(
    BASE_LENGTH=80,       # base plate length (X)
    BASE_WIDTH=40,        # bracket width (Z)
    THICKNESS=5,          # plate thickness
    FLANGE_HEIGHT=50,     # total height of the vertical leg (Y)
    FILLET_RADIUS=5,      # inside bend radius, outside = this + THICKNESS, 0 = sharp
    HOLE_DIA=6,
    EDGE_OFFSET=10,       # hole centre distance from the nearest edge
    HOLE_COUNT_BASE=2,    # 0, 1, 2 or 4
    HOLE_COUNT_FLANGE=2,  # 0, 1 or 2
)
# ---------------------------------------------------------------------------


def base_hole_centres(p):
    """(x, z) of the holes in the base plate."""
    zs = p["BASE_WIDTH"] / 2 - p["EDGE_OFFSET"]
    x_far = p["BASE_LENGTH"] - p["EDGE_OFFSET"]
    x_near = p["THICKNESS"] + p["FILLET_RADIUS"] + p["EDGE_OFFSET"]
    return {
        0: [],
        1: [(x_far, 0.0)],
        2: [(x_far, -zs), (x_far, zs)],
        4: [(x, z) for x in (x_near, x_far) for z in (-zs, zs)],
    }[p["HOLE_COUNT_BASE"]]


def flange_hole_centres(p):
    """(y, z) of the holes in the flange."""
    y = p["FLANGE_HEIGHT"] - p["EDGE_OFFSET"]
    zs = p["BASE_WIDTH"] / 2 - p["EDGE_OFFSET"]
    return {0: [], 1: [(y, 0.0)], 2: [(y, -zs), (y, zs)]}[p["HOLE_COUNT_FLANGE"]]


def build_bracket(**overrides):
    """Return the bracket as a cq.Workplane. Any PARAMS value can be overridden."""
    p = {**PARAMS, **overrides}
    L, W, T = p["BASE_LENGTH"], p["BASE_WIDTH"], p["THICKNESS"]
    H, R = p["FLANGE_HEIGHT"], p["FILLET_RADIUS"]

    # 1. L-shaped profile in the XY plane, extruded W/2 each way along Z
    if R == 0:
        sketch = cq.Workplane("XY").polyline([(0, 0), (L, 0), (L, T), (T, T), (T, H), (0, H)]).close()
    else:
        c = T + R                      # bend arcs share the centre (c, c)
        k = math.sqrt(0.5)             # point halfway round a 90 degree arc
        outer_mid = (c - (R + T) * k, c - (R + T) * k)
        inner_mid = (c - R * k, c - R * k)
        sketch = (
            cq.Workplane("XY")
            .moveTo(0, c)
            .threePointArc(outer_mid, (c, 0))
            .lineTo(L, 0)
            .lineTo(L, T)
            .lineTo(c, T)
            .threePointArc(inner_mid, (T, c))
            .lineTo(T, H)
            .lineTo(0, H)
            .close()
        )
    body = sketch.extrude(W / 2, both=True)

    # 2. holes: plain cylinders cut out of the body
    r = p["HOLE_DIA"] / 2
    for x, z in base_hole_centres(p):           # drilled along Y through the base
        body = body.cut(cq.Workplane(obj=cq.Solid.makeCylinder(r, T + 2, cq.Vector(x, -1, z), cq.Vector(0, 1, 0))))
    for y, z in flange_hole_centres(p):         # drilled along X through the flange
        body = body.cut(cq.Workplane(obj=cq.Solid.makeCylinder(r, T + 2, cq.Vector(-1, y, z), cq.Vector(1, 0, 0))))
    return body


def export_step(body, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cq.exporters.export(body, path)
    return path


if __name__ == "__main__":
    out = export_step(build_bracket(), os.path.join(DATA_DIR, "bracket.step"))
    print("Saved", out)
