"""Parametric L-bracket: editable values + pure geometry (no SolidWorks needed).

Edit the values in the PARAMETERS block, then run build_bracket.py.

Coordinates (mm), seen from the Front plane:
  X runs along the base length, Y runs up the flange, Z is the bracket width
  (the part is centred on Z = 0).

"""

# ----------------------------- PARAMETERS (mm) -----------------------------
BASE_LENGTH = 80        # L: base plate length (X)
BASE_WIDTH = 40         # W: bracket width (Z)
THICKNESS = 5           # T: plate thickness (base and flange)
FLANGE_HEIGHT = 50      # H: total height of the vertical leg (Y)
FILLET_RADIUS = 5       # inside bend radius (outside radius = this + T), 0 = sharp

HOLE_DIA = 5            # diameter of every hole
EDGE_OFFSET = 10        # hole centre distance from the nearest edge
HOLE_COUNT_BASE = 4     # holes in the base plate: 0, 1, 2 or 4
HOLE_COUNT_FLANGE = 2   # holes in the flange: 0, 1 or 2
# ---------------------------------------------------------------------------


def validate():
    """Raise ValueError if the values cannot make a sensible bracket."""
    L, W, T, H, R = BASE_LENGTH, BASE_WIDTH, THICKNESS, FLANGE_HEIGHT, FILLET_RADIUS
    d, e = HOLE_DIA, EDGE_OFFSET
    problems = []

    if min(L, W, T, H, d) <= 0 or R < 0 or e < 0:
        problems.append("all sizes must be positive (fillet radius may be 0)")
    if HOLE_COUNT_BASE not in (0, 1, 2, 4):
        problems.append("HOLE_COUNT_BASE must be 0, 1, 2 or 4")
    if HOLE_COUNT_FLANGE not in (0, 1, 2):
        problems.append("HOLE_COUNT_FLANGE must be 0, 1 or 2")
    if problems:
        raise ValueError("; ".join(problems))

    if T + R >= L or T + R >= H:
        problems.append("THICKNESS + FILLET_RADIUS must be smaller than BASE_LENGTH and FLANGE_HEIGHT")
    if HOLE_COUNT_BASE or HOLE_COUNT_FLANGE:
        if e < d / 2 + 1:
            problems.append("EDGE_OFFSET is too small for HOLE_DIA")
    if HOLE_COUNT_BASE:
        # holes must clear the bend (flange + fillet) and stay inside the plate
        x_min = T + R + d / 2 + 1
        if L - e < x_min:
            problems.append("BASE_LENGTH too short: base hole would hit the bend")
        if HOLE_COUNT_BASE == 4 and (L - e) - (T + R + e) < 2 * d:
            problems.append("BASE_LENGTH too short for 4 base holes")
        if HOLE_COUNT_BASE == 4 and T + R + e < x_min:
            problems.append("near base holes would hit the bend; increase BASE_LENGTH or reduce EDGE_OFFSET")
        if HOLE_COUNT_BASE in (2, 4) and W / 2 - e < d / 2 + 1:
            problems.append("BASE_WIDTH too small for the base holes")
    if HOLE_COUNT_FLANGE:
        y = H - e
        if y - d / 2 < T + R + 1:
            problems.append("FLANGE_HEIGHT too short: flange hole would hit the bend")
        if HOLE_COUNT_FLANGE == 2 and W / 2 - e < d / 2 + 1:
            problems.append("BASE_WIDTH too small for 2 flange holes")
    if problems:
        raise ValueError("; ".join(problems))


def profile():
    """L cross-section as a list of segments, CCW, in the XY plane.

    Segments are in drawing order and each starts where the previous one ended
    (SolidWorks only joins endpoints that way). Each item is
    ("line", (x1, y1), (x2, y2)) or
    ("arc", (cx, cy), (x_start, y_start), (x_end, y_end), direction)
    with direction 1 = counter-clockwise, -1 = clockwise.
    """
    L, T, H, R = BASE_LENGTH, THICKNESS, FLANGE_HEIGHT, FILLET_RADIUS
    if R == 0:
        pts = [(0, 0), (L, 0), (L, T), (T, T), (T, H), (0, H)]
        return [("line", pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    c = (T + R, T + R)  # inner and outer bend arcs share this centre
    # Start on the outside bend arc: SolidWorks fails to close the loop when the
    # last segment is an arc ending on the first segment's start point.
    return [
        ("arc", c, (0, T + R), (T + R, 0), 1),   # outside bend, radius R + T
        ("line", (T + R, 0), (L, 0)),
        ("line", (L, 0), (L, T)),
        ("line", (L, T), (T + R, T)),
        ("arc", c, (T + R, T), (T, T + R), -1),  # inside bend, radius R
        ("line", (T, T + R), (T, H)),
        ("line", (T, H), (0, H)),
        ("line", (0, H), (0, T + R)),
    ]


def base_holes():
    """Base-plate hole centres as (x, z); the sketch lies in the XZ (Top) plane."""
    L, W, T, R, e = BASE_LENGTH, BASE_WIDTH, THICKNESS, FILLET_RADIUS, EDGE_OFFSET
    zs = W / 2 - e
    x_far = L - e
    n = HOLE_COUNT_BASE
    if n == 0:
        return []
    if n == 1:
        return [(x_far, 0.0)]
    if n == 2:
        return [(x_far, -zs), (x_far, zs)]
    x_near = T + R + e
    return [(x, z) for x in (x_near, x_far) for z in (-zs, zs)]


def flange_holes():
    """Flange hole centres as (z, y); the sketch lies in the YZ (Right) plane."""
    W, H, e = BASE_WIDTH, FLANGE_HEIGHT, EDGE_OFFSET
    y = H - e
    n = HOLE_COUNT_FLANGE
    if n == 0:
        return []
    if n == 1:
        return [(0.0, y)]
    zs = W / 2 - e
    return [(-zs, y), (zs, y)]


if __name__ == "__main__":
    validate()
    print("profile:", *profile(), sep="\n  ")
    print("base holes (x, z):", base_holes())
    print("flange holes (z, y):", flange_holes())
