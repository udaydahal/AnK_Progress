"""Render a STEP part into 2D views, and record which BRep face every pixel shows.

For each of 7 views (front, back, top, bottom, right, left, iso) we make
  shaded   a grey picture of the part                         (H, W, 3) uint8
  drawing  an engineering-style line drawing, hidden edges dashed (H, W)  uint8
  face_id  for every pixel, which face is visible there       (H, W)    uint16
           0 = background, k = face k-1 = node k-1 of brep_to_graph.step_to_graph

Because we render the part ourselves, face_id is exact: no learned matching is needed to say
which pixels belong to which node of the graph.

Run:  python render_views.py [file.step]        render one part, save a picture sheet
      python render_views.py --check [count]    self-test + coverage/speed on random MFCAD++ parts
"""
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared code: brep_gnn/

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_Line
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

from brep_to_graph import DATA_DIR, get_face_list, read_step, step_to_graph

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")
MFCAD_TEST = os.path.join(os.path.dirname(HERE), "mfcad", "data", "MFCAD_dataset", "MFCAD++_dataset", "step", "test")

IMG_SIZE = 384          # 384 / 16 = 24 x 24 patches for a ViT encoder later
MESH_DEFLECTION = 0.1   # mm, how finely the surfaces are cut into triangles
FILL = 0.9              # the part's bounding diagonal fills this much of the image

# name: (direction towards the viewer, which way is up)
VIEWS = {
    "front": ((0, -1, 0), (0, 0, 1)),
    "back": ((0, 1, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, -1, 0)),
    "right": ((1, 0, 0), (0, 0, 1)),
    "left": ((-1, 0, 0), (0, 0, 1)),
    "iso": ((1, -1, 1), (0, 0, 1)),
}


def camera(name):
    """Orthonormal (towards viewer, right, up) for a named view."""
    w, up = (np.array(v, float) for v in VIEWS[name])
    w /= np.linalg.norm(w)
    right = np.cross(up, w)
    right /= np.linalg.norm(right)
    return w, right, np.cross(w, right)


def triangulate(shape):
    """Cut every face into triangles. Returns (n, 3, 3) corner coordinates and the face index of each triangle."""
    BRepMesh_IncrementalMesh(shape, MESH_DEFLECTION, False, 0.3, False)
    corners, owner = [], []
    for i, face in enumerate(get_face_list(shape)):
        loc = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, loc)
        if mesh is None:
            continue
        trsf = loc.Transformation()
        pts = []
        for k in range(1, mesh.NbNodes() + 1):
            p = mesh.Node(k).Transformed(trsf)
            pts.append((p.X(), p.Y(), p.Z()))
        for k in range(1, mesh.NbTriangles() + 1):
            a, b, c = mesh.Triangle(k).Get()
            corners.append((pts[a - 1], pts[b - 1], pts[c - 1]))
            owner.append(i)
    return np.array(corners), np.array(owner)


def rasterize(tri_px, size):
    """Z-buffer. tri_px is (n, 3, 3): pixel x, pixel y, depth (bigger = closer) of each corner.
    Returns the index of the closest triangle at every pixel (-1 = none) and the depth there."""
    hit = np.full((size, size), -1, np.int32)
    depth = np.full((size, size), -np.inf)
    for t, (a, b, c) in enumerate(tri_px):
        x0, x1 = max(int(np.floor(min(a[0], b[0], c[0]))), 0), min(int(np.ceil(max(a[0], b[0], c[0]))), size - 1)
        y0, y1 = max(int(np.floor(min(a[1], b[1], c[1]))), 0), min(int(np.ceil(max(a[1], b[1], c[1]))), size - 1)
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if x1 < x0 or y1 < y0 or abs(denom) < 1e-12:   # off-screen, or seen exactly edge-on
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / denom
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / denom
        w2 = 1 - w0 - w1
        z = w0 * a[2] + w1 * b[2] + w2 * c[2]
        window = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        win = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (z > depth[window])
        depth[window][win] = z[win]
        hit[window][win] = t
    return hit, depth


def fit(corners):
    """Image centre (3D point) and pixels-per-mm, the same for every view of a part."""
    pts = corners.reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, FILL * IMG_SIZE / np.linalg.norm(hi - lo)


def project(points, centre, scale, cam):
    """3D points -> pixel x, pixel y, depth in one view."""
    w, right, up = cam
    d = points - centre
    return np.stack([IMG_SIZE / 2 + scale * d @ right, IMG_SIZE / 2 - scale * d @ up, d @ w], axis=-1)


def render_mesh(corners, owner, centre, scale, cam):
    """Shaded picture and face-id buffer of the triangulated part in one view."""
    tri_px = project(corners, centre, scale, cam)
    hit, depth = rasterize(tri_px, IMG_SIZE)
    face_id = np.where(hit >= 0, owner[hit] + 1, 0).astype(np.uint16)

    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    w, right, up = cam
    light = w + 0.4 * right + 0.6 * up
    shade = 0.4 + 0.6 * np.abs(normal @ (light / np.linalg.norm(light)))   # two-sided
    grey = shade[hit]
    seen = hit >= 0
    if seen.any():                                       # farther surfaces a little darker
        z = depth[seen]
        grey[seen] *= 0.65 + 0.35 * (z - z.min()) / max(z.max() - z.min(), 1e-9)
    image = np.where(seen, 255 * grey, 255).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2), face_id


def edge_polylines(compound, n=24):
    """Sample every edge of an HLR result into a list of (x, y) points in the view plane."""
    lines = []
    if compound is None or compound.IsNull():
        return lines
    exp = TopExp_Explorer(compound, TopAbs_EDGE)
    while exp.More():
        curve = BRepAdaptor_Curve(TopoDS.Edge_s(exp.Current()))
        count = 2 if curve.GetType() == GeomAbs_Line else n
        us = np.linspace(curve.FirstParameter(), curve.LastParameter(), count)
        lines.append([(curve.Value(u).X(), curve.Value(u).Y()) for u in us])
        exp.Next()
    return lines


def draw_lines(draw, lines, centre, scale, cam, dashed):
    _, right, up = cam
    cx, cy = centre @ right, centre @ up
    for line in lines:
        pts = [(IMG_SIZE / 2 + scale * (x - cx), IMG_SIZE / 2 - scale * (y - cy)) for x, y in line]
        for p, q in zip(pts, pts[1:]):
            if not dashed:
                draw.line([p, q], fill=0, width=1)
                continue
            length = max(np.hypot(q[0] - p[0], q[1] - p[1]), 1e-9)
            for s in np.arange(0, length, 8):           # 5 px dash, 3 px gap
                e = min(s + 5, length)
                draw.line([(p[0] + (q[0] - p[0]) * s / length, p[1] + (q[1] - p[1]) * s / length),
                           (p[0] + (q[0] - p[0]) * e / length, p[1] + (q[1] - p[1]) * e / length)], fill=110, width=1)


def drawing_view(shape, centre, scale, cam):
    """Engineering-style line drawing: visible edges solid, hidden edges dashed (hidden-line removal)."""
    w, right, _ = cam
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*w), gp_Dir(*right))))
    algo.Update()
    algo.Hide()
    hlr = HLRBRep_HLRToShape(algo)
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    draw = ImageDraw.Draw(img)
    draw_lines(draw, edge_polylines(hlr.HCompound()), centre, scale, cam, dashed=True)
    for visible in (hlr.VCompound(), hlr.OutLineVCompound(), hlr.Rg1LineVCompound()):
        draw_lines(draw, edge_polylines(visible), centre, scale, cam, dashed=False)
    return np.array(img)


def render_part(path, drawings=True):
    """STEP file -> dict of arrays, one entry per view in VIEWS order.
    drawings=False skips the line drawings (slow, and not needed for the face features)."""
    shape = read_step(path)
    corners, owner = triangulate(shape)
    centre, scale = fit(corners)
    shaded, lines, face_ids = [], [], []
    for name in VIEWS:
        cam = camera(name)
        image, face_id = render_mesh(corners, owner, centre, scale, cam)
        shaded.append(image)
        face_ids.append(face_id)
        if drawings:
            lines.append(drawing_view(shape, centre, scale, cam))
    result = {"shaded": np.stack(shaded), "face_id": np.stack(face_ids), "n_faces": len(get_face_list(shape))}
    if drawings:
        result["drawing"] = np.stack(lines)
    return result


def render_for_features(path):
    """render_part without line drawings; None if the part cannot be rendered (used by worker processes)."""
    try:
        return render_part(path, drawings=False)
    except Exception:
        return None


def visible_faces(face_ids):
    """Set of face indices (graph node numbers) seen in at least one view."""
    return set(np.unique(face_ids).tolist()) - {0}


def false_colour(face_id):
    rng = np.random.RandomState(1)
    palette = rng.randint(60, 255, size=(int(face_id.max()) + 1, 3)).astype(np.uint8)
    palette[0] = 255
    return palette[face_id]


def save_sheet(result, path):
    """One picture: shaded / face-id (with face numbers) / drawing for every view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(VIEWS)
    fig, axes = plt.subplots(3, n, figsize=(2.6 * n, 8))
    for j, name in enumerate(VIEWS):
        axes[0, j].imshow(result["shaded"][j])
        axes[1, j].imshow(false_colour(result["face_id"][j]))
        axes[2, j].imshow(result["drawing"][j], cmap="gray", vmin=0, vmax=255)
        axes[0, j].set_title(name)
        for k in visible_faces(result["face_id"][j]):
            ys, xs = np.nonzero(result["face_id"][j] == k)
            if len(xs) > 40:
                axes[1, j].text(xs.mean(), ys.mean(), f"F{k - 1}", fontsize=6, ha="center", va="center")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def self_test():
    """The default bracket has 14 faces; the views must agree with the graph about them."""
    path = os.path.join(DATA_DIR, "bracket.step")
    G = step_to_graph(path)
    result = render_part(path)
    assert result["n_faces"] == G.number_of_nodes() == 14, (result["n_faces"], G.number_of_nodes())
    seen = {k - 1 for k in visible_faces(result["face_id"])}
    assert seen == set(range(14)), f"faces never seen in any view: {set(range(14)) - seen}"

    # a flat face looking straight at a camera and fully visible (pixel count matches its area):
    # the centre of its pixels must be where its 3D centroid projects to
    checked = 0
    shape = read_step(path)
    corners, _ = triangulate(shape)
    centre, scale = fit(corners)
    for j, name in enumerate(VIEWS):
        cam = camera(name)
        for i, n in G.nodes(data=True):
            if n["type"] != "plane" or np.dot(n["direction"], cam[0]) < 0.99:
                continue
            ys, xs = np.nonzero(result["face_id"][j] == i + 1)
            if len(xs) == 0:
                continue
            if abs(len(xs) / (n["area"] * scale ** 2) - 1) > 0.1:
                continue
            px, py, _ = project(np.array(n["centroid"]), centre, scale, cam)
            off = np.hypot(xs.mean() + 0.5 - px, ys.mean() + 0.5 - py)
            assert off < 2, f"face {i} in view {name}: pixel centre is {off:.1f}px from its projected centroid"
            checked += 1
    assert checked >= 3, checked
    print(f"self-test: bracket -> 14 faces, all seen, {checked} centroid checks  OK")


def check(count=20, seed=0):
    self_test()
    files = sorted(os.listdir(MFCAD_TEST))
    files = random.Random(seed).sample(files, count)
    start, coverage = time.time(), []
    for f in files:
        path = os.path.join(MFCAD_TEST, f)
        result = render_part(path)
        assert result["n_faces"] == step_to_graph(path).number_of_nodes()
        coverage.append(len(visible_faces(result["face_id"])) / result["n_faces"])
    per_part = (time.time() - start) / count
    print(f"{count} MFCAD++ parts: faces seen in some view {100 * np.mean(coverage):.1f}% on average "
          f"(worst {100 * min(coverage):.1f}%), {per_part:.2f} s per part")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        step = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "bracket.step")
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(step))[0] + "_views.png")
        save_sheet(render_part(step), out)
        print("Saved", out)
