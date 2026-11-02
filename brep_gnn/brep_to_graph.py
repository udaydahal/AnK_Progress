"""Read a STEP file (a BRep) and turn it into a face adjacency graph.

  node  = one face of the part (flat, cylinder, ...) with a few numbers about it
  edge  = two faces that share a BRep edge, with the length and the convexity

Run:  python brep_to_graph.py [file.step]    (default: data/bracket.step)
Out:  a printed table and data/<name>_graph.png
"""
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import networkx as nx  
from OCP.BRep import BRep_Tool  
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface  
from OCP.BRepGProp import BRepGProp, BRepGProp_Face  
from OCP.GeomAbs import (  
    GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
    GeomAbs_Sphere, GeomAbs_Torus,
)
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf  
from OCP.gp import gp_Pnt, gp_Vec  
from OCP.GProp import GProp_GProps  
from OCP.IFSelect import IFSelect_RetDone  
from OCP.STEPControl import STEPControl_Reader  
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED  
from OCP.TopExp import TopExp, TopExp_Explorer  
from OCP.TopoDS import TopoDS  
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape  

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SURFACE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BSplineSurface: "bspline",
}
SMOOTH_ANGLE_DEG = 1.0  # faces meeting at less than this angle count as "smooth"


def read_step(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise SystemExit(f"Could not read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()


def _vec(v):
    return (v.X(), v.Y(), v.Z())


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def face_info(face):
    """Numbers describing one face: surface type, area, centroid, normal/axis."""
    surf = BRepAdaptor_Surface(face)
    kind = SURFACE_NAMES.get(surf.GetType(), "other")
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    info = {
        "type": kind,
        "area": round(props.Mass(), 3),
        "centroid": tuple(round(c, 3) for c in _vec(props.CentreOfMass())),
    }
    if kind == "plane":
        n = _vec(surf.Plane().Axis().Direction())
        if face.Orientation() == TopAbs_REVERSED:  # make it point out of the solid
            n = tuple(-c for c in n)
        info["direction"] = tuple(round(c, 3) for c in n)
    elif kind == "cylinder":
        cyl = surf.Cylinder()
        info["direction"] = tuple(round(c, 3) for c in _vec(cyl.Axis().Direction()))
        info["radius"] = round(cyl.Radius(), 3)
    return info


def _edge_midpoint_and_tangent(edge, face):
    """Point in the middle of `edge`, and its direction as the face boundary runs along it."""
    curve = BRepAdaptor_Curve(edge)
    u = 0.5 * (curve.FirstParameter() + curve.LastParameter())
    p, t = gp_Pnt(), gp_Vec()
    curve.D1(u, p, t)
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        if exp.Current().IsSame(edge):
            if exp.Current().Orientation() == TopAbs_REVERSED:
                t.Reverse()
            break
        exp.Next()
    return p, t.Normalized()


def _outward_normal(face, p):
    """Normal of `face` at the point of the face closest to `p`, pointing out of the solid."""
    proj = GeomAPI_ProjectPointOnSurf(p, BRep_Tool.Surface_s(face))
    u, v = proj.LowerDistanceParameters()
    pnt, vec = gp_Pnt(), gp_Vec()
    BRepGProp_Face(face).Normal(u, v, pnt, vec)
    return vec.Normalized()


def edge_convexity(edge, face1, face2):
    """('convex' | 'concave' | 'smooth', angle in degrees) for an edge between two faces."""
    p, t = _edge_midpoint_and_tangent(edge, face1)
    n1 = _vec(_outward_normal(face1, p))
    n2 = _vec(_outward_normal(face2, p))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, _dot(n1, n2)))))
    if angle < SMOOTH_ANGLE_DEG:
        return "smooth", angle
    # looking along the boundary of face1, face2 bends away (convex) or towards (concave)
    return ("convex" if _dot(_cross(n1, n2), _vec(t)) > 0 else "concave"), angle


def step_to_graph(path):
    """Build the face adjacency graph (a networkx.Graph) of the part in `path`."""
    shape = read_step(path)

    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, faces)
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)

    face_list = [TopoDS.Face_s(faces.FindKey(i)) for i in range(1, faces.Extent() + 1)]
    G = nx.Graph()
    for i, face in enumerate(face_list):
        G.add_node(i, **face_info(face))

    for k in range(1, edge_faces.Extent() + 1):
        edge = TopoDS.Edge_s(edge_faces.FindKey(k))
        touching = {faces.FindIndex(f) - 1 for f in edge_faces.FindFromIndex(k)}
        if len(touching) != 2:  # edge inside one face (e.g. cylinder seam): not a link
            continue
        a, b = sorted(touching)
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        kind, angle = edge_convexity(edge, face_list[a], face_list[b])
        if G.has_edge(a, b):  # two faces can share several edges: keep the total length
            G[a][b]["length"] = round(G[a][b]["length"] + props.Mass(), 3)
        else:
            G.add_edge(a, b, length=round(props.Mass(), 3), convexity=kind, angle=round(angle, 1))
    return G


def print_graph(G):
    print(f"\n{G.number_of_nodes()} faces (nodes), {G.number_of_edges()} links (edges)\n")
    print(f"{'face':<5}{'type':<10}{'area':>9}  {'direction':<20}{'radius':>7}  neighbours")
    for i, d in G.nodes(data=True):
        direction = str(d["direction"]) if "direction" in d else ""
        radius = d.get("radius", "")
        print(f"F{i:<4}{d['type']:<10}{d['area']:>9.1f}  {direction:<20}{radius!s:>7}  "
              + ", ".join(f"F{j}" for j in sorted(G[i])))
    print("\nface types:", {t: sum(1 for _, d in G.nodes(data=True) if d["type"] == t)
                            for t in sorted({d["type"] for _, d in G.nodes(data=True)})})
    print("edge kinds:", {k: sum(1 for *_, d in G.edges(data=True) if d["convexity"] == k)
                          for k in ("convex", "concave", "smooth")})


def plot_graph(G, path):
    colours = {"plane": "#9aa5b8", "cylinder": "#f0a04b"}
    styles = {"convex": ("black", "solid"), "concave": ("red", "solid"), "smooth": ("tab:blue", "dashed")}
    pos = nx.spring_layout(G, seed=3, k=1.2)
    fig, ax = plt.subplots(figsize=(9, 7))
    for kind, (colour, style) in styles.items():
        es = [(a, b) for a, b, d in G.edges(data=True) if d["convexity"] == kind]
        nx.draw_networkx_edges(G, pos, edgelist=es, edge_color=colour, style=style, width=1.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=900, ax=ax,
                           node_color=[colours.get(d["type"], "#cccccc") for _, d in G.nodes(data=True)])
    nx.draw_networkx_labels(G, pos, labels={i: f"F{i}\n{d['type']}" for i, d in G.nodes(data=True)},
                            font_size=7, ax=ax)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in colours.items()
    ] + [
        plt.Line2D([], [], color=c, ls=s, label=f"{k} edge") for k, (c, s) in styles.items()
    ], loc="upper left")
    ax.set_title("Face adjacency graph: node = face, line = shared edge")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "bracket.step")
    graph = step_to_graph(step)
    print_graph(graph)
    out = os.path.join(DATA_DIR, os.path.splitext(os.path.basename(step))[0] + "_graph.png")
    plot_graph(graph, out)
    print("\nSaved", out)
