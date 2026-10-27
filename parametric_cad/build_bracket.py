"""Build the parametric L-bracket in SolidWorks.

1. Edit the values at the top of bracket_model.py
2. Run:  python build_bracket.py
   SolidWorks is started if it is not already running.

Output (parametric_cad/output/): bracket.SLDPRT and bracket.png
"""
import inspect
import os
import sys

import win32com.client as wc
from win32com.client import gencache

import bracket_model as bm

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
NAME = "bracket"

MM = 1 / 1000.0  # the SolidWorks API works in meters

# SolidWorks type libraries (GUID, lcid, major, minor); major 33 = SolidWorks 2025.
# Early-bound wrappers are required: late-bound calls to FeatureExtrusion3 and
# friends silently return None.
SW_DIR = r"C:\Program Files\SOLIDWORKS Corp\SOLIDWORKS"
SLDWORKS_TLB = ("{83A33D31-27C5-11CE-BFD4-00400513BB57}", 0, 33, 0)

# SolidWorks constants used below
SW_DEFAULT_TEMPLATE_PART = 8      # swUserPreferenceStringValue_e.swDefaultTemplatePart
SW_END_BLIND = 0                  # swEndCondBlind
SW_END_THROUGH_ALL = 1            # swEndCondThroughAll
SW_END_MID_PLANE = 6              # swEndCondMidPlane
SW_SAVE_SILENT = 1                # swSaveAsOptions_Silent
SW_ISOMETRIC_VIEW = 7             # swIsometricView


def call(obj, name, *args):
    """obj.<name>(*args), whether COM exposes it as a method or a property."""
    v = getattr(obj, name)
    return v(*args) if inspect.isroutine(v) else v


def load_typelib():
    """Return the generated SolidWorks wrapper module (built once from sldworks.tlb)."""
    try:
        return gencache.GetModuleForTypelib(*SLDWORKS_TLB)
    except Exception:
        pass
    import win32com.client.makepy as makepy
    for tlb in ("sldworks.tlb", "swconst.tlb"):
        sys.argv = ["makepy", os.path.join(SW_DIR, tlb)]
        makepy.main()
    return gencache.GetModuleForTypelib(*SLDWORKS_TLB)


def connect():
    """Return (wrapper module, ISldWorks). Starts SolidWorks if needed."""
    sw = load_typelib()
    raw = wc.Dispatch("SldWorks.Application")
    app = sw.ISldWorks(raw._oleobj_)
    app.Visible = True
    return sw, app


def new_part(sw, app):
    template = call(app, "GetUserPreferenceStringValue", SW_DEFAULT_TEMPLATE_PART)
    if not template:
        sys.exit("No default part template set in SolidWorks (Tools > Options > Default Templates).")
    doc = call(app, "NewDocument", template, 0, 0, 0)
    if doc is None:
        sys.exit("SolidWorks could not create a new part.")
    return sw.IModelDoc2(doc._oleobj_)


def close_if_open(app, path):
    """Close `path` in SolidWorks if open (an open part keeps the file locked)."""
    doc = call(app, "GetOpenDocumentByName", path)
    if doc is not None:
        call(app, "CloseDoc", call(doc, "GetTitle"))


def reference_planes(model):
    """[Front, Top, Right] reference plane features (template order)."""
    planes = []
    feat = call(model, "FirstFeature")
    while feat is not None:
        if call(feat, "GetTypeName2") == "RefPlane":
            planes.append(feat)
        feat = call(feat, "GetNextFeature")
    if len(planes) < 3:
        sys.exit("The part template does not contain Front/Top/Right planes.")
    return planes[:3]


def select(feature):
    if not call(feature, "Select2", False, 0):
        sys.exit("Could not select a reference plane.")


def mm(v):
    return v * MM


def draw_profile(sk):
    for seg in bm.profile():
        if seg[0] == "line":
            (x1, y1), (x2, y2) = seg[1], seg[2]
            call(sk, "CreateLine", mm(x1), mm(y1), 0, mm(x2), mm(y2), 0)
        else:
            (cx, cy), (x1, y1), (x2, y2), direction = seg[1], seg[2], seg[3], seg[4]
            call(sk, "CreateArc", mm(cx), mm(cy), 0, mm(x1), mm(y1), 0, mm(x2), mm(y2), 0, direction)


def draw_circles(sk, centres):
    r = mm(bm.HOLE_DIA / 2)
    for a, b in centres:
        call(sk, "CreateCircleByRadius", mm(a), mm(b), 0, r)


def extrude_body(model):
    fm = call(model, "FeatureManager")
    feat = call(
        fm, "FeatureExtrusion3",
        True, False, False,                       # Sd, Flip, Dir
        SW_END_MID_PLANE, 0,                      # T1, T2
        mm(bm.BASE_WIDTH), 0,                     # D1, D2
        False, False, False, False,               # Dchk1, Dchk2, Ddir1, Ddir2
        0, 0,                                     # Dang1, Dang2
        False, False, False, False,               # OffsetReverse1/2, TranslateSurface1/2
        True, True, True,                         # Merge, UseFeatScope, UseAutoSelect
        0, 0, False,                              # T0, StartOffset, FlipStartOffset
    )
    if feat is None:
        sys.exit("Extrude failed (is the profile sketch closed?).")


def cut_through_all(model):
    fm = call(model, "FeatureManager")
    feat = call(
        fm, "FeatureCut4",
        False, False, False,                      # Sd (two directions), Flip, Dir
        SW_END_THROUGH_ALL, SW_END_THROUGH_ALL,   # T1, T2
        0, 0,                                     # D1, D2
        False, False, False, False,               # Dchk1, Dchk2, Ddir1, Ddir2
        0, 0,                                     # Dang1, Dang2
        False, False, False, False,               # OffsetReverse1/2, TranslateSurface1/2
        False, True, True,                        # NormalCut, UseFeatScope, UseAutoSelect
        True, True, False,                        # AssemblyFeatureScope, AutoSelectComponents, PropagateFeatureToParts
        0, 0, False,                              # T0, StartOffset, FlipStartOffset
        False,                                    # OptimizeGeometry
    )
    if feat is None:
        sys.exit("Hole cut failed.")


def sketch_on(model, plane):
    """Select `plane` and open a sketch on it; returns the SketchManager."""
    select(plane)
    sk = call(model, "SketchManager")
    call(sk, "InsertSketch", True)
    return sk


def build():
    bm.validate()
    sw, app = connect()
    model = new_part(sw, app)
    front, top, right = reference_planes(model)

    # 1. L-shaped body, centred on the Front plane
    sk = sketch_on(model, front)
    draw_profile(sk)
    call(sk, "InsertSketch", True)       # close the sketch
    # extrude uses the just-closed sketch, so re-select it
    call(model, "ClearSelection2", True)
    call(model, "Extension").SelectByID2("Sketch1", "SKETCH", 0, 0, 0, False, 0, None, 0)
    extrude_body(model)

    # 2. holes in the base plate (sketch on the Top plane, cut both ways)
    if bm.base_holes():
        sk = sketch_on(model, top)
        draw_circles(sk, bm.base_holes())
        call(sk, "InsertSketch", True)
        cut_through_all(model)

    # 3. holes in the flange (sketch on the Right plane)
    if bm.flange_holes():
        sk = sketch_on(model, right)
        draw_circles(sk, bm.flange_holes())
        call(sk, "InsertSketch", True)
        cut_through_all(model)

    # 4. save part + isometric PNG
    os.makedirs(OUT_DIR, exist_ok=True)
    part_path = os.path.join(OUT_DIR, NAME + ".SLDPRT")
    png_path = os.path.join(OUT_DIR, NAME + ".png")
    call(model, "ClearSelection2", True)  # drop the highlight on the last sketch
    call(model, "ShowNamedView2", "*Isometric", SW_ISOMETRIC_VIEW)
    call(model, "ViewZoomtofit2")
    close_if_open(app, part_path)
    for path in (part_path, png_path):
        if os.path.exists(path):
            os.remove(path)
    call(model, "SaveAs3", part_path, 0, SW_SAVE_SILENT)
    call(model, "SaveAs3", png_path, 0, SW_SAVE_SILENT)
    print("Saved", part_path)
    print("Saved", png_path)


if __name__ == "__main__":
    build()
