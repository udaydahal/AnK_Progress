# Parametric L-bracket (SolidWorks)

Edit a few numbers in code, run one command, get the part in SolidWorks.

## Use
1. Edit the PARAMETERS block at the top of `bracket_model.py`
   (length, width, thickness, flange height, fillet, hole size/count/edge offset).
2. `pip install -r requirements.txt` (once)
3. `python build_bracket.py` (SolidWorks starts if it is not running)

Output goes to `output/`: `bracket.SLDPRT` and `bracket.png` (isometric view).
Re-running overwrites them and closes the old copy in SolidWorks first.

## Files
- `bracket_model.py`  values + validation + geometry (no SolidWorks needed; `python bracket_model.py` prints the geometry)
- `build_bracket.py`  draws the L-profile, extrudes it (centred on the Front plane), cuts the holes, saves

## Notes
- Hole counts: base 0/1/2/4, flange 0/1/2. Bad values raise a clear error before SolidWorks is touched.
- Early-bound COM wrappers are generated once from `sldworks.tlb` (SolidWorks 2025, `SLDWORKS_TLB` in `build_bracket.py`).
  Late-bound calls to `FeatureExtrusion3` silently return nothing.
- Sketch quirk: SolidWorks only joins an arc and a line when the line comes first in the drawing chain, which is why the profile starts on the outside bend arc.
