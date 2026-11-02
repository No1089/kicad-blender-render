#!/usr/bin/env python3
"""Dump every pad of a .kicad_pcb to JSON for the Blender solder generator.

Runs under the system python3 with KiCad's pcbnew module (KiCad 10).
Coordinates are in mm, KiCad board frame (x right, y DOWN), plus the board
centre so the Blender side can align with the GLB export (which is centred).
"""
import json
import math
import sys

import pcbnew


def mm(v):
    return pcbnew.ToMM(v)


SHAPES = {
    pcbnew.PAD_SHAPE_CIRCLE: "circle",
    pcbnew.PAD_SHAPE_RECTANGLE if hasattr(pcbnew, "PAD_SHAPE_RECTANGLE") else pcbnew.PAD_SHAPE_RECT: "rect",
    pcbnew.PAD_SHAPE_OVAL: "oval",
    pcbnew.PAD_SHAPE_ROUNDRECT: "roundrect",
    pcbnew.PAD_SHAPE_TRAPEZOID: "rect",
    pcbnew.PAD_SHAPE_CHAMFERED_RECT: "rect",
    pcbnew.PAD_SHAPE_CUSTOM: "custom",
}


def main(pcb_path, out_path):
    board = pcbnew.LoadBoard(pcb_path)
    bbox = board.GetBoardEdgesBoundingBox()
    centre = (mm(bbox.GetCenter().x), mm(bbox.GetCenter().y))
    outline = pcbnew.SHAPE_POLY_SET()
    board.GetBoardPolygonOutlines(outline, False)
    footprints = []
    pads = []
    for fp in board.GetFootprints():
        fpos = fp.GetPosition()
        footprints.append({
            "ref": fp.GetReference(), "value": fp.GetValue(),
            "side": "B" if fp.IsFlipped() else "F",
            "onboard": bool(outline.Contains(pcbnew.VECTOR2I(fpos.x, fpos.y))),
        })
        ref = fp.GetReference()
        dnp = bool(fp.IsDNP()) if hasattr(fp, "IsDNP") else False
        no_bom = bool(fp.IsExcludedFromBOM())
        fp_side = "B" if fp.IsFlipped() else "F"
        for pad in fp.Pads():
            lset = pad.GetLayerSet()
            attr = pad.GetAttribute()
            tht = attr in (pcbnew.PAD_ATTRIB_PTH,)
            npth = attr == pcbnew.PAD_ATTRIB_NPTH
            if npth:
                continue
            copper_sides = [s for s, l in (("F", pcbnew.F_Cu), ("B", pcbnew.B_Cu)) if lset.Contains(l)]
            paste_sides = [s for s, l in (("F", pcbnew.F_Paste), ("B", pcbnew.B_Paste)) if lset.Contains(l)]
            layer = pcbnew.F_Cu if lset.Contains(pcbnew.F_Cu) else pcbnew.B_Cu
            try:
                size = pad.GetSize(layer)
                shape = pad.GetShape(layer)
                rr = pad.GetRoundRectRadiusRatio(layer)
            except TypeError:
                size = pad.GetSize()
                shape = pad.GetShape()
                rr = pad.GetRoundRectRadiusRatio()
            drill = pad.GetDrillSize()
            pos = pad.GetPosition()
            # Polygon outline (copper, no clearance) for exact shape incl. custom pads
            poly = []
            try:
                sps = pcbnew.SHAPE_POLY_SET()
                pad.TransformShapeToPolygon(sps, layer, 0, pcbnew.FromMM(0.005), pcbnew.ERROR_INSIDE)
                if sps.OutlineCount():
                    ol = sps.Outline(0)
                    poly = [(mm(ol.CPoint(i).x), mm(ol.CPoint(i).y)) for i in range(ol.PointCount())]
            except Exception as e:  # pragma: no cover
                print("poly fail", ref, pad.GetNumber(), e, file=sys.stderr)
            pads.append({
                "ref": ref,
                "num": pad.GetNumber(),
                "fp_side": fp_side,
                "dnp": dnp,
                "no_bom": no_bom,
                "tht": tht,
                "copper": copper_sides,
                "paste": paste_sides,
                "shape": SHAPES.get(shape, "custom"),
                "x": mm(pos.x), "y": mm(pos.y),
                "w": mm(size.x), "h": mm(size.y),
                "rot": pad.GetOrientation().AsDegrees(),
                "rr": rr,
                "drill": [mm(drill.x), mm(drill.y)],
                "poly": poly,
                "fp_x": mm(fp.GetPosition().x), "fp_y": mm(fp.GetPosition().y),
            })
    # stackup colours / finish (parsed from the file: simplest across KiCad versions)
    import re
    txt = open(pcb_path, encoding="utf-8").read()
    stackup = {}
    for lname in ("F.Mask", "B.Mask", "F.SilkS", "B.SilkS"):
        m = re.search(r'\(layer "' + re.escape(lname) + r'"(.*?)\n\t\t\t\)', txt, re.S)
        if m:
            c = re.search(r'\(color "([^"]+)"\)', m.group(1))
            if c:
                stackup[lname] = c.group(1)
    f = re.search(r'\(copper_finish "([^"]+)"\)', txt)
    if f:
        stackup["finish"] = f.group(1)
    stack = board.GetDesignSettings()
    out = {
        "pcb": pcb_path,
        "centre": centre,
        "thickness": mm(stack.GetBoardThickness()),
        "pads": pads,
        "footprints": footprints,
        "stackup": stackup,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    n_paste = sum(1 for p in pads if p["paste"])
    off = [f["ref"] for f in footprints if not f["onboard"]]
    if off:
        print(f"{len(off)} footprints outside the board outline: {' '.join(off)}")
    print("stackup:", stackup)
    print(f"{len(pads)} pads ({n_paste} with paste, {sum(p['tht'] for p in pads)} THT) -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
