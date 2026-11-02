# solder_joints

Realistic solder joint meshes for 3D PCB renders. Give it your pads and your component
meshes; it probes the actual 3D models above each pad and builds one smooth joint per pad
that behaves like reflowed solder:

- **SMD**: a concave meniscus from the terminal down to zero at the pad edge. Solder only
  climbs **metal** (end caps, leads, plated flanks), about 42 % of the terminal height, with
  a thin wetting layer over the rest of the pad. Plastic over a pad never pulls solder up;
  a small standoff gap under a leadless body (QFN/DFN/LGA) is filled; metal side flanks get a
  small toe fillet.
- **THT**: every point inside the hole is probed along the hole axis, so the real pin
  cross-section (round pin, flat blade, offset lead) shapes the joint: it clings to the pin
  and falls away concavely to the annular-ring edge. Big on the solder side, small on the
  component side.
- **HASL**: bare pads and empty holes get a thin, domed, slightly wavy levelled-tin coat.
- Optional **imperfection**: per-joint paste volume and gentle lumpiness, so a board doesn't
  look CAD-perfect.

The core is pure NumPy and independent of Blender (`solder_joints.core`); the Blender part
(`solder_joints.blender`) supplies a BVH-backed prober, mesh building and a SAC305 solder
material. It was written for [kicad-blender-render](../README.md) and designed to be dropped
into other KiCad→Blender tools such as pcb2blender.

## Use in Blender

```python
import numpy as np
from solder_joints import Pad, Settings, generate
from solder_joints.blender import BoardFrame, BVHProber, build_objects, solder_material

# board frame: KiCad mm (y down) -> Blender metres, copper surfaces at z_top / z_bottom
frame = BoardFrame(origin=(-0.1, 0.1), mm=0.001, z_top=0.0016, z_bottom=0.0, flip_y=True)

pads = [
    Pad(outline=np.array([[...], ...]),   # pad copper outline, mm, board frame
        center=(x, y), kind="smd",          # "smd" (has paste), "tht", or "bare"
        sides=[1],                          # +1 top, -1 bottom (THT: both copper sides)
        part_center=(fx, fy),               # footprint centre (leadless toe probes)
        component_side=1, drill=0.0, size=(w, h), name="R1.1"),
    ...
]
# which materials count as metal (solder wets them): names, or a callable(material) -> bool
prober = BVHProber(component_objects, metal_material_names, frame)
joints, report = generate(pads, prober, Settings(imperfection=0.5, hasl=False))
objects = build_objects(joints, frame, {"solder": solder_material(frame.mm)})
print(report.as_dict(), report.no_pin)
```

`BoardFrame` also accepts `xy=callable(x_mm, y_mm) -> (X, Y)` for any other mapping (e.g. one
fitted from an imported model), and handles face winding for mirrored frames and the bottom side.

## How it works

For each pad, a grid (~36 cells across the smallest pad dimension, max 140) is laid over the
pad outline. The prober answers three kinds of rays in the board frame (`h` = height above the
copper, outward on either side):

1. **Up** from the pad surface: what covers this point? Starting *inside* a part (models often
   sit on the laminate, below the raised pad) counts as covered; entering hits count within
   0.15 mm.
2. **Down** from above: is the topmost surface here metal? Then that is the terminal height
   (vendor models are messy: end caps are thin shells, leads overlap housings, so this is more
   robust than measuring the first solid).
3. **Sideways** towards the part, only when no metal is above the pad at all: a metal flank right
   next to the pad (leadless parts).

Covered metal points are the joint's source at one height per joint (80th percentile of the
terminal heights x the fraction, so chamfers don't make ridges). Every pad point takes its
nearest source and falls off as `(1 - t)^1.7` with `t = d_source / (d_source + d_edge)`, a
concave meniscus that reaches zero exactly at the pad outline, blended (smooth max) with a thin
wetting layer and lightly box-blurred like surface tension would. Only grid cells fully inside
the pad become faces, which avoids folds at the outline.

Nothing here simulates fluid; it's a fast, robust approximation (~0.05-0.2 s per pad).

## Tests

`python3 solder_joints/tests/test_core.py` (or pytest) runs the core against box-geometry
probers: end-cap fillet height and concavity, plastic doesn't pull solder up, leadless standoff
fill and flank toe, THT pin cone with component side, HASL rings and missing pins. No Blender
needed.

## Integrating into another tool (e.g. pcb2blender)

What the module needs from the host:

- per pad: copper outline polygon in board mm (or build one with `pad_outline(shape, size,
  position, rotation, roundness)`), centre, paste yes/no (→ `kind`), copper side(s), drill for
  THT, optionally the footprint centre (enables leadless toe probes) and side;
- the board-frame mapping into Blender and the copper surface heights;
- the component meshes in their final positions, and which materials are metal.

**pcb2blender** (checked against its `pcb2blender_exporter/export.py` and
`pcb2blender_importer/solder_joints.py`, Nov 2026): its `.pcb3d` already carries, per pad,
position, flipped side, has_model, has_paste, pad type (THT/SMD), shape, size, orientation,
roundrect ratio and drill shape/size. Its current joints are one parametric template mesh per
pad type/shape/size, copied onto every pad; they don't look at the component models. An adapter
would map each exported pad to `Pad(outline=pad_outline(...), kind="tht" | "smd" | "bare",
sides=[-1 if flipped else 1] (THT: both), drill=min(drill_size), ...)`, build a `BVHProber` over
the imported component objects (metal = its pin/metal materials), and call `generate` +
`build_objects`, e.g. as a new "Model-aware" choice next to its existing *Add Solder Joints*
option. The footprint centre isn't exported today; without it the leadless flank probe is
simply skipped.

## Licence

LGPL-3.0-or-later (`COPYING.LESSER` together with `COPYING`, the GPL-3.0 it extends).
Copyright (C) 2026 Chris Fourie (https://github.com/No1089).

You may use this module in any project, including GPL-3.0 ones such as pcb2blender and
non-free ones, provided changes to the module itself stay under the LGPL.
