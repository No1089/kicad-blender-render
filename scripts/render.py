"""KiCad GLB -> photoreal Cycles render, with generated solder joints.

Run headless:
  blender -b --factory-startup -P scripts/render.py -- \
      --glb out/xs.glb --pads out/xs_pads.json --preset hero --side F \
      --out out/xs_hero.png [--samples 256] [--res 2560x1600] [--focus U5] \
      [--fstop 8] [--backdrop dark|light] [--device auto|cpu|gpu] [--save-blend f.blend]

Units: the GLB is in metres. Everything board-related is computed in mm from the
pad JSON and mapped into Blender units (BU) via an automatically fitted transform.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: solder_joints
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

T0 = time.time()
MM = 0.001  # Blender units per mm, set from the board fit
REFS = set()
METALS = set()  # component material names classified as metal/gold


def log(*a):
    print(f"[pcb-render {time.time() - T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- arguments
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--pads", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shots", default=None,
                    help="JSON file: list of per-shot option dicts (preset, side, focus, zoom, out, ...) "
                         "rendered in one session; options given on the command line are the defaults")
    ap.add_argument("--preset", default="hero", choices=["hero", "top", "macro", "low"])
    ap.add_argument("--side", default="F", choices=["F", "B"])
    ap.add_argument("--focus", default=None, help="reference designator for macro/hero focus")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--res", default="2560x1600")
    ap.add_argument("--fstop", type=float, default=None)
    ap.add_argument("--azimuth", type=float, default=None, help="camera azimuth deg (0 = from front/-Y)")
    ap.add_argument("--elevation", type=float, default=None, help="camera elevation deg")
    ap.add_argument("--backdrop", default="dark", choices=["dark", "light", "none"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu", "both"],
                    help="auto: GPU if available; both: GPU + CPU together (helps on APUs)")
    ap.add_argument("--no-solder", action="store_true")
    ap.add_argument("--keep-offboard", action="store_true",
                    help="keep footprints placed outside the board outline (hidden by default)")
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--imperfection", type=float, default=0.6,
                    help="0 = CAD-perfect, 1 = rough hand-built look (placement skew, solder volume, smudges)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--viewport", default=None,
                    help="name of a 3D viewport saved in KiCad's 3D viewer (stored in the .kicad_pro)")
    ap.add_argument("--zoom", type=float, default=None,
                    help="framing multiplier: <1 = tighter (default 1, macro with --focus 2.2)")
    ap.add_argument("--lens", type=float, default=None, help="focal length mm")
    ap.add_argument("--hdri", default="studio_small_09",
                    help="studio HDRI name in assets/hdri (Poly Haven, CC0) or a path; 'builtin' = Blender's")
    ap.add_argument("--env-strength", type=float, default=None)
    ap.add_argument("--save-blend", default=None)
    return ap.parse_args(argv)


# ---------------------------------------------------------------- materials
def principled(mat):
    if not mat.use_nodes:
        mat.use_nodes = True
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    n = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    out = next(x for x in mat.node_tree.nodes if x.type == "OUTPUT_MATERIAL")
    mat.node_tree.links.new(n.outputs[0], out.inputs[0])
    return n


def setp(bsdf, name, val):
    if name in bsdf.inputs:
        bsdf.inputs[name].default_value = val


def clear_links(mat, bsdf, names):
    for name in names:
        if name in bsdf.inputs:
            for l in list(bsdf.inputs[name].links):
                mat.node_tree.links.remove(l)


def new_mat(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(b.outputs[0], out.inputs[0])
    return m, b


def noise_node(mat, per_mm, detail=8.0, seed_offset=0.0):
    """Noise texture in object space with features ~1/per_mm mm across."""
    nt = mat.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = per_mm / MM
    if "Detail" in noise.inputs:
        noise.inputs["Detail"].default_value = detail
    if "W" in noise.inputs and seed_offset:
        noise.noise_dimensions = "4D"
        noise.inputs["W"].default_value = seed_offset
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    return noise


def drive_range(mat, src, sock, lo, hi):
    nt = mat.node_tree
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = lo
    mr.inputs["To Max"].default_value = hi
    nt.links.new(src, mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], sock)


def drive_color(mat, src, sock, c1, c2):
    nt = mat.node_tree
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = c1
    mix.inputs["B"].default_value = c2
    nt.links.new(src, mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], sock)


def add_noise_bump(mat, bsdf, scale, strength, distance, roughness_var=0.0, base_rough=0.3):
    """Micro-surface variation: noise -> bump on normal, optional roughness variation.
    scale is features per mm."""
    nt = mat.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale / MM
    if "Detail" in noise.inputs:
        noise.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if roughness_var > 0:
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = base_rough - roughness_var
        mr.inputs["To Max"].default_value = base_rough + roughness_var
        nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])


# Soldermask colours (linear RGB): (over bare laminate, over copper). LPI mask is a thin
# translucent film, so copper underneath shows as a lighter/warmer shade (tracks visible).
MASK_COLOURS = {
    "black": ((0.006, 0.006, 0.007), (0.011, 0.010, 0.009)),
    "matte black": ((0.008, 0.008, 0.009), (0.012, 0.011, 0.010)),
    "green": ((0.005, 0.045, 0.014), (0.016, 0.092, 0.020)),   # JLCPCB green: deep bottle green
    "red": ((0.22, 0.008, 0.008), (0.32, 0.04, 0.015)),
    "blue": ((0.004, 0.025, 0.16), (0.02, 0.07, 0.20)),
    "white": ((0.72, 0.72, 0.70), (0.68, 0.66, 0.58)),
    "yellow": ((0.55, 0.40, 0.015), (0.62, 0.46, 0.04)),
    "purple": ((0.10, 0.015, 0.17), (0.16, 0.04, 0.20)),
}
SILK_COLOURS = {"white": (0.84, 0.84, 0.82), "black": (0.02, 0.02, 0.02), "yellow": (0.70, 0.55, 0.05)}


def srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def mask_colours(name):
    # no colour in the stackup: KiCad's and the fabs' default is green
    n = (name or "green").strip().lower()
    if n in MASK_COLOURS:
        return MASK_COLOURS[n]
    if n.startswith("#") and len(n) >= 7:
        lam = tuple(srgb_to_lin(int(n[i:i + 2], 16) / 255) for i in (1, 3, 5))
        cu = tuple(min(1.0, v * 1.5 + w) for v, w in zip(lam, (0.02, 0.012, 0.004)))
        return lam, cu
    log(f"unknown mask colour {name!r}, using black")
    return MASK_COLOURS["black"]


def finish_material(finish, mm_to_bu, imp):
    """Exposed copper finish from the stackup's copper_finish."""
    f = (finish or "ENIG").lower()
    m, b = new_mat("Finish")
    setp(b, "Metallic", 1.0)
    if "hal" in f or "hasl" in f:
        # hot-air levelled solder: bright tin, slightly uneven (levelling ripples)
        add_noise_bump(m, b, scale=3.0, strength=0.25 + 0.3 * imp, distance=0.006 * mm_to_bu,
                       roughness_var=0.04, base_rough=0.16)
        setp(b, "Base Color", (0.80, 0.80, 0.82, 1))
        kind = "HASL"
    elif "silver" in f:
        add_noise_bump(m, b, scale=40.0, strength=0.05, distance=0.002 * mm_to_bu,
                       roughness_var=0.04, base_rough=0.2)
        setp(b, "Base Color", (0.93, 0.93, 0.92, 1))
        kind = "immersion silver"
    elif "tin" in f:
        add_noise_bump(m, b, scale=40.0, strength=0.08, distance=0.002 * mm_to_bu,
                       roughness_var=0.05, base_rough=0.35)
        setp(b, "Base Color", (0.74, 0.74, 0.74, 1))
        kind = "immersion tin"
    elif "osp" in f or f in ("none", "not specified", ""):
        add_noise_bump(m, b, scale=40.0, strength=0.06, distance=0.002 * mm_to_bu,
                       roughness_var=0.06, base_rough=0.28)
        drive_color(m, noise_node(m, 0.6, 3).outputs["Fac"], b.inputs["Base Color"],
                    (0.95, 0.60, 0.48, 1), (0.85 - 0.1 * imp, 0.50, 0.40, 1))
        kind = "bare copper / OSP"
    else:
        # ENIG / ENEPIG / hard gold: warm, slightly matte, fine grain, faint batch variation
        add_noise_bump(m, b, scale=40.0, strength=0.06 + 0.1 * imp,
                       distance=0.002 * mm_to_bu, roughness_var=0.04 + 0.06 * imp, base_rough=0.24)
        drive_color(m, noise_node(m, 0.6, 3).outputs["Fac"], b.inputs["Base Color"],
                    (1.0, 0.76, 0.40, 1), (1.0 - 0.06 * imp, 0.70 - 0.04 * imp, 0.38, 1))
        kind = "ENIG"
    log(f"copper finish: {finish!r} -> {kind}")
    return m


def make_materials(mm_to_bu, imp, stackup=None):
    stackup = stackup or {}
    mats = {}
    mats["enig"] = finish_material(stackup.get("finish"), mm_to_bu, imp)

    # Solder (SAC305 after reflow): bright but slightly frosty, soft ripples
    m, b = new_mat("Solder")
    setp(b, "Base Color", (0.80, 0.80, 0.82, 1))
    setp(b, "Metallic", 1.0)
    add_noise_bump(m, b, scale=5.0, strength=0.015 + 0.03 * imp,
                   distance=0.004 * mm_to_bu, roughness_var=0.03 + 0.07 * imp, base_rough=0.12 + 0.1 * imp)
    mats["solder"] = m

    # Black LPI soldermask: glossy with a coat, but smudged: low-frequency roughness patches
    m, b = new_mat("Soldermask")
    setp(b, "Base Color", (0.006, 0.006, 0.007, 1))
    setp(b, "Coat Weight", 0.6)
    setp(b, "IOR", 1.55)
    add_noise_bump(m, b, scale=1.8, strength=0.25 + 0.5 * imp, distance=0.004 * mm_to_bu)  # orange peel
    smudge = noise_node(m, 0.35, 4, seed_offset=3.7).outputs["Fac"]
    drive_range(m, smudge, b.inputs["Roughness"], 0.3, 0.3 + 0.12 * imp)
    drive_range(m, smudge, b.inputs["Coat Roughness"], 0.17, 0.17 + 0.12 * imp)
    mats["mask"] = m

    # Silkscreen: matte white epoxy ink, slightly grainy and uneven
    m, b = new_mat("Silkscreen")
    setp(b, "Subsurface Weight", 0.1)
    add_noise_bump(m, b, scale=25.0, strength=0.1 + 0.3 * imp, distance=0.004 * mm_to_bu,
                   roughness_var=0.1, base_rough=0.65)
    sc_ = SILK_COLOURS.get((stackup.get("F.SilkS") or "white").lower(), SILK_COLOURS["white"])
    drive_color(m, noise_node(m, 3.0, 4, seed_offset=1.3).outputs["Fac"], b.inputs["Base Color"],
                (*sc_, 1), (sc_[0] * (1 - 0.14 * imp), sc_[1] * (1 - 0.14 * imp), sc_[2] * (1 - 0.16 * imp), 1))
    mats["silk"] = m

    # FR4 core (seen at edges and holes)
    m, b = new_mat("FR4")
    setp(b, "Base Color", (0.42, 0.38, 0.2, 1))
    setp(b, "Roughness", 0.55)
    setp(b, "Subsurface Weight", 0.15)
    mats["fr4"] = m
    return mats


# STEP colours (sRGB, as written in the model) that mean "satin / bead-blasted nickel", e.g. the
# magnet ring of a magnetic pogo-pin connector model (its generator script uses this marker).
SATIN_METAL_RGB = [(0.74, 0.74, 0.73)]
# KiCad's stock library colours that the hue rules below miss: "gold pins" on every
# PinHeader / PinSocket / connector model (saturation ~0.42, same band as tan nylon).
GOLD_PIN_RGB = [(0.859, 0.738, 0.496)]
# STEP colours that mean "solder mask on a module PCB" (glossy coat over the colour), used by
# the module generators (Teensy green, Adafruit blue, generic green). sRGB as in the STEP.
MODULE_MASK_RGB = [(0.05, 0.22, 0.62), (0.06, 0.30, 0.14), (0.04, 0.36, 0.20), (0.02, 0.30, 0.47)]  # Adafruit, PJRC, generic green, Raytac teal


def _srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def material_body_fractions(objs):
    """For each material: the largest share of any single part's surface it covers.
    Pins/contacts are small shares; a moulded body is most of the part."""
    per_part = {}
    for o in objs:
        part = o
        while part.parent is not None and part.parent.name.split(".")[0] in REFS:
            part = part.parent
        key = part.name.split(".")[0] if part.name.split(".")[0] in REFS else o.name
        me = o.data
        n = len(me.polygons)
        if n == 0:
            continue
        area = np.empty(n)
        mi = np.empty(n, dtype=np.int64)
        me.polygons.foreach_get("area", area)
        me.polygons.foreach_get("material_index", mi)
        d = per_part.setdefault(key, {})
        for k, sl in enumerate(o.material_slots):
            if sl.material:
                d[sl.material.name] = d.get(sl.material.name, 0.0) + float(area[mi == k].sum())
    frac, big = {}, {}
    for d in per_part.values():
        tot = sum(d.values()) or 1.0
        for name, a in d.items():
            frac[name] = max(frac.get(name, 0.0), a / tot)
            # "big sheet metal": most of a part AND physically large (shield, RJ45/USB shell),
            # not the end caps of a 0402 (also most of its tiny surface)
            if a / tot > 0.4 and a / (MM * MM) > 3.0:
                big[name] = True
    return frac, big


def classify_component_material(mat, body_frac=0.0, big_sheet=False):
    """Upgrade KiCad's flat STEP colours to plausible PBR materials."""
    b = principled(mat)
    col = b.inputs["Base Color"].default_value
    r, g, bl = col[0], col[1], col[2]
    alpha = col[3] if len(col) > 3 else 1.0
    mx, mn = max(r, g, bl), min(r, g, bl)
    sat = 0 if mx == 0 else (mx - mn) / mx
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * bl
    if "Alpha" in b.inputs and not b.inputs["Alpha"].links:
        alpha = min(alpha, b.inputs["Alpha"].default_value)
    if alpha < 0.95:
        setp(b, "Roughness", 0.05)
        setp(b, "Transmission Weight", 0.9)
        return "glass"
    # Marker colours (glTF colours are linear; compare in sRGB)
    srgb = (_srgb(r), _srgb(g), _srgb(bl))
    def near(lst):
        return any(max(abs(a - b_) for a, b_ in zip(srgb, m)) < 0.006 for m in lst)
    if near(GOLD_PIN_RGB):
        setp(b, "Metallic", 1.0)
        setp(b, "Base Color", (1.0, 0.76, 0.38, 1))
        setp(b, "Roughness", 0.22)
        return "gold"
    if near(MODULE_MASK_RGB):
        # LPI mask: coloured, satin base under a thin glossy clear coat
        setp(b, "Metallic", 0.0)
        clear_links(mat, b, ["Roughness", "Normal"])
        add_noise_bump(mat, b, scale=6.0, strength=0.05, distance=0.003 * MM,
                       roughness_var=0.05, base_rough=0.38)
        setp(b, "Coat Weight", 0.7)
        setp(b, "Coat Roughness", 0.12)
        return "plastic"
    if near(SATIN_METAL_RGB):
        setp(b, "Metallic", 1.0)
        setp(b, "Base Color", (0.78, 0.77, 0.74, 1))
        clear_links(mat, b, ["Roughness", "Normal"])
        # fine grain: bead-blast sparkle + slight roughness variation
        add_noise_bump(mat, b, scale=30.0, strength=0.35, distance=0.004 * MM,
                       roughness_var=0.08, base_rough=0.30)
        return "metal"
    # Gold-ish hue (pins, plated contacts)
    # (sat > 0.55: tan nylon housings and ceramic bodies sit around 0.45 and must stay plastic)
    # ... and only when it is a small part of a component (pins, contacts). A gold-coloured
    # material covering most of a part is a yellow moulded body (tantalum/polymer caps).
    if sat > 0.55 and r > 0.45 and g > 0.25 and bl < 0.25 and r >= g and body_frac < 0.4:
        setp(b, "Metallic", 1.0)
        setp(b, "Base Color", (1.0, 0.76, 0.38, 1))
        setp(b, "Roughness", 0.2)
        return "gold"
    # Pale, slightly yellow (flash-gold / plated flanks of leadless packages)
    # (only small parts of a component: an off-white ceramic/alumina body is not plating)
    if 0.12 <= sat < 0.3 and lum > 0.55 and r >= g >= bl and body_frac < 0.4:
        setp(b, "Metallic", 1.0)
        setp(b, "Base Color", (min(1, r * 1.1), min(1, g * 1.1), min(1, bl * 1.15), 1))
        setp(b, "Roughness", 0.25)
        return "metal"
    # Light, desaturated grey -> tin / nickel plated metal
    if sat < 0.12 and 0.3 < lum < 0.9:
        setp(b, "Metallic", 1.0)
        setp(b, "Base Color", (min(1, r * 1.1), min(1, g * 1.1), min(1, bl * 1.1), 1))
        if big_sheet:
            # big sheet-metal parts (RJ45/USB shells, shield cans, crystal cans): stamped,
            # tin-plated steel is satin with a fine grain, not a mirror
            clear_links(mat, b, ["Roughness", "Normal"])
            add_noise_bump(mat, b, scale=60.0, strength=0.12, distance=0.002 * MM,
                           roughness_var=0.06, base_rough=0.4)
            return "satin"
        setp(b, "Roughness", 0.3)
        return "metal"
    # Everything else: plastic / ceramic / epoxy. Mould compound is matte with a fine texture.
    setp(b, "Metallic", 0.0)
    setp(b, "Coat Weight", 0.0)
    clear_links(mat, b, ["Roughness", "Normal"])
    add_noise_bump(mat, b, scale=40.0, strength=0.04, distance=0.002 * MM,
                   roughness_var=0.06, base_rough=0.62 if lum < 0.1 else 0.5)
    return "plastic"


# ---------------------------------------------------------------- geometry helpers
def world_verts(obj):
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    M = np.array(obj.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


def build_bvh(objs, metal_names=None):
    """BVH over the objects' triangles. With metal_names, also returns a per-triangle
    bool array: True where the triangle's material is metal (solder wets it)."""
    verts, polys, flags, off = [], [], [], 0
    for o in objs:
        v = world_verts(o)
        me = o.data
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tri = np.empty(nt * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        verts.append(v)
        polys.append(tri.reshape(nt, 3) + off)
        off += len(v)
        if metal_names is not None:
            mi = np.empty(nt, dtype=np.int64)
            me.loop_triangles.foreach_get("material_index", mi)
            slot_metal = np.array([bool(sl.material and sl.material.name in metal_names)
                                   for sl in o.material_slots] or [False])
            flags.append(slot_metal[np.clip(mi, 0, len(slot_metal) - 1)])
    if not verts:
        return (None, None) if metal_names is not None else None
    V = np.concatenate(verts)
    P = np.concatenate(polys)
    bvh = BVHTree.FromPolygons(V.tolist(), P.tolist(), all_triangles=True)
    if metal_names is not None:
        return bvh, np.concatenate(flags)
    return bvh


class BoardMap:
    """Maps KiCad mm (x right, y down) into Blender world coordinates."""

    def __init__(self, pads_json, pad_objs):
        P = [p for p in pads_json.get("all_pads", pads_json["pads"]) if p["copper"] and p["poly"]]
        allpts = np.array([pt for p in P for pt in p["poly"]])
        jmin, jmax = allpts.min(0), allpts.max(0)
        V = np.concatenate([world_verts(o) for o in pad_objs])
        bmin, bmax = V.min(0), V.max(0)
        self.s = (bmax[0] - bmin[0]) / (jmax[0] - jmin[0])  # BU per mm
        sy = (bmax[1] - bmin[1]) / (jmax[1] - jmin[1])
        log(f"scale BU/mm x={self.s:.6g} y={sy:.6g}")
        self.jmin, self.jmax, self.bmin, self.bmax = jmin, jmax, bmin, bmax
        self.z_top = V[:, 2].max()
        self.z_bot = V[:, 2].min()
        bvh = build_bvh(pad_objs)
        best = None
        sample = random.Random(1).sample(P, min(60, len(P)))
        for flip in (True, False):
            self.flip = flip
            hits = 0
            for p in sample:
                # through-hole pads: aim at the copper ring, not into the hole
                dx = 0.5 * (0.5 * min(p["drill"]) + 0.5 * min(p["w"], p["h"])) if p.get("tht") else 0.0
                x, y = self.xy(np.array([p["x"] + dx]), np.array([p["y"]]))
                o = Vector((x[0], y[0], self.z_top + 0.01))
                if bvh.ray_cast(o, Vector((0, 0, -1)), 0.1)[0] is not None:
                    hits += 1
                o = Vector((x[0], y[0], self.z_bot - 0.01))
                if bvh.ray_cast(o, Vector((0, 0, 1)), 0.1)[0] is not None:
                    hits += 1
            log(f"alignment flip_y={flip}: {hits}/{2 * len(sample)} pad hits")
            if best is None or hits > best[1]:
                best = (flip, hits)
        self.flip = best[0]

    def xy(self, x, y):
        bx = (x - self.jmin[0]) * self.s + self.bmin[0]
        if self.flip:
            by = (self.jmax[1] - y) * self.s + self.bmin[1]
        else:
            by = (y - self.jmin[1]) * self.s + self.bmin[1]
        return bx, by


def cull_offboard_faces(layer_objs, bmap):
    """Delete board-layer faces (pads of parked parts etc.) that don't sit over the PCB body."""
    import bmesh
    body = build_bvh(layer_objs["PCB"])
    reach = 0.3 * bmap.s  # mm above/below the laminate surface
    total = 0
    for key in ("pad", "copper", "via", "silkscreen", "soldermask"):
        for o in layer_objs.get(key, []):
            M = o.matrix_world
            bm = bmesh.new()
            bm.from_mesh(o.data)
            bad = []
            for f in bm.faces:
                c = M @ f.calc_center_median()
                hit = False
                for dz in (-1, 1):
                    o2 = Vector((c.x, c.y, c.z - dz * 0.001 * bmap.s))
                    loc = body.ray_cast(o2, Vector((0, 0, dz)), reach + (bmap.z_top - bmap.z_bot))[0]
                    if loc is not None:
                        hit = True
                        break
                if not hit:
                    bad.append(f)
            if bad:
                bmesh.ops.delete(bm, geom=bad, context="FACES")
                bm.to_mesh(o.data)
                total += len(bad)
            bm.free()
    log(f"culled {total} off-board layer faces")


def jitter_placement(pads_json, imp, rng):
    """Real reflow leaves parts slightly skewed/offset; THT parts stay put."""
    if imp <= 0:
        return
    npads, tht = {}, set()
    for p in pads_json["pads"]:
        if p["copper"]:
            npads[p["ref"]] = npads.get(p["ref"], 0) + 1
        if p["tht"]:
            tht.add(p["ref"])
    moves = {}
    for ref, n in sorted(npads.items()):
        if ref in tht:
            continue
        small = n <= 4
        rot = math.radians(rng.normal(0, (1.6 if small else 0.4) * imp))
        sh = rng.normal(0, (0.035 if small else 0.015) * imp, 2)
        moves[ref] = (rot, sh)
    count = 0
    for o in bpy.context.scene.objects:
        ref = o.name.split(".")[0]
        if ref in moves and (o.parent is None or o.parent.name.split(".")[0] not in moves):
            rot, sh = moves[ref]
            piv = o.matrix_world.translation.copy()
            M = (Matrix.Translation(piv + Vector((sh[0] * MM, sh[1] * MM, 0)))
                 @ Matrix.Rotation(rot, 4, "Z") @ Matrix.Translation(-piv))
            o.matrix_world = M @ o.matrix_world
            count += 1
    bpy.context.view_layer.update()
    log(f"placement jitter on {count} parts")


def box_blur(a, r, passes=3):
    """Separable box blur (3 passes ~ gaussian) of a 2D array, radius r px."""
    if r < 1:
        return a
    for _ in range(passes):
        for ax in (0, 1):
            c = np.cumsum(np.pad(a, [(r + 1, r) if i == ax else (0, 0) for i in range(2)], mode="edge"), axis=ax)
            hi = np.take(c, np.arange(2 * r + 1, c.shape[ax]), axis=ax)
            lo = np.take(c, np.arange(0, c.shape[ax] - 2 * r - 1), axis=ax)
            a = (hi - lo) / (2 * r + 1)
    return a


def copper_relief(layer_objs, bmap, out_dir, imp, px_mm=0.015, cache_key=None):
    """Rasterise each side's copper (tracks, pours, vias) from above with Workbench, blur it,
    and use it to raise the soldermask: real LPI mask shows the copper underneath as soft relief."""
    s = bmap.s
    pcb = layer_objs["PCB"][0]
    V = world_verts(pcb)
    (x0, y0), (x1, y1) = V[:, :2].min(0), V[:, :2].max(0)
    zmid = (bmap.z_top + bmap.z_bot) / 2
    W, H = x1 - x0, y1 - y0
    px_mm = max(px_mm, max(W, H) / s / 4096)  # cap the long side at 4096 px
    rx, ry = int(math.ceil(W / (px_mm * s))), int(math.ceil(H / (px_mm * s)))

    sc = bpy.data.scenes.new("relief")
    for key in ("copper", "via"):
        for o in layer_objs.get(key, []):
            sc.collection.objects.link(o)
    w = bpy.data.worlds.new("relief_bg")
    w.color = (0, 0, 0)
    sc.world = w
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "FLAT"
    sh.color_type = "SINGLE"
    sh.single_color = (1, 1, 1)
    sc.view_settings.view_transform = "Standard"
    sc.render.resolution_x, sc.render.resolution_y = rx, ry
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "BW"
    sc.render.filter_size = 0.0
    cd = bpy.data.cameras.new("relief_cam")
    cd.type = "ORTHO"
    cd.ortho_scale = max(W, H)
    cam = bpy.data.objects.new("relief_cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cz = bmap.z_top + 1.0 * s
    cam.location = ((x0 + x1) / 2, (y0 + y1) / 2, cz)
    split = cz - zmid

    imgs = {}
    cache = {sd: os.path.join(out_dir, f"relief_{sd}_blur.png") for sd in ("F", "B")}
    stamp = os.path.join(out_dir, "relief.key")
    if cache_key and all(os.path.exists(c) for c in cache.values()) and os.path.exists(stamp) \
            and open(stamp).read() == f"{cache_key} {rx}x{ry}":
        for sd in ("F", "B"):
            img = bpy.data.images.load(cache[sd], check_existing=False)
            img.colorspace_settings.name = "Non-Color"
            imgs[sd] = img
        bpy.data.scenes.remove(sc)
        log(f"copper relief: cached {rx}x{ry} px")
        rx = None  # skip rendering below
    for side in (("F", "B") if rx else ()):
        if side == "F":
            cd.clip_start, cd.clip_end = 0.01 * s, split
        else:
            cd.clip_start, cd.clip_end = split, split + 5 * s
        # per-process raw render (other renders may run at the same time on the same board)
        path = os.path.join(out_dir, f".relief_{side}.{os.getpid()}.png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True, scene=sc.name)
        img = bpy.data.images.load(path, check_existing=False)
        a = np.empty(rx * ry * 4, dtype=np.float32)
        img.pixels.foreach_get(a)
        a = a.reshape(ry, rx, 4)[:, :, 0].astype(np.float64)
        a = box_blur(a, max(1, int(round(0.03 / px_mm))))  # mask flows ~30 um over edges
        out = np.repeat(a[:, :, None], 4, axis=2).astype(np.float32)
        out[:, :, 3] = 1
        bpy.data.images.remove(img)
        try:
            os.remove(path)
        except OSError:
            pass
        img = bpy.data.images.new(f"relief_{side}", rx, ry, alpha=False, is_data=True)
        img.pixels.foreach_set(out.ravel())
        tmp = f"{cache[side]}.{os.getpid()}.png"
        img.filepath_raw = tmp
        img.file_format = "PNG"
        img.save()
        os.replace(tmp, cache[side])  # atomic: readers see the old or the new map, never half
        img.filepath_raw = cache[side]
        img.colorspace_settings.name = "Non-Color"
        imgs[side] = img
        log(f"copper relief {side}: {rx}x{ry} px, coverage {a.mean() * 100:.0f} %")
    if rx:
        bpy.data.scenes.remove(sc)
        if cache_key:
            tmp = f"{stamp}.{os.getpid()}"
            open(tmp, "w").write(f"{cache_key} {rx}x{ry}")
            os.replace(tmp, stamp)

    # UVs on the mask meshes = board XY in the relief image frame (pre-flip world coords)
    for o in layer_objs.get("soldermask", []):
        Vw = world_verts(o)
        me = o.data
        uvl = me.uv_layers.get("relief") or me.uv_layers.new(name="relief")
        vi = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", vi)
        u = (Vw[vi, 0] - (x0 + x1) / 2) / max(W, H) * (max(W, H) / W) + 0.5
        v = (Vw[vi, 1] - (y0 + y1) / 2) / max(W, H) * (max(W, H) / H) + 0.5
        uvl.data.foreach_set("uv", np.stack([u, v], 1).ravel().astype(np.float32))
    return imgs


def mask_material_for_side(base_mat, img, imp, name, colours):
    """Copy of the soldermask material with copper relief, tint over copper and dust."""
    m = base_mat.copy()
    m.name = name
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "relief"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Cubic"
    tex.extension = "EXTEND"
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.035 * MM  # copper (35 um) showing through the mask
    nt.links.new(tex.outputs["Color"], bump.inputs["Height"])
    prev = b.inputs["Normal"].links[0].from_socket if b.inputs["Normal"].links else None
    if prev is not None:
        nt.links.new(prev, bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    if "Coat Normal" in b.inputs:
        nt.links.new(bump.outputs["Normal"], b.inputs["Coat Normal"])
    # mask over copper reads a touch lighter/warmer; sparse dust specks
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.inputs["A"].default_value = (*colours[0], 1)
    tint.inputs["B"].default_value = (*colours[1], 1)
    nt.links.new(tex.outputs["Color"], tint.inputs["Factor"])
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 2.5 / MM
    gtc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(gtc.outputs["Object"], vor.inputs["Vector"])
    small = nt.nodes.new("ShaderNodeMath")
    small.operation = "LESS_THAN"
    small.inputs[1].default_value = 0.045
    nt.links.new(vor.outputs["Distance"], small.inputs[0])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(vor.outputs["Color"], sep.inputs["Color"])
    rare = nt.nodes.new("ShaderNodeMath")
    rare.operation = "GREATER_THAN"
    rare.inputs[1].default_value = 1.0 - 0.12 * imp
    nt.links.new(sep.outputs[0], rare.inputs[0])
    dust = nt.nodes.new("ShaderNodeMath")
    dust.operation = "MULTIPLY"
    nt.links.new(small.outputs[0], dust.inputs[0])
    nt.links.new(rare.outputs[0], dust.inputs[1])
    dcol = nt.nodes.new("ShaderNodeMix")
    dcol.data_type = "RGBA"
    nt.links.new(dust.outputs[0], dcol.inputs["Factor"])
    nt.links.new(tint.outputs["Result"], dcol.inputs["A"])
    dcol.inputs["B"].default_value = (0.22, 0.21, 0.19, 1)
    nt.links.new(dcol.outputs["Result"], b.inputs["Base Color"])
    if os.environ.get("PCB_RENDER_DEBUG_RELIEF"):
        nt.links.new(tex.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = 2.0
    if b.inputs["Coat Weight"].links:
        pass
    cw = nt.nodes.new("ShaderNodeMath")  # dust sits on top: no gloss there
    cw.operation = "MULTIPLY_ADD"
    cw.inputs[1].default_value = -0.6
    cw.inputs[2].default_value = 0.6
    nt.links.new(dust.outputs[0], cw.inputs[0])
    nt.links.new(cw.outputs[0], b.inputs["Coat Weight"])
    return m


# ---------------------------------------------------------------- solder generation
def solder_jobs(pads_json):
    """pads.json -> solder_joints.Pad list (which pads get solder, HASL or nothing)."""
    from solder_joints import Pad as SJPad
    from solder_joints.core import point_in_polygon
    pads = pads_json["pads"]
    paste_only = [p for p in pads if not p["copper"] and p["paste"]]
    out = []
    for p in pads:
        if not p["copper"] or not p["poly"] or p["dnp"]:
            continue
        poly = np.array(p["poly"])
        sgn = {"F": 1, "B": -1}
        if p["tht"]:
            kind, sides = "tht", [sgn[c] for c in p["copper"]]
        else:
            has_paste = bool(p["paste"]) or any(
                q["ref"] == p["ref"] and point_in_polygon(np.array([q["x"]]), np.array([q["y"]]), poly)[0]
                for q in paste_only)
            kind = "smd" if has_paste else "bare"
            sides = [sgn[p["copper"][0] if len(p["copper"]) == 1 else p["fp_side"]]]
        out.append(SJPad(outline=poly, center=(p["x"], p["y"]), kind=kind, sides=sides,
                         drill=float(min(p["drill"])) if p["tht"] else 0.0,
                         part_center=(p["fp_x"], p["fp_y"]), component_side=sgn[p["fp_side"]],
                         name=f"{p['ref']}.{p['num']}", size=(p["w"], p["h"])))
    return out


def gen_solder(pads_json, bmap, comps, imp, rng):
    """Solder joints via the standalone solder_joints module (LGPL-3.0)."""
    from solder_joints import Settings as SJSettings, generate as sj_generate
    from solder_joints.blender import BoardFrame, BVHProber, build_objects
    frame = BoardFrame(xy=bmap.xy, mm=bmap.s, z_top=bmap.z_top, z_bottom=bmap.z_bot)
    prober = BVHProber(comps, METALS, frame)
    log(f"component triangles: {prober.triangle_count}, metal {int(prober.metal.sum())}")
    finish = (pads_json.get("stackup", {}).get("finish") or "").lower()
    settings = SJSettings(hasl="hal" in finish or "hasl" in finish, imperfection=imp)
    pads = solder_jobs(pads_json)
    log(f"generating solder for {len(pads)} pads")
    joints, report = sj_generate(pads, prober, settings, rng)
    log(f"solder stats: {report.as_dict()}")
    if report.no_pin:
        log(f"through-hole pads with no pin in the 3D model: {' '.join(report.no_pin)}")
    return build_objects(joints, frame)


# ---------------------------------------------------------------- scene
def setup_render(args):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.samples = args.samples
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    cy.max_bounces = 12
    cy.glossy_bounces = 8
    cy.transmission_bounces = 8
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    W, H = (int(v) for v in args.res.lower().split("x"))
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = args.backdrop == "none"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "16"
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
        sc.view_settings.exposure = args.exposure
    except Exception as e:
        log("colour management:", e)
    device = "CPU"
    if args.device in ("auto", "gpu", "both"):
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("HIP", "OPTIX", "CUDA", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == backend]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == backend or (args.device == "both" and d.type == "CPU")
                    device = "GPU"
                    log(f"GPU backend {backend}: {[d.name for d in gpus]}")
                    break
            except Exception:
                continue
        if device == "CPU":
            log("no usable GPU backend, rendering on CPU")
            if args.device == "gpu":
                raise SystemExit("--device gpu requested but no GPU backend found")
    cy.device = device


HDRI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "hdri")


def setup_world(mm_to_bu, hdri=None, strength=None):
    w = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.2
    nt.links.new(bg.outputs[0], out.inputs[0])
    hdr = None
    if hdri:
        cand = hdri if os.path.exists(hdri) else os.path.join(HDRI_DIR, hdri if hdri.endswith(".exr") else hdri + "_2k.exr")
        if os.path.exists(cand):
            hdr = cand
            bg.inputs["Strength"].default_value = 0.5 if strength is None else strength
        else:
            log(f"HDRI {hdri!r} not found, falling back to Blender's studio.exr")
    if strength is not None:
        bg.inputs["Strength"].default_value = strength
    try:
        d = bpy.utils.system_resource("DATAFILES", path="studiolights/world") if hdr is None else None
        for name in ("studio.exr", "interior.exr", "city.exr", "forest.exr"):
            if d and os.path.exists(os.path.join(d, name)):
                hdr = os.path.join(d, name)
                break
    except Exception:
        pass
    if hdr:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdr)
        nt.links.new(env.outputs[0], bg.inputs["Color"])
        log("environment:", os.path.basename(hdr))
    else:
        bg.inputs["Color"].default_value = (0.05, 0.05, 0.055, 1)
    try:
        w.visibility.camera = False
    except Exception:
        pass


def add_area(name, loc, target, size, power, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "DISK"
    ld.size = size
    ld.energy = power
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


def scene_bounds(objs):
    V = np.concatenate([world_verts(o) for o in objs if o.type == "MESH" and len(o.data.vertices)])
    return V.min(0), V.max(0)


def look_at(cam, target):
    d = Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def kicad_viewport(pcb_path, name):
    """Camera rotation from a viewport saved in KiCad's 3D viewer (.kicad_pro "3dviewports").

    KiCad stores the camera's view matrix (world -> camera, OpenGL: camera looks down -Z,
    +Y up). KiCad's 3D world is x right, y up (board y flipped), z out of the top side,
    which is the same frame as the imported GLB, so camera-to-world rotation = R^T.
    Returns (forward, up) world vectors, i.e. camera -Z and +Y.
    """
    pro = os.path.splitext(pcb_path)[0] + ".kicad_pro"
    data = json.load(open(pro))
    vps = data.get("board", {}).get("3dviewports", []) or data.get("3dviewports", [])
    names = [v.get("name") for v in vps]
    vp = next((v for v in vps if v.get("name") == name), None)
    if vp is None:
        raise SystemExit(f"viewport '{name}' not in {pro}; saved viewports: {names}")
    m = vp.get("matrix")
    flat = np.array(m, dtype=float).ravel()
    if flat.size != 16:
        raise SystemExit(f"viewport '{name}': unexpected matrix {m!r}")
    A = flat.reshape(4, 4)
    # Column-major (glm) puts translation in the last row; row-major in the last column.
    if np.allclose(A[:3, 3], 0) and not np.allclose(A[3, :3], 0):
        A = A.T
        layout = "column-major"
    elif np.allclose(A[3, :3], 0) and not np.allclose(A[:3, 3], 0):
        layout = "row-major"
    else:
        layout = "ambiguous (no translation), read row-major"
    R = A[:3, :3]
    U, _, Vt = np.linalg.svd(R)  # strip any scale
    R = U @ Vt
    fwd = Vector((-R[2, 0], -R[2, 1], -R[2, 2]))  # camera -Z in world = -(row 2 of R)
    up = Vector((R[1, 0], R[1, 1], R[1, 2]))
    log(f"viewport '{name}' ({layout}): forward {tuple(round(v, 3) for v in fwd)}, up {tuple(round(v, 3) for v in up)}")
    return fwd.normalized(), up.normalized()


def fit_distance(lo, hi, target, view_dir, up_hint, fov_x, fov_y, margin=1.06):
    """Camera distance along view_dir (unit, from target towards the camera) so that the whole
    bounding box lo..hi fits the frame. Fits long, thin boards properly (a diagonal-based
    extent leaves them tiny in a mostly empty frame)."""
    f = -view_dir
    r = f.cross(up_hint)
    if r.length < 1e-9:
        r = f.cross(Vector((0, 1, 0)))
    r.normalize()
    u = r.cross(f).normalized()
    tx, ty = math.tan(fov_x / 2), math.tan(fov_y / 2)
    need = 0.0
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                v = Vector((cx, cy, cz)) - target
                along = v.dot(view_dir)          # towards the camera
                need = max(need, along + abs(v.dot(r)) * margin / tx, along + abs(v.dot(u)) * margin / ty)
    return need


def auto_fstop(extent_bu, sensor_mm=36.0, coc_mm=0.03, sharp_frac=0.7):
    """f-number giving ~sharp_frac * frame width of depth of field (focus-stacked look)."""
    ext_mm = extent_bu / MM
    m = sensor_mm / ext_mm
    dof = sharp_frac * ext_mm
    return float(np.clip(dof * m * m / (2 * coc_mm * (m + 1)), 4, 4000))


# ---------------------------------------------------------------- main
def build_scene(args):
    """Everything that is the same for every shot: import, materials, relief, normals, solder."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    pads_json = json.load(open(args.pads))
    global REFS
    REFS = {p["ref"] for p in pads_json["pads"]}

    log("importing", args.glb)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(args.glb))
    offboard = set()
    if not args.keep_offboard:
        offboard = {f["ref"] for f in pads_json.get("footprints", []) if not f["onboard"]}
        doomed = set()
        for o in bpy.context.scene.objects:
            if o.name.split(".")[0] in offboard:
                doomed.add(o)
                doomed.update(o.children_recursive)
        gone = len(doomed)
        for o in doomed:
            bpy.data.objects.remove(o, do_unlink=True)
        pads_json["all_pads"] = pads_json["pads"]
        pads_json["pads"] = [p for p in pads_json["pads"] if p["ref"] not in offboard]
        log(f"hid {len(offboard)} off-board footprints ({gone} objects): {' '.join(sorted(offboard))}")
    imported = list(bpy.context.scene.objects)

    # Board layers are meshes named "<board>_<layer>" hanging directly off the root,
    # not inside a footprint node (component models can have meshes named "..._pad").
    board = os.path.splitext(os.path.basename(pads_json["pcb"]))[0]
    layer_objs = {}
    comps = []
    for o in imported:
        if o.type != "MESH":
            continue
        dn = o.data.name.split(".")[0]
        for key in ("copper", "pad", "via", "silkscreen", "soldermask", "PCB"):
            if dn == f"{board}_{key}":
                layer_objs.setdefault(key, []).append(o)
                break
        else:
            comps.append(o)
    log({k: len(v) for k, v in layer_objs.items()}, f"{len(comps)} component meshes")

    bmap = BoardMap(pads_json, layer_objs["pad"])
    if offboard:
        cull_offboard_faces(layer_objs, bmap)
    s = bmap.s  # BU per mm
    global MM
    MM = s
    rng = np.random.default_rng(args.seed)
    mats = make_materials(s, args.imperfection, pads_json.get("stackup"))
    log("stackup:", pads_json.get("stackup", {}))

    for key, objs in layer_objs.items():
        m = {"copper": "enig", "pad": "enig", "via": "enig", "silkscreen": "silk",
             "soldermask": "mask", "PCB": "fr4"}[key]
        for o in objs:
            o.data.materials.clear()
            o.data.materials.append(mats[m])

    # Silkscreen: KiCad exports it as a zero-thickness sheet floating ~25 um above the mask.
    # Give it real ink thickness down to the mask, with softly rounded edges.
    zmid_ = (bmap.z_top + bmap.z_bot) / 2
    mask_z = {}
    for o in layer_objs.get("soldermask", []):
        V = world_verts(o)
        sd = "F" if V[:, 2].mean() > zmid_ else "B"
        mask_z[sd] = V[:, 2].max() if sd == "F" else V[:, 2].min()
    for o in layer_objs.get("silkscreen", []):
        V = world_verts(o)
        sd = "F" if V[:, 2].mean() > zmid_ else "B"
        gap = abs(V[:, 2].mean() - mask_z.get(sd, V[:, 2].mean())) / s  # mm
        th = float(np.clip(gap + 0.002, 0.012, 0.04))
        me = o.data
        n = np.empty(len(me.polygons) * 3)
        me.polygons.foreach_get("normal", n)
        nz = n.reshape(-1, 3)[:, 2].mean() * o.matrix_world[2][2]
        outward = 1.0 if sd == "F" else -1.0
        sol = o.modifiers.new("ink", "SOLIDIFY")
        sol.thickness = th * s
        sol.offset = -1.0 if nz * outward > 0 else 1.0  # grow towards the board
        sol.use_even_offset = True
        bev = o.modifiers.new("ink_edge", "BEVEL")
        bev.width = 0.008 * s
        bev.segments = 2
        bev.limit_method = "ANGLE"
        log(f"silkscreen {sd}: {th * 1000:.0f} um ink (gap to mask {gap * 1000:.0f} um)")

    # copper showing through the mask as relief (+ tint, dust)
    try:
        gst = os.stat(args.glb)
        relief = copper_relief(layer_objs, bmap, os.path.dirname(os.path.abspath(args.glb)), args.imperfection,
                               cache_key=f"{gst.st_size}-{int(gst.st_mtime)}")
        zmid = (bmap.z_top + bmap.z_bot) / 2
        for o in layer_objs.get("soldermask", []):
            side = "F" if world_verts(o)[:, 2].mean() > zmid else "B"
            col = mask_colours(pads_json.get("stackup", {}).get(f"{side}.Mask"))
            mm_ = mask_material_for_side(mats["mask"], relief[side], args.imperfection, f"Soldermask.{side}", col)
            o.data.materials.clear()
            o.data.materials.append(mm_)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log("copper relief failed, flat mask:", e)
    # Many vendor models (EasyEDA/LCSC .wrl) smooth normals across hard edges -> "melted" look.
    # Drop imported custom normals and smooth by angle instead.
    done, fixed = set(), 0
    for o in comps:
        me = o.data
        if me.name in done:
            continue
        done.add(me.name)
        try:
            if me.has_custom_normals:
                with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o]):
                    bpy.ops.mesh.customdata_custom_splitnormals_clear()
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(30))
            fixed += 1
        except Exception as e:
            log("normals:", me.name, e)
    log(f"re-derived normals on {fixed} component meshes")

    kinds = {}
    seen = set()
    body_frac, big_sheet = material_body_fractions(comps)
    for o in comps:
        for slot in o.material_slots:
            if slot.material and slot.material.name not in seen:
                seen.add(slot.material.name)
                k = classify_component_material(slot.material, body_frac.get(slot.material.name, 0.0),
                                                big_sheet.get(slot.material.name, False))
                if k in ("metal", "gold", "satin"):
                    METALS.add(slot.material.name)
                kinds[k] = kinds.get(k, 0) + 1
    log("component materials:", kinds)

    if not args.no_solder:
        log("building component BVH")
        jitter_placement(pads_json, args.imperfection, rng)
        objs = gen_solder(pads_json, bmap, comps, args.imperfection, rng)
        for bucket, ob in objs.items():
            ob.data.materials.append(mats["solder"] if bucket == "solder" else mats["enig"])
            imported.append(ob)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    roots = [o for o in bpy.context.scene.objects if o.parent is None]
    return {
        "pads_json": pads_json, "bmap": bmap, "s": s, "meshes": meshes,
        "base": {o.name: o.matrix_world.copy() for o in roots},
    }


SHOT_KEYS = ("preset", "side", "focus", "zoom", "azimuth", "elevation", "fstop", "lens", "viewport",
             "out", "samples", "res", "backdrop", "env_strength", "exposure", "save_blend")


def shot_collection():
    """Objects that belong to one shot (camera, lights, backdrop); emptied before the next."""
    col = bpy.data.collections.get("Shot")
    if col is None:
        col = bpy.data.collections.new("Shot")
        bpy.context.scene.collection.children.link(col)
    for o in list(col.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    return col


def render_shot(ctx, args):
    """Pose the board for the shot's side, then backdrop, world, camera, lights, render."""
    s, bmap, meshes, pads_json = ctx["s"], ctx["bmap"], ctx["meshes"], ctx["pads_json"]
    col = shot_collection()

    def link(o):
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
        return o

    # always start from the imported pose, then flip for the bottom side (180 deg about Y)
    for name, M in ctx["base"].items():
        o = bpy.data.objects.get(name)
        if o is not None:
            o.matrix_world = M
    bpy.context.view_layer.update()
    lo, hi = scene_bounds(meshes)
    centre = (lo + hi) / 2
    surface_z = bmap.z_top  # laminate+pad surface facing the camera
    if args.viewport:
        args.side = "F"  # camera from the KiCad viewport decides which side is seen
    if args.side == "B":
        surface_z = 2 * centre[2] - bmap.z_bot
        R = (Matrix.Translation(Vector(centre)) @ Matrix.Rotation(math.pi, 4, "Y")
             @ Matrix.Translation(-Vector(centre)))
        for name in ctx["base"]:
            o = bpy.data.objects.get(name)
            if o is not None:
                o.matrix_world = R @ o.matrix_world
        bpy.context.view_layer.update()
        lo, hi = scene_bounds(meshes)
    centre = (lo + hi) / 2
    size = hi - lo
    diag = float(np.linalg.norm(size[:2]))
    log(f"board extent {size[0] / s:.1f} x {size[1] / s:.1f} mm, height {size[2] / s:.2f} mm")

    # backdrop
    if args.backdrop != "none":
        bpy.ops.mesh.primitive_plane_add(size=diag * 40, location=(centre[0], centre[1], lo[2] - 0.0005 * s))
        bd = link(bpy.context.active_object)
        bd.name = "Backdrop"
        m, b = new_mat("Backdrop")
        if args.backdrop == "dark":
            setp(b, "Base Color", (0.018, 0.018, 0.02, 1))
            setp(b, "Roughness", 0.45)
        else:
            setp(b, "Base Color", (0.6, 0.6, 0.62, 1))
            setp(b, "Roughness", 0.8)
        bd.data.materials.append(m)

    # Back-side shots: front-side modules hang below the flipped board with their undersides
    # facing up, and the environment lit them as hard as the board top. The area lights carry
    # the board, so a dimmer environment keeps those undersides in believable shade.
    env = args.env_strength
    if env is None and args.side == "B":
        env = 0.2
    setup_world(s, args.hdri, env)
    setup_render(args)

    # focus target
    target = Vector(centre)
    focus_extent = diag
    if args.focus:
        fo = [o for o in bpy.context.scene.objects if o.name.split(".")[0] == args.focus]
        kids = []
        for o in fo:
            kids += [c for c in o.children_recursive if c.type == "MESH"] + ([o] if o.type == "MESH" else [])
        if kids:
            flo, fhi = scene_bounds(kids)
            target = Vector(((flo[0] + fhi[0]) / 2, (flo[1] + fhi[1]) / 2, 0))
            target.z = surface_z + 0.15 * s  # focus near the joints
            focus_extent = max(float(np.linalg.norm((fhi - flo)[:2])), 1.5 * s)
            log(f"focus {args.focus} at {tuple(round(v, 4) for v in target)}, extent {focus_extent / s:.1f} mm")
        else:
            log(f"focus {args.focus} not found")

    # camera
    cd = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cd)
    col.objects.link(cam)
    bpy.context.scene.camera = cam
    cd.clip_start = diag * 0.005
    cd.clip_end = diag * 200
    W, H = (int(v) for v in args.res.lower().split("x"))
    aspect = W / H

    preset = args.preset
    az = 0.0
    if args.viewport:
        fwd, up = kicad_viewport(pads_json["pcb"], args.viewport)
        if fwd.z > 0:  # looking up at the bottom side
            log("viewport looks at the bottom side")
        cd.lens = args.lens or 90.0
        extent = (focus_extent * 2.2 if args.focus else diag) * (args.zoom or 1.0)
        fov = 2 * math.atan(cd.sensor_width / 2 / cd.lens)
        if aspect < 1:
            fov *= aspect
        dist = (extent / 2) / math.tan(fov / 2) * 1.05
        cam.location = target - fwd * dist
        right = fwd.cross(up).normalized()
        up = right.cross(fwd).normalized()
        Rm = Matrix((right, up, -fwd)).transposed()
        cam.rotation_euler = Rm.to_euler()
        az = math.degrees(math.atan2(-fwd.x, fwd.y)) if abs(fwd.z) < 0.999 else 0.0
        cd.dof.use_dof = True
        cd.dof.focus_distance = dist
        cd.dof.aperture_fstop = args.fstop or (auto_fstop(extent) if args.focus else 32.0)
    elif preset == "top":
        cd.type = "ORTHO"
        # ortho_scale spans the larger image dimension
        cd.ortho_scale = (max(size[0], size[1] * aspect) if aspect >= 1 else max(size[0] / aspect, size[1])) * 1.08
        cam.location = (centre[0], centre[1], hi[2] + diag)
        cam.rotation_euler = (0, 0, 0)
    else:
        az = {"hero": -35.0, "macro": -30.0, "low": -20.0}[preset]
        el = {"hero": 38.0, "macro": 32.0, "low": 14.0}[preset]
        if args.azimuth is not None:
            az = args.azimuth
        if args.elevation is not None:
            el = args.elevation
        cd.lens = args.lens or {"hero": 90.0, "macro": 100.0, "low": 70.0}[preset]
        extent = focus_extent if (preset == "macro" or args.focus) else diag
        if preset == "macro" and args.focus:
            extent *= 2.2
        extent *= args.zoom or 1.0
        fov = 2 * math.atan(cd.sensor_width / 2 / cd.lens)
        if aspect < 1:
            fov *= aspect
        dist = (extent / 2) / math.tan(fov / 2) * 1.05
        a, e = math.radians(az), math.radians(el)
        dirv = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        if not args.focus and preset != "macro":
            # whole-board shots: fit the actual bounding box from this angle
            fov_x = 2 * math.atan(cd.sensor_width / 2 / cd.lens)   # sensor fit AUTO: width spans the long side
            fov_y = 2 * math.atan(math.tan(fov_x / 2) / aspect) if aspect >= 1 else fov_x
            if aspect < 1:
                fov_y, fov_x = fov_x, 2 * math.atan(math.tan(fov_x / 2) * aspect)
            dist = fit_distance(lo, hi, target, dirv, Vector((0, 0, 1)), fov_x, fov_y) * (args.zoom or 1.0)
            extent = 2 * dist * math.tan(fov_x / 2)
        cam.location = target + dirv * dist
        look_at(cam, target)
        cd.dof.use_dof = True
        cd.dof.focus_distance = (target - cam.location).length
        if args.fstop:
            cd.dof.aperture_fstop = args.fstop
        elif preset == "macro" or args.focus:
            cd.dof.aperture_fstop = auto_fstop(extent)
        else:
            cd.dof.aperture_fstop = {"hero": 32.0, "low": 22.0}[preset]
        log(f"camera: {cd.lens:.0f} mm lens, f/{cd.dof.aperture_fstop:.0f}, frame {extent / MM:.1f} mm")

    # lights, scaled to the board; power ~ distance^2 keeps exposure constant
    L = diag
    dk = 2.2 * L
    # positions are given for a camera at -Y and rotated with the camera azimuth so that
    # no light sits in the mirror direction of the glossy mask as seen from the camera
    rot = Matrix.Rotation(math.radians(az), 3, "Z")

    def lp(v):
        return centre + (rot @ Vector(v).normalized()) * dk

    for lt in (add_area("Key", lp((-1.0, -0.7, 1.3)), centre, 1.2 * L, 14 * (dk ** 2), (1.0, 0.97, 0.92)),
               add_area("Fill", lp((1.1, -0.5, 1.4)), centre, 2.0 * L, 6 * (dk ** 2), (0.92, 0.95, 1.0)),
               add_area("Rim", lp((-1.3, 1.0, 0.9)), centre, 0.8 * L, 25 * (dk ** 2)),
               add_area("Top", lp((0, -0.25, 1)), centre, 1.5 * L, 6 * (dk ** 2)) if preset != "top" else None):
        if lt is not None:
            link(lt)

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
        log("saved", args.save_blend)

    sc = bpy.context.scene
    sc.render.filepath = os.path.abspath(args.out)
    log(f"rendering {preset} side {args.side} {W}x{H} @ {args.samples} spp on {sc.cycles.device}")
    bpy.ops.render.render(write_still=True)
    log("wrote", args.out)


def main():
    import copy
    args = parse_args()
    shots = [{}]
    if not args.shots and not args.out:
        raise SystemExit("--out or --shots is required")
    if args.shots:
        shots = json.load(open(args.shots))
        log(f"{len(shots)} shots in one session")
    ctx = build_scene(args)
    for k, shot in enumerate(shots):
        a = copy.copy(args)
        for key, val in shot.items():
            if key not in SHOT_KEYS:
                raise SystemExit(f"shot {k + 1}: unknown key {key!r}")
            setattr(a, key, val)
        if len(shots) > 1:
            log(f"=== shot {k + 1}/{len(shots)}: {shot}")
        try:
            render_shot(ctx, a)
        except Exception:
            import traceback
            traceback.print_exc()
            log(f"shot {k + 1} failed")


main()
