# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Chris Fourie (https://github.com/No1089)
"""Solder joint heightfields for PCB pads (pure NumPy, no Blender).

Everything here works in the *board frame*: x, y in millimetres on the board, and a height
``h`` in millimetres measured outward from the copper surface of the side being soldered
(so ``h > 0`` points away from the board on both top and bottom). The caller supplies a
:class:`Prober` that answers ray queries against the component geometry in that frame;
:mod:`solder_joints.blender` provides one backed by Blender BVH trees.

For every pad the generator samples a grid inside the pad outline, probes what sits above
each point, and builds one joint as a heightfield:

* **SMD**: points under *metal* (lead, end cap, flank) become the terminal; one concave
  meniscus runs from ~55 % of that metal's height down to zero at the pad edge, with a thin
  wetting layer elsewhere. Plastic over a pad never pulls solder up, but a small standoff gap
  under a leadless body is filled (QFN/DFN/LGA toe). Metal side flanks next to the pad (found
  by sideways probes towards the part) give a small toe fillet.
* **THT**: every point inside the hole is probed along the hole axis, so the real pin
  cross-section (round pin, flat blade, offset lead) drives the fillet, which clings to the
  pin and falls away concavely to the annular-ring edge. Large on the solder side, small on
  the component side.
* **HASL / bare**: a thin domed, slightly wavy levelled-tin coating (no part needed).

Optional ``imperfection`` varies paste volume per joint and adds gentle lumpiness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np


# ---------------------------------------------------------------- data
@dataclass
class Pad:
    """One copper pad, in board-frame millimetres.

    kind: "smd" (has paste), "tht" (plated hole) or "bare" (no paste; HASL coating if enabled).
    sides: which copper sides to process, +1 = top, -1 = bottom.
    """
    outline: np.ndarray                       # (N, 2) closed polygon, mm
    center: tuple[float, float]               # pad centre, mm
    kind: str = "smd"
    sides: Sequence[int] = (1,)
    drill: float = 0.0                        # mm, THT only (smallest drill dimension)
    part_center: Optional[tuple[float, float]] = None   # footprint centre (toe-fillet probes)
    component_side: int = 1                   # side the part body sits on (THT fillet size)
    name: str = ""
    size: Optional[tuple[float, float]] = None  # pad width/height in mm (unrotated), if known

    @property
    def min_size(self) -> float:
        if self.size is not None:
            return float(max(min(self.size), 0.1))
        span = self.outline.max(0) - self.outline.min(0)
        return float(max(min(span), 0.1))


@dataclass
class Hit:
    distance: float      # mm along the ray
    exiting: bool        # True if the ray left a solid (it started inside the part model)
    metal: bool          # the surface hit is metal (solder wets it)
    h: float             # height of the hit point above the pad surface, mm


class Prober(Protocol):
    def cast(self, x: float, y: float, h: float, direction: tuple[float, float, float],
             max_dist: float, side: int) -> Optional[Hit]:
        """Ray from (x, y, h) in the board frame of `side`; direction (dx, dy, dh), unit length."""


@dataclass
class Settings:
    fillet_fraction: float = 0.42   # fillet climbs this share of the terminal height
    fillet_max: float = 0.28        # mm cap for SMD fillets
    fillet_min: float = 0.10        # mm: thin leads still get a visible heel
    gap_max: float = 0.15           # mm: something this close above a pad point covers it
    standoff_max: float = 0.12      # mm: plastic closer than this is a leadless underside
    wetting: float = 0.02           # mm: thin solder layer over the rest of a pasted pad
    tht_reach: float = 3.0          # mm: how far outside the board to start THT probes
    tht_component_side: float = 0.18   # mm fillet height on the component side
    hasl: bool = False              # coat bare pads / empty holes with HASL tin
    imperfection: float = 0.0       # 0 = CAD-perfect, 1 = rough hand-built look
    lift: float = 0.003             # mm above the pad surface (avoids z-fighting)
    max_cells: int = 140            # per axis, per pad


@dataclass
class Joint:
    """A generated joint: grid vertices in the board frame plus quads, ready to map to 3D."""
    pad: Pad
    side: int
    bucket: str                     # "solder" or "hasl"
    x: np.ndarray                   # (V,) mm
    y: np.ndarray                   # (V,) mm
    h: np.ndarray                   # (V,) mm above the pad surface (outward)
    quads: np.ndarray               # (Q, 4) vertex indices, counter-clockwise in the x/y frame


@dataclass
class Report:
    fillet: int = 0
    dome: int = 0
    tht: int = 0
    hasl: int = 0
    skip: int = 0
    no_pin: list = field(default_factory=list)   # THT pads whose 3D model has no pin

    def as_dict(self):
        return {"fillet": self.fillet, "dome": self.dome, "tht": self.tht,
                "hasl": self.hasl, "skip": self.skip}


# ---------------------------------------------------------------- outlines
def pad_outline(shape, size, position=(0.0, 0.0), rotation=0.0, roundness=0.25, segments=12):
    """Pad copper outline from parametric pad data (for hosts that don't export polygons).

    shape: "rect", "roundrect", "circle" or "oval"; size (w, h) mm; rotation in radians
    (counter-clockwise, KiCad's convention in a y-up frame); roundness = corner radius / min(w, h)
    (KiCad's RoundRectRadiusRatio, 0..0.5).
    """
    w, h = float(size[0]), float(size[1])
    shape = shape.lower()
    if shape in ("circle", "circular"):
        r, rr = w / 2, w / 2
        w = h = 2 * r
    elif shape == "oval":
        rr = min(w, h) / 2
    elif shape in ("roundrect", "rounded"):
        rr = max(0.0, min(0.5, roundness)) * min(w, h)
    else:
        rr = 0.0
    hx, hy = w / 2 - rr, h / 2 - rr
    if rr <= 0:
        pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    else:
        pts = []
        for cx, cy, a0 in ((hx, -hy, -math.pi / 2), (hx, hy, 0.0), (-hx, hy, math.pi / 2), (-hx, -hy, math.pi)):
            for k in range(segments + 1):
                a = a0 + (math.pi / 2) * k / segments
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    P = np.array(pts)
    c, s_ = math.cos(rotation), math.sin(rotation)
    P = P @ np.array([[c, s_], [-s_, c]])
    return P + np.asarray(position, dtype=float)


# ---------------------------------------------------------------- 2D helpers
def point_in_polygon(px, py, poly):
    """Vectorised even-odd point-in-polygon."""
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    c = np.zeros(px.shape, dtype=bool)
    for i in range(len(x)):
        dy = y2[i] - y[i]
        if dy == 0:
            continue
        cond = ((y[i] > py) != (y2[i] > py)) & (px < (x2[i] - x[i]) * (py - y[i]) / dy + x[i])
        c ^= cond
    return c


def edge_distance(px, py, poly):
    """Distance from each point to the polygon outline."""
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    best = np.full(px.shape, np.inf)
    for i in range(len(x)):
        ex, ey = x2[i] - x[i], y2[i] - y[i]
        L2 = ex * ex + ey * ey
        if L2 == 0:
            continue
        t = np.clip(((px - x[i]) * ex + (py - y[i]) * ey) / L2, 0, 1)
        best = np.minimum(best, np.hypot(px - (x[i] + t * ex), py - (y[i] + t * ey)))
    return best


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def smooth_grid(h, ins, ny, nx, passes=2):
    """Light 3x3 box blur of a heightfield, restricted to inside points (surface tension)."""
    H = h.reshape(ny, nx).copy()
    M = ins.reshape(ny, nx).astype(float)
    for _ in range(passes):
        P = np.pad(H * M, 1)
        W = np.pad(M, 1)
        acc = sum(P[1 + dy:ny + 1 + dy, 1 + dx:nx + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        wts = sum(W[1 + dy:ny + 1 + dy, 1 + dx:nx + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        H = np.where(M > 0, acc / np.maximum(wts, 1), 0)
    return H.ravel()


def lumpy(px, py, rng, amp):
    """Sum of a few random long waves -> 1 +/- amp."""
    f = np.zeros_like(px)
    for _ in range(4):
        k = 2 * math.pi / rng.uniform(0.3, 0.9)
        a = rng.uniform(0, 2 * math.pi)
        f += np.sin(k * (px * math.cos(a) + py * math.sin(a)) + rng.uniform(0, 6.3))
    return 1 + amp * f / 2


def hasl_height(ed, ins, dmin, px, py, rng, imp):
    """HASL: hot-air-levelled tin, thicker on small pads, domed and a little wavy."""
    hd = 0.004 + 0.022 * math.exp(-dmin / 0.8)  # mm: ~20 um on fine pads, few um on big ones
    h = hd * smoothstep(ed / max(0.35 * dmin, 0.02)) * ins
    return h * lumpy(px, py, rng, 0.25 + 0.25 * imp)


def meniscus(px, py, idx, sources, src_h, ed, exponent):
    """Concave meniscus: each point takes the height of its nearest source, falling to zero at
    the pad edge along t = d_source / (d_source + d_edge)."""
    Dn = np.full(len(idx), np.inf)
    Hn = np.zeros(len(idx))
    for k in range(0, len(sources), 256):
        qq = sources[k:k + 256]
        D = np.hypot(px[idx][:, None] - px[qq][None, :], py[idx][:, None] - py[qq][None, :])
        j = D.argmin(1)
        dj = D[np.arange(len(idx)), j]
        better = dj < Dn
        Dn[better] = dj[better]
        Hn[better] = src_h[k:k + 256][j[better]]
    t = Dn / np.maximum(Dn + ed[idx], 1e-9)
    return Hn * (1 - t) ** exponent


def blend_wetting(hf, edi, dmin, scale=1.0, wetting=0.02):
    """Smooth-max of the fillet and a thin wetting layer (~25 deg contact angle at the edge)."""
    hw = min(wetting, 0.06 * dmin) * scale * (1 - np.exp(-edi / 0.04))
    kb = 0.015
    hb = np.clip(kb - np.abs(hf - hw), 0, None) / kb
    return np.maximum(hf, hw) + hb * hb * kb / 4


# ---------------------------------------------------------------- generator
def _grid(pad, s):
    dmin = pad.min_size
    c = float(np.clip(dmin / 36, 0.006, 0.04))
    lo, hi = pad.outline.min(0) - c, pad.outline.max(0) + c
    nx = int(min(s.max_cells, math.ceil((hi[0] - lo[0]) / c) + 1))
    ny = int(min(s.max_cells, math.ceil((hi[1] - lo[1]) / c) + 1))
    GX, GY = np.meshgrid(np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny))
    return GX.ravel(), GY.ravel(), nx, ny


def _smd(pad, side, px, py, ins, ed, ny, nx, prober, s, rng, rep):
    dmin = pad.min_size
    imp = s.imperfection
    idx = np.nonzero(ins)[0]
    covered, Hq, is_metal, open_pts = [], [], [], []
    metal_above = False
    for i in idx:
        # Long enough to find the top of a terminal we may already be inside (3D models often
        # sit on the laminate, i.e. sunk into the raised pad); entering hits only count within
        # gap_max, so a housing far above the pad doesn't cover it.
        hit = prober.cast(px[i], py[i], 0.002, (0.0, 0.0, 1.0), 1.2, side)
        if hit is not None and not hit.exiting and hit.distance > s.gap_max:
            hit = None
        if hit is None:
            open_pts.append(i)
            continue
        covered.append(i)
        if not hit.metal:
            # plastic body over the pad: the joint fills a small standoff gap (leadless
            # underside terminal), but solder never climbs plastic
            g = hit.distance
            Hq.append(min(g + 0.01, 0.15) if not hit.exiting and g < s.standoff_max else 0.0)
            is_metal.append(False)
            continue
        # Terminal height. Vendor models are messy (end caps are thin shells around the body,
        # leads overlap housings), so first look from above: if the topmost surface here is
        # metal, that's the terminal top. Only if plastic sits on top (lead tucked under a
        # body) measure this piece of metal's own thickness instead.
        top = prober.cast(px[i], py[i], 1.2, (0.0, 0.0, -1.0), 1.2, side)
        if top is not None and top.metal and top.h > 0:
            T = top.h
        elif hit.exiting:
            T = hit.distance
        else:
            h2 = prober.cast(px[i], py[i], hit.h + 0.001, (0.0, 0.0, 1.0), 1.2, side)
            T = hit.distance + (h2.distance if h2 is not None else 0.1)
        T = float(np.clip(T, 0.02, 1.0))
        Hq.append(min(max(s.fillet_fraction * T, min(T, s.fillet_min)), s.fillet_max))
        is_metal.append(True)
        metal_above = True

    # Leadless parts (DFN/LGA flanks, castellations): no metal above the pad at all, but a metal
    # flank right next to it, towards the part -> small toe fillet up the flank. Skipped when the
    # pad already has a terminal above it (that would put a flat shoulder outside the terminal).
    if pad.part_center is not None and not metal_above:
        for i in open_pts:
            dx, dy = pad.part_center[0] - px[i], pad.part_center[1] - py[i]
            L = math.hypot(dx, dy)
            if L <= 1e-9:
                continue
            for hh in (0.04, 0.1, 0.18, 0.26):  # models often float above the pad
                h2 = prober.cast(px[i], py[i], hh, (dx / L, dy / L, 0.0), 0.03, side)
                if h2 is not None and h2.metal:
                    covered.append(i)
                    Hq.append(max(0.14, hh + 0.05))
                    is_metal.append(True)
                    break

    hgt = np.zeros(px.shape)
    if not covered:
        # reflowed paste with nothing on it (test pad / missing model): a pillow
        rep.dome += 1
        hd = min(0.1, 0.22 * dmin)
        return hd * smoothstep(ed / (0.45 * dmin)) * ins
    rep.fillet += 1
    Q = np.array(covered)
    vol = float(np.clip(1 + imp * 0.2 * rng.normal(), 0.65, 1.4))  # per-joint paste volume
    Hq = np.array(Hq)
    metal = np.array(is_metal) & (Hq > 0)
    standoff = ~np.array(is_metal) & (Hq > 0)
    if metal.any():
        # one fillet height per joint for the metal (terminal height varies at chamfers ->
        # ridges). Leadless pads (no metal directly above, only a flank) also keep their filled
        # standoff column at its own lower height; under a chip body that would only add ridges.
        Hm = float(np.percentile(Hq[metal], 80)) * vol
        keep = metal if metal_above else (metal | standoff)
        Q, Hq = Q[keep], np.where(metal[keep], Hm, Hq[keep] * vol)
    elif standoff.any():
        Q, Hq = Q[standoff], Hq[standoff] * vol
    else:
        Hq = np.zeros(len(Q))
    if len(Q) > 1500:
        sel = np.random.default_rng(0).choice(len(Q), 1500, replace=False)
        Q, Hq = Q[sel], Hq[sel]
    hf = meniscus(px, py, idx, Q, Hq, ed, 1.7)
    hgt[idx] = blend_wetting(hf, ed[idx], dmin, vol, s.wetting)
    return smooth_grid(hgt, ins, ny, nx, passes=3)


def _tht(pad, side, px, py, ins, ed, ny, nx, prober, s, rng, rep):
    """Returns (height, inside-mask, bucket) or None when the pad is skipped."""
    dmin = pad.min_size
    hole_r = 0.5 * pad.drill if pad.drill > 0 else 0.4 * dmin
    r = np.hypot(px - pad.center[0], py - pad.center[1])
    inhole = ins & (r <= hole_r)
    pin_pts, prot = [], []
    for i in np.nonzero(inhole)[0]:
        hit = prober.cast(px[i], py[i], s.tht_reach, (0.0, 0.0, -1.0), s.tht_reach + 0.05, side)
        if hit is None:
            continue
        pr = s.tht_reach - hit.distance  # how far the part sticks out beyond the pad here
        if pr > -0.02:                   # pin reaches at least the pad surface
            pin_pts.append(i)
            prot.append(pr)
    if not pin_pts:
        rep.no_pin.append(f"{pad.name}({'F' if side > 0 else 'B'})")
        if not s.hasl:
            rep.skip += 1
            return None
        rep.hasl += 1   # empty hole (mounting hole / wire pad): HASL ring, hole left open
        ins = ins & (r > hole_r)
        return hasl_height(ed, ins, dmin, px, py, rng, s.imperfection), ins, "hasl"
    rep.tht += 1
    vol = float(np.clip(1 + s.imperfection * 0.2 * rng.normal(), 0.7, 1.35))
    if side == pad.component_side:
        H = s.tht_component_side * vol   # solder wicks through: small fillet on the part side
    else:
        H = float(np.clip(0.7 * float(np.median(prot)), 0.25, 0.9)) * vol
    # the pin alone drives the shape: solder wets up the pin/tab and falls away concavely
    idx = np.nonzero(ins)[0]
    src = np.array(pin_pts)
    hf = meniscus(px, py, idx, src, np.full(len(src), H), ed, 2.2)
    hgt = np.zeros(px.shape)
    hgt[idx] = blend_wetting(hf, ed[idx], dmin, 1.0, s.wetting)
    return smooth_grid(hgt, ins, ny, nx, passes=2), ins, "solder"


def generate(pads: Iterable[Pad], prober: Prober, settings: Optional[Settings] = None,
             rng: Optional[np.random.Generator] = None):
    """Build joints for all pads. Returns (list[Joint], Report)."""
    s = settings or Settings()
    rng = rng if rng is not None else np.random.default_rng(0)
    rep = Report()
    joints = []
    for pad in pads:
        dmin = pad.min_size
        for side in pad.sides:
            px, py, nx, ny = _grid(pad, s)
            ins = point_in_polygon(px, py, pad.outline)
            if ins.sum() < 4:
                rep.skip += 1
                continue
            ed = edge_distance(px, py, pad.outline)
            bucket = "solder"
            if pad.kind == "bare":
                if not s.hasl:
                    continue
                rep.hasl += 1
                bucket = "hasl"
                hgt = hasl_height(ed, ins, dmin, px, py, rng, s.imperfection)
            elif pad.kind == "smd":
                hgt = _smd(pad, side, px, py, ins, ed, ny, nx, prober, s, rng, rep)
            elif pad.kind == "tht":
                res = _tht(pad, side, px, py, ins, ed, ny, nx, prober, s, rng, rep)
                if res is None:
                    continue
                hgt, ins, bucket = res
            else:
                raise ValueError(f"unknown pad kind {pad.kind!r}")
            if s.imperfection > 0 and bucket == "solder":
                hgt = hgt * lumpy(px, py, rng, 0.03 * s.imperfection)

            # Only cells fully inside the pad: the joint already reaches zero height at the
            # outline, so the sub-cell gap to the copper edge is invisible. (Snapping outside
            # nodes onto the outline folds the mesh and looks like crumpled foil.)
            ins2 = ins.reshape(ny, nx)
            a = ins2[:-1, :-1] & ins2[1:, :-1] & ins2[:-1, 1:] & ins2[1:, 1:]
            jj, ii = np.nonzero(a)
            if len(jj) == 0:
                continue
            v00 = jj * nx + ii
            quads = np.stack([v00, v00 + 1, v00 + nx + 1, v00 + nx], 1)
            used = np.unique(quads)
            remap = -np.ones(nx * ny, dtype=np.int64)
            remap[used] = np.arange(len(used))
            joints.append(Joint(pad, side, bucket, px[used], py[used], hgt[used] + s.lift, remap[quads]))
    return joints, rep
