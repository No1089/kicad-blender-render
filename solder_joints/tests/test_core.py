# SPDX-License-Identifier: LGPL-3.0-or-later
"""Blender-free tests for solder_joints.core, using a box-geometry prober.

Run:  python3 -m pytest solder_joints/tests   (or just: python3 solder_joints/tests/test_core.py)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from solder_joints import Hit, Pad, Settings, generate, pad_outline  # noqa: E402


class BoxProber:
    """Axis-aligned boxes in board-frame mm: (xmin, xmax, ymin, ymax, hmin, hmax, metal)."""

    def __init__(self, boxes):
        self.boxes = boxes

    def cast(self, x, y, h, d, max_dist, side):
        best = None
        o = np.array([x, y, h], float)
        d = np.array(d, float)
        for (x0, x1, y0, y1, h0, h1, metal) in self.boxes:
            lo, hi = np.array([x0, y0, h0]), np.array([x1, y1, h1])
            inside = bool(np.all(o >= lo) and np.all(o <= hi))
            tmin, tmax, miss = -np.inf, np.inf, False
            for k in range(3):  # slab test
                if abs(d[k]) < 1e-12:
                    if o[k] < lo[k] or o[k] > hi[k]:
                        miss = True
                        break
                    continue
                ta, tb = sorted(((lo[k] - o[k]) / d[k], (hi[k] - o[k]) / d[k]))
                tmin, tmax = max(tmin, ta), min(tmax, tb)
            if miss or tmax < max(tmin, 0):
                continue
            t, exiting = (tmax, True) if inside else (tmin, False)
            if 0 <= t <= max_dist and (best is None or t < best.distance):
                best = Hit(distance=float(t), exiting=exiting, metal=metal, h=float(h + d[2] * t))
        return best


def rect(cx, cy, w, h):
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                     [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def joint_height_at(joint, x, y):
    i = np.argmin(np.hypot(joint.x - x, joint.y - y))
    return joint.h[i]


def test_chip_end_cap_fillet():
    # 0.5 x 0.5 mm pad at x=0; a 0.35 mm tall metal end cap covers x in [-0.25, 0.0]
    pad = Pad(rect(0, 0, 0.5, 0.5), (0, 0), "smd", (1,), part_center=(-1.0, 0.0), name="R1.1")
    prober = BoxProber([(-0.25, 0.0, -0.25, 0.25, 0.0, 0.35, True)])
    joints, rep = generate([pad], prober, Settings())
    assert rep.fillet == 1 and len(joints) == 1
    j = joints[0]
    under_cap = joint_height_at(j, -0.05, 0.0)
    at_cap = joint_height_at(j, 0.01, 0.0)
    near_edge = joint_height_at(j, 0.24, 0.0)
    mid = joint_height_at(j, 0.12, 0.0)
    # ~42 % of the 0.35 mm cap (0.15 mm) at the terminal, rounded off by the surface-tension
    # smoothing right at the cap face, concave fall-off, ~0 at the pad edge
    assert 0.13 < under_cap < 0.17, under_cap
    assert 0.11 < at_cap <= under_cap + 1e-9, at_cap
    assert near_edge < 0.02, near_edge
    assert near_edge < mid < at_cap
    assert mid < (at_cap + near_edge) / 2, "meniscus should be concave"


def test_plastic_body_does_not_pull_solder_up():
    pad = Pad(rect(0, 0, 0.5, 0.5), (0, 0), "smd", (1,), name="U1.1")
    # tall plastic body 0.3 mm above the pad: not a standoff gap, not metal -> wetting only
    prober = BoxProber([(-0.25, 0.25, -0.25, 0.25, 0.3, 1.0, False)])
    joints, rep = generate([pad], prober, Settings(gap_max=0.5))
    assert joints[0].h.max() < 0.03


def test_leadless_standoff_gap_is_filled():
    pad = Pad(rect(0, 0, 0.5, 0.5), (0, 0), "smd", (1,), name="U2.1")
    # plastic underside 0.05 mm above the inner half of the pad
    prober = BoxProber([(-0.25, 0.0, -0.25, 0.25, 0.05, 0.9, False)])
    joints, _ = generate([pad], prober, Settings())
    assert 0.04 < joints[0].h.max() < 0.08


def test_leadless_flank_toe_fillet():
    # DFN-style: plastic body over the inner part of the pad (0.05 mm standoff), with a metal
    # flank on the body side face at x = 0 (0.02 mm thick, 0.05..0.3 mm high)
    pad = Pad(rect(0.1, 0, 0.5, 0.3), (0.1, 0), "smd", (1,), part_center=(-1.0, 0.0), name="U3.1")
    prober = BoxProber([(-0.15, -0.02, -0.15, 0.15, 0.05, 0.8, False),
                        (-0.02, 0.0, -0.12, 0.12, 0.05, 0.3, True)])
    joints, rep = generate([pad], prober, Settings())
    j = joints[0]
    toe = joint_height_at(j, 0.02, 0.0)
    out = joint_height_at(j, 0.3, 0.0)
    assert 0.08 < toe < 0.2, toe        # climbs the flank a little
    assert out < 0.03, out              # and falls away to the pad edge


def test_tht_pin_cone_and_component_side():
    ring = np.array([[math.cos(a) * 0.8, math.sin(a) * 0.8] for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)])
    pad = Pad(ring, (0, 0), "tht", (1, -1), drill=1.0, component_side=1, name="J1.1")
    pin = (-0.32, 0.32, -0.32, 0.32, -5.0, 5.0, True)  # 0.64 mm square pin through both sides

    class SideProber(BoxProber):
        def cast(self, x, y, h, d, max_dist, side):
            # pin sticks out 1.5 mm on the solder side (-1); flush-ish on the component side
            top = 1.5 if side < 0 else 0.0
            return BoxProber([(*pin[:4], -5.0, top, True)]).cast(x, y, h, d, max_dist, side)

    joints, rep = generate([pad], SideProber([]), Settings())
    assert rep.tht == 2
    by_side = {j.side: j for j in joints}
    solder_side = by_side[-1]
    comp_side = by_side[1]
    assert joint_height_at(solder_side, 0.35, 0) > 0.6        # clings to the pin
    assert joint_height_at(solder_side, 0.78, 0) < 0.05       # thin at the ring edge
    assert comp_side.h.max() < 0.25                           # small fillet on the part side


def test_hasl_and_missing_pin():
    ring = np.array([[math.cos(a) * 1.5, math.sin(a) * 1.5] for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)])
    hole = Pad(ring, (0, 0), "tht", (1,), drill=2.0, name="H1.1")
    bare = Pad(rect(5, 0, 1.0, 1.0), (5, 0), "bare", (1,), name="TP1.1")
    joints, rep = generate([hole, bare], BoxProber([]), Settings(hasl=True))
    assert rep.hasl == 2 and rep.no_pin == ["H1.1(F)"]
    assert all(j.bucket == "hasl" and j.h.max() < 0.04 for j in joints)
    ring_j = [j for j in joints if j.pad is hole][0]
    assert np.hypot(ring_j.x, ring_j.y).min() > 0.95          # hole left open
    joints, rep = generate([hole], BoxProber([]), Settings(hasl=False))
    assert joints == [] and rep.skip == 1


def test_pad_outline_shapes():
    for shape, size in (("rect", (1.0, 0.5)), ("roundrect", (1.0, 0.5)), ("circle", (0.8, 0.8)), ("oval", (1.2, 0.6))):
        P = pad_outline(shape, size, position=(2.0, 3.0), rotation=math.pi / 2)
        span = P.max(0) - P.min(0)
        assert abs(span[0] - size[1]) < 1e-6 and abs(span[1] - size[0]) < 1e-6, (shape, span)  # rotated 90 deg
        assert np.allclose((P.max(0) + P.min(0)) / 2, (2.0, 3.0))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
