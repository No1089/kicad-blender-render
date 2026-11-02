# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Chris Fourie (https://github.com/No1089)
"""Blender side of solder_joints: board frame mapping, BVH prober, meshes and a solder material.

Typical use (inside Blender):

    from solder_joints import Pad, Settings, generate
    from solder_joints.blender import BoardFrame, BVHProber, build_objects, solder_material

    frame = BoardFrame(origin=(0, 0), mm=0.001, z_top=0.0016, z_bottom=0.0, flip_y=True)
    prober = BVHProber(component_objects, metal_material_names, frame)
    joints, report = generate(pads, prober, Settings(imperfection=0.5))
    objs = build_objects(joints, frame, {"solder": solder_material(frame.mm)})
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .core import Hit, Joint


@dataclass
class BoardFrame:
    """Maps board millimetres (x right, y down or up) to Blender world coordinates.

    world_x = origin_x + x * mm
    world_y = origin_y + (-y if flip_y else y) * mm      (KiCad's y points down -> flip_y=True)
    The copper surfaces are at z_top (side +1) and z_bottom (side -1), in world units.
    Alternatively pass `xy`, a callable (x_mm, y_mm) -> (X, Y) arrays, for any other mapping.
    """
    origin: tuple = (0.0, 0.0)
    mm: float = 0.001
    z_top: float = 0.0016
    z_bottom: float = 0.0
    flip_y: bool = True
    xy: Optional[Callable] = None

    def to_world_xy(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.xy is not None:
            return self.xy(x, y)
        return self.origin[0] + x * self.mm, self.origin[1] + (-y if self.flip_y else y) * self.mm

    def surface(self, side):
        return self.z_top if side > 0 else self.z_bottom

    def to_world(self, x, y, h, side):
        X, Y = self.to_world_xy(x, y)
        return X, Y, self.surface(side) + side * np.asarray(h) * self.mm

    def direction(self, x, y, d, side):
        """Board-frame unit direction (dx, dy, dh) at (x, y) -> world unit Vector."""
        dx, dy, dh = d
        if dx or dy:
            X0, Y0 = self.to_world_xy([x], [y])
            X1, Y1 = self.to_world_xy([x + dx * 0.01], [y + dy * 0.01])
            v = Vector((float(X1[0] - X0[0]), float(Y1[0] - Y0[0]), 0.0))
            v.normalize()
            return v
        return Vector((0.0, 0.0, side * dh))

    @property
    def mirrored(self):
        """True if board x/y -> world x/y flips handedness (affects face winding)."""
        X, Y = self.to_world_xy([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        ex = (X[1] - X[0], Y[1] - Y[0])
        ey = (X[2] - X[0], Y[2] - Y[0])
        return ex[0] * ey[1] - ex[1] * ey[0] < 0


def _world_verts(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(obj.matrix_world)
    return co @ M[:3, :3].T + M[:3, 3]


class BVHProber:
    """Ray queries against component meshes, with per-triangle 'is metal' flags.

    metal: a set of material names, or a callable(material) -> bool, deciding which surfaces
    solder wets (leads, end caps, plated flanks). Everything else counts as plastic.
    """

    def __init__(self, objects: Iterable, metal, frame: BoardFrame):
        self.frame = frame
        is_metal = metal if callable(metal) else (lambda m, names=set(metal): m.name in names)
        verts, tris, flags, off = [], [], [], 0
        for o in objects:
            if o.type != "MESH" or not len(o.data.vertices):
                continue
            me = o.data
            me.calc_loop_triangles()
            nt = len(me.loop_triangles)
            t = np.empty(nt * 3, dtype=np.int64)
            me.loop_triangles.foreach_get("vertices", t)
            mi = np.empty(nt, dtype=np.int64)
            me.loop_triangles.foreach_get("material_index", mi)
            slot_metal = np.array([bool(sl.material and is_metal(sl.material)) for sl in o.material_slots]
                                  or [False])
            v = _world_verts(o)
            verts.append(v)
            tris.append(t.reshape(-1, 3) + off)
            flags.append(slot_metal[np.clip(mi, 0, len(slot_metal) - 1)])
            off += len(v)
        if not verts:
            self.bvh, self.metal = None, np.zeros(0, bool)
            return
        self.bvh = BVHTree.FromPolygons(np.concatenate(verts).tolist(), np.concatenate(tris).tolist(),
                                        all_triangles=True)
        self.metal = np.concatenate(flags)

    @property
    def triangle_count(self):
        return len(self.metal)

    def cast(self, x, y, h, direction, max_dist, side):
        if self.bvh is None:
            return None
        f = self.frame
        X, Y, Z = f.to_world([x], [y], [h], side)
        o = Vector((float(X[0]), float(Y[0]), float(Z[0])))
        d = f.direction(x, y, direction, side)
        loc, nrm, idx, dist = self.bvh.ray_cast(o, d, max_dist * f.mm)
        if loc is None:
            return None
        hit_h = (loc.z - f.surface(side)) * side / f.mm
        return Hit(distance=dist / f.mm, exiting=nrm.dot(d) > 0, metal=bool(self.metal[idx]), h=hit_h)


def build_objects(joints: Iterable[Joint], frame: BoardFrame, materials: Optional[Dict] = None,
                  collection=None, names=None) -> Dict[str, "bpy.types.Object"]:
    """One mesh object per bucket ("solder", "hasl"), smooth-shaded, duplicate verts merged."""
    import bmesh
    names = names or {"solder": "Solder", "hasl": "HASL"}
    collection = collection or bpy.context.scene.collection
    buckets = {}
    for j in joints:
        X, Y, Z = frame.to_world(j.x, j.y, j.h, j.side)
        q = j.quads[:, ::-1] if (j.side < 0) != frame.mirrored else j.quads
        V, F, off = buckets.setdefault(j.bucket, ([], [], [0]))
        V.append(np.stack([X, Y, Z], 1))
        F.append(q + off[0])
        off[0] += len(X)
    objs = {}
    for bucket, (V, F, _) in buckets.items():
        name = names.get(bucket, bucket)
        me = bpy.data.meshes.new(name)
        me.from_pydata(np.concatenate(V).tolist(), [], np.concatenate(F).tolist())
        me.update()
        for p in me.polygons:
            p.use_smooth = True
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005 * frame.mm)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        collection.objects.link(ob)
        if materials and bucket in materials:
            me.materials.append(materials[bucket])
        objs[bucket] = ob
    return objs


def solder_material(mm=0.001, imperfection=0.6, name="Solder"):
    """SAC305 after reflow: bright, slightly frosty metal with soft ripples (Principled BSDF)."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(b.outputs[0], out.inputs[0])
    b.inputs["Base Color"].default_value = (0.80, 0.80, 0.82, 1)
    b.inputs["Metallic"].default_value = 1.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 5.0 / mm
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.015 + 0.03 * imperfection
    bump.inputs["Distance"].default_value = 0.004 * mm
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    base = 0.12 + 0.1 * imperfection
    var = 0.03 + 0.07 * imperfection
    mr.inputs["To Min"].default_value = base - var
    mr.inputs["To Max"].default_value = base + var
    nt.links.new(noise.outputs["Fac"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    return m
