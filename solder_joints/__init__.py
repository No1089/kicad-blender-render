# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Chris Fourie (https://github.com/No1089)
"""solder_joints: realistic solder joint meshes for 3D PCB renders.

The core (`solder_joints.core`) is pure NumPy and Blender-independent; `solder_joints.blender`
adds a BVH-backed prober, mesh building and a solder material. See README.md.
"""
from .core import Hit, Joint, Pad, Prober, Report, Settings, generate, pad_outline

__all__ = ["Hit", "Joint", "Pad", "Prober", "Report", "Settings", "generate", "pad_outline"]
__version__ = "0.1.0"
