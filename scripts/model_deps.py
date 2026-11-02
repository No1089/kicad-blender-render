#!/usr/bin/env python3
"""List the 3D model files a .kicad_pcb references, resolved to real paths.

  model_deps.py BOARD.kicad_pcb              one resolved path per line (missing ones prefixed "missing ")
  model_deps.py BOARD.kicad_pcb --newer F    print the newest model newer than file F (and exit 0),
                                             or nothing (exit 1); used by pcb-render to invalidate the GLB
  model_deps.py BOARD.kicad_pcb --rewrite OUT.kicad_pcb
                                             copy of the board with every model path resolved to an
                                             existing absolute file, for kicad-cli (KiCad 10 does not
                                             expand ${KICAD6_3DMODEL_DIR} etc. from older boards)

Resolves ${KIPRJMOD}, KiCad's stock model dirs (${KICADn_3DMODEL_DIR}), the user's path variables
from kicad_common.json, process environment variables, and relative paths (to the project). Like
`kicad-cli pcb export glb --subst-models`, a .wrl reference also depends on a STEP of the same name.
"""
import glob
import json
import os
import re
import sys


def kicad_vars(pcb_path):
    v = {}
    for d in sorted(glob.glob("/usr/share/kicad/3dmodels")) + sorted(glob.glob("/usr/share/kicad*/3dmodels")):
        for n in range(6, 12):
            v.setdefault(f"KICAD{n}_3DMODEL_DIR", d)
    for cfg in sorted(glob.glob(os.path.expanduser("~/.config/kicad/*/kicad_common.json"))):
        try:
            v.update(json.load(open(cfg)).get("environment", {}).get("vars") or {})
        except Exception:
            pass
    v.update({k: val for k, val in os.environ.items() if k.startswith("KICAD") or k in v})
    v["KIPRJMOD"] = os.path.dirname(os.path.abspath(pcb_path))
    return v


def resolve(ref, variables, base):
    if ref.startswith("kicad-embed://"):
        return None  # embedded in the board file itself: covered by the board's own mtime
    path = re.sub(r"\$\{([^}]+)\}", lambda m: variables.get(m.group(1), m.group(0)), ref)
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return os.path.normpath(path)


def model_files(pcb_path):
    txt = open(pcb_path, encoding="utf-8").read()
    refs = sorted(set(re.findall(r'\(model "([^"]+)"', txt)))
    variables = kicad_vars(pcb_path)
    base = os.path.dirname(os.path.abspath(pcb_path))
    out = []
    for ref in refs:
        p = resolve(ref, variables, base)
        if p is None:
            continue
        cands = [p]
        stem, ext = os.path.splitext(p)
        if ext.lower() == ".wrl":  # --subst-models swaps in a STEP with the same name
            cands += [stem + e for e in (".step", ".stp", ".STEP", ".STP")]
        out += cands
    return out


def best_file(ref, variables, base):
    """Existing file for a model reference: as named, else a STEP of the same name (or None)."""
    p = resolve(ref, variables, base)
    if p is None:
        return None
    stem, ext = os.path.splitext(p)
    for c in [p] + [stem + e for e in (".step", ".stp", ".STEP", ".STP", ".wrl")]:
        if os.path.exists(c):
            return c
    return None


def rewrite(pcb_path, out_path):
    txt = open(pcb_path, encoding="utf-8").read()
    variables = kicad_vars(pcb_path)
    base = os.path.dirname(os.path.abspath(pcb_path))
    stats = {"rewritten": 0, "unchanged": 0, "missing": []}

    def sub(m):
        ref = m.group(1)
        f = best_file(ref, variables, base)
        if f is None:
            if not ref.startswith("kicad-embed://"):
                stats["missing"].append(ref)
            stats["unchanged"] += 1
            return m.group(0)
        stats["rewritten"] += 1
        return '(model "' + f.replace("\\", "/").replace('"', '\\"') + '"'

    out = re.sub(r'\(model "([^"]+)"', sub, txt)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    msg = f"models: {stats['rewritten']} resolved"
    if stats["missing"]:
        msg += f", {len(stats['missing'])} not found: " + ", ".join(sorted(set(stats["missing"]))[:5])
    print(msg, file=sys.stderr)


def main():
    pcb = sys.argv[1]
    if "--rewrite" in sys.argv:
        rewrite(pcb, sys.argv[sys.argv.index("--rewrite") + 1])
        return 0
    files = model_files(pcb)
    if "--newer" in sys.argv:
        ref = sys.argv[sys.argv.index("--newer") + 1]
        t0 = os.path.getmtime(ref) if os.path.exists(ref) else 0
        newest = max(((os.path.getmtime(f), f) for f in files if os.path.exists(f)), default=None)
        if newest and newest[0] > t0:
            print(newest[1])
            return 0
        return 1
    for f in files:
        print(f if os.path.exists(f) else f"missing {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
