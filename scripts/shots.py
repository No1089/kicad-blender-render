#!/usr/bin/env python3
"""Print the shot list for a board, one shot per line (pcb-render arguments).

  shots.py pads.json standard     hero F/B, top F/B, biggest IC per side, THT solder side
  shots.py pads.json quick        hero F/B only
  shots.py pads.json standard --json OUTDIR NAME
                                  same, as a JSON list for render.py --shots, with output paths
"""
import json
import sys


def out_name(shot_args, outdir, board):
    """Same naming as single renders: <board>_<preset>[-<ref>|vp-<name>]_<side>.png"""
    a = dict(zip(shot_args[::2], shot_args[1::2]))
    preset, side = a.get("--preset", "hero"), a.get("--side", "F")
    if "--focus" in a:
        preset = f"{preset}-{a['--focus']}"
    if "--viewport" in a:
        preset = f"vp-{a['--viewport']}"
    return f"{outdir}/{board}_{preset}_{side}.png"


def as_json(lines, outdir, board):
    keymap = {"--preset": "preset", "--side": "side", "--focus": "focus", "--zoom": "zoom",
              "--viewport": "viewport"}
    out = []
    for line in lines:
        t = line.split()
        d = {}
        for k, v in zip(t[::2], t[1::2]):
            d[keymap[k]] = float(v) if k == "--zoom" else v
        d["out"] = out_name(t, outdir, board)
        out.append(d)
    return json.dumps(out, indent=1)


def main(pads_path, name):
    d = json.load(open(pads_path))
    onboard = {f["ref"] for f in d.get("footprints", []) if f.get("onboard", True)}
    side_of = {f["ref"]: f["side"] for f in d.get("footprints", [])}
    pads = [p for p in d["pads"] if p["ref"] in onboard and p["copper"]]
    count, tht = {}, {}
    for p in pads:
        count[p["ref"]] = count.get(p["ref"], 0) + 1
        if p["tht"]:
            tht[p["ref"]] = tht.get(p["ref"], 0) + 1
    sides = sorted({side_of[r] for r in count})  # sides that have parts

    shots = [f"--preset hero --side {s}" for s in ("F", "B") if s in sides or s == "F"]
    if name == "quick":
        return shots
    shots += [f"--preset top --side {s}" for s in ("F", "B") if s in sides or s == "F"]

    # biggest IC (most pads, U*/IC* refs, SMD) on each side
    for s in ("F", "B"):
        ics = [(n, r) for r, n in count.items()
               if side_of[r] == s and r.rstrip("0123456789").upper() in ("U", "IC", "AC") and r not in tht]
        if ics:
            ref = max(ics)[1]
            shots.append(f"--preset macro --focus {ref} --side {s}")
    # through-hole part with most pins, from its solder side
    if tht:
        ref = max((n, r) for r, n in tht.items())[1]
        solder_side = "B" if side_of[ref] == "F" else "F"
        shots.append(f"--preset macro --focus {ref} --side {solder_side} --zoom 0.8")
    return list(dict.fromkeys(shots))


if __name__ == "__main__":
    lines = main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "standard")
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        print(as_json(lines, sys.argv[i + 1], sys.argv[i + 2]))
    else:
        print("\n".join(lines))
