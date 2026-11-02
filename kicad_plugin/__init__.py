"""KiCad action plugin: render the open board with pcb-render (Blender Cycles).

Install: symlink this folder into KiCad's plugin dir, e.g.
  ln -s ~/dev/kicad-blender-render/kicad_plugin ~/.local/share/kicad/10.0/scripting/plugins/pcb_render
then PCB Editor -> Tools -> External Plugins -> Refresh Plugins.
"""
import json
import os
import shlex
import subprocess

import pcbnew
import wx

HERE = os.path.dirname(os.path.realpath(__file__))
PCB_RENDER = os.path.join(os.path.dirname(HERE), "pcb-render")

QUALITY = {
    "Preview (fast)": ["--samples", "32", "--res", "1280x800"],
    "Good": ["--samples", "128", "--res", "1920x1200"],
    "Final": ["--samples", "384", "--res", "2560x1600"],
}
SHOTS = ["hero", "top", "low", "macro (selected part)", "standard set + contact sheet", "quick set (hero F/B)"]


def saved_viewports(pcb_path):
    try:
        pro = json.load(open(os.path.splitext(pcb_path)[0] + ".kicad_pro"))
        vps = pro.get("board", {}).get("3dviewports", []) or pro.get("3dviewports", [])
        return [v.get("name") for v in vps if v.get("name")]
    except Exception:
        return []


class RenderDialog(wx.Dialog):
    def __init__(self, parent, board_name, selected, viewports):
        super().__init__(parent, title=f"Blender render: {board_name}")
        g = wx.FlexGridSizer(0, 2, 8, 12)
        self.shot = wx.Choice(self, choices=SHOTS)
        self.shot.SetSelection(3 if selected else 0)
        self.side = wx.Choice(self, choices=["Front (F)", "Back (B)"])
        self.side.SetSelection(0)
        self.focus = wx.TextCtrl(self, value=selected or "")
        self.viewport = wx.Choice(self, choices=["(none)"] + viewports)
        self.viewport.SetSelection(0)
        self.quality = wx.Choice(self, choices=list(QUALITY))
        self.quality.SetSelection(0)
        self.imp = wx.SpinCtrlDouble(self, min=0.0, max=1.0, initial=0.6, inc=0.1)
        for label, ctrl in (("Shot", self.shot), ("Side", self.side), ("Focus part (ref)", self.focus),
                            ("KiCad 3D viewport", self.viewport), ("Quality", self.quality),
                            ("Imperfection 0-1", self.imp)):
            g.Add(wx.StaticText(self, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
            g.Add(ctrl, 1, wx.EXPAND)
        g.AddGrowableCol(1)
        note = wx.StaticText(self, label="Renders the board as last saved. Runs in the background;\n"
                                          "the image opens when it is done.")
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(g, 0, wx.ALL | wx.EXPAND, 12)
        box.Add(note, 0, wx.LEFT | wx.RIGHT, 12)
        box.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.ALL | wx.ALIGN_RIGHT, 12)
        self.SetSizerAndFit(box)

    def args(self):
        shot = SHOTS[self.shot.GetSelection()]
        side = "F" if self.side.GetSelection() == 0 else "B"
        a = []
        if shot.startswith("standard"):
            a += ["--set", "standard"]
        elif shot.startswith("quick"):
            a += ["--set", "quick"]
        else:
            vp = self.viewport.GetSelection()
            if vp > 0:
                a += ["--viewport", self.viewport.GetString(vp)]
            elif shot.startswith("macro"):
                a += ["--preset", "macro"]
            else:
                a += ["--preset", shot]
            a += ["--side", side]
            ref = self.focus.GetValue().strip()
            if ref and (shot.startswith("macro") or vp > 0):
                a += ["--focus", ref]
        a += QUALITY[self.quality.GetString(self.quality.GetSelection())]
        a += ["--imperfection", f"{self.imp.GetValue():.2f}"]
        return a


class PcbRenderPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Blender render (pcb-render)"
        self.category = "Render"
        self.description = "Photoreal Cycles render with solder joints, via ~/dev/kicad-blender-render"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(HERE, "pcb_render.png")

    def Run(self):
        board = pcbnew.GetBoard()
        path = board.GetFileName()
        parent = wx.FindWindowByName("PcbFrame") or None
        if not path or not os.path.exists(path):
            wx.MessageBox("Save the board first.", "Blender render", wx.ICON_WARNING)
            return
        selected = [fp.GetReference() for fp in board.GetFootprints() if fp.IsSelected()]
        name = os.path.splitext(os.path.basename(path))[0]
        dlg = RenderDialog(parent, name, selected[0] if selected else "", saved_viewports(path))
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        args = dlg.args()
        dlg.Destroy()
        if not os.path.exists(PCB_RENDER):
            wx.MessageBox(f"pcb-render not found at {PCB_RENDER}", "Blender render", wx.ICON_ERROR)
            return
        log_dir = os.path.join(os.path.dirname(HERE), "out", name)
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(log_dir, "kicad-plugin.log")
        cmd = " ".join(shlex.quote(c) for c in [PCB_RENDER, path] + args)
        # background job: render, then notify + open the result (last line of the log)
        script = (
            f"{cmd} > {shlex.quote(log)} 2>&1; "
            f"out=$(tail -1 {shlex.quote(log)}); "
            f'if [ -s "$out" ]; then notify-send -i image-x-generic "pcb-render done" "$out"; xdg-open "$out"; '
            f'else notify-send -u critical "pcb-render failed" "see {log}"; fi'
        )
        subprocess.Popen(["bash", "-c", script], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wx.MessageBox(f"Rendering in the background:\n{' '.join(args)}\n\nLog: {log}",
                      "Blender render", wx.ICON_INFORMATION)


PcbRenderPlugin().register()
