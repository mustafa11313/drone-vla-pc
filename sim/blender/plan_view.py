"""plan_view.py  -  top view of the lab with every object labelled, to check positions.

    OUT=lab_plan.png python3 sim/blender/plan_view.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["x"]
import build_lab as B
import bpy
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont
B.build()
GROUPS = [  # prefix, label, colour
 ("cabinet", "cabinet", (150,150,150)), ("cab_screen", None, (40,40,40)),
 ("yellow_robot", "Fanuc (yellow)", (230,180,0)), ("fanuc", None, (200,200,200)),
 ("fanuc_controller", "Fanuc controller", (210,190,140)), ("control_grey", "grey box", (110,110,110)),
 ("control_white", "white unit", (225,225,225)), ("duct_down", "duct", (190,190,190)),
 ("white_robot", "ABB (big white arm)", (240,240,240)), ("abb_pallet", "pallet", (190,140,80)),
 ("abb_controller", "ABB controller", (110,110,110)), ("crate", "blue wrapped crate", (40,70,140)),
 ("mobile_base", "mobile robot base", (70,70,70)), ("cobot_table", "UR cobot table", (180,180,190)), ("cobot_", None, (220,225,235)),
 ("spindle", "spindle", (30,30,30)), ("euro", "grey euro boxes", (130,135,140)),
 ("airtech", "Airtech boxes", (170,125,80)), ("mats_out", "grey mats", (60,60,60)), ("booster_k1", "Booster K1", (60,60,60)),
 ("carton_e", "carton + black tray", (170,125,80)), ("board_e", "wooden board", (190,140,80)),
 ("table_e", "table + monitor", (220,220,220)), ("desk_s2", "desk row", (200,160,110)), ("desk_s", "desk, PCs, monitors", (200,160,110)),
 ("desk_se", "desk, PC, monitor", (200,160,110)), ("humanoid", "humanoid on stand", (60,60,60)),
 ("crane", "crane", (30,30,30)), ("desk_w", "desk", (200,160,110)), ("desk_e", "desk", (200,160,110)),
 ("cases", "black cases", (30,30,30)), ("orange_bench", "orange bench", (230,110,30)), ("chair", None, (40,40,40)),
 ("cage_mats", "CAGE 2 x 2 m", (25,25,25)),
]
X0, X1, Y0, Y1, PX = -5.5, 2.0, -5.5, 2.5, 110
img = Image.new("RGB", (int((X1-X0)*PX), int((Y1-Y0)*PX)), (245,245,242))
d = ImageDraw.Draw(img)
def P(x, y): return ((x-X0)*PX, (Y1-y)*PX)
try:
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except Exception:
    f = ImageFont.load_default()
for gx in range(int(X0), int(X1)+1):
    d.line([P(gx, Y0), P(gx, Y1)], fill=(225,225,225))
for gy in range(int(Y0), int(Y1)+1):
    d.line([P(X0, gy), P(X1, gy)], fill=(225,225,225))
d.rectangle([P(X0, 2.5), P(X1, 2.3)], fill=(160,160,160)); d.text(P(-0.6, 2.49), "north wall", fill="black", font=f)
d.rectangle([P(B.ROOM["xmax"], 2.5), P(X1, Y0)], fill=(160,160,160))
labels = {}
for o in bpy.data.objects:
    if o.type != "MESH": continue
    for pre, lab, col in sorted(GROUPS, key=lambda g: -len(g[0])):
        if o.name.startswith(pre):
            cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
            xs, ys = [c.x for c in cs], [c.y for c in cs]
            a, b = P(min(xs), max(ys)), P(max(xs), min(ys))
            d.rectangle([a, b], fill=col, outline=(0,0,0))
            if lab:
                L = labels.setdefault(lab + pre, [lab, 1e9, -1e9, 1e9, -1e9])
                L[1] = min(L[1], min(xs)); L[2] = max(L[2], max(xs)); L[3] = min(L[3], min(ys)); L[4] = max(L[4], max(ys))
            break
# the pad and drone-cage outline
d.rectangle([P(-1, 1), P(1, -1)], outline=(200,0,0), width=3)
for lab, x0, x1, y0, y1 in labels.values():
    tx, ty = P((x0 + x1) / 2, (y0 + y1) / 2)
    w = d.textlength(lab, font=f)
    d.rectangle([tx - w/2 - 3, ty - 9, tx + w/2 + 3, ty + 9], fill=(255,255,255))
    d.text((tx - w/2, ty - 8), lab, fill=(0,0,0), font=f)
d.text((10, img.height - 25), "Top view, 1 m grid. North (wall with the lab sign) is up; you stand at the desks south of the cage.", fill="black", font=f)
img.save(os.environ.get("OUT", "lab_plan.png"))
