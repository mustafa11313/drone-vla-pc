"""
build_lab.py  -  the lab around the flight cage as a real 3D scene in Blender.

    pip install bpy                       # Blender as a Python module (no GPU needed)
    python3 sim/blender/build_lab.py --save sim/blender/lab.blend --previews out/

Builds everything from simple shapes, placed from the lab video:

    cage       2 x 2 m, 2 m high, silver aluminium posts and top frame, black rope net (sagging, with
               the extra net lying on the floor), black worn puzzle mats inside, the pad
    room       light grey floor with red/white tape, white walls, windows, ceiling ducts,
               pipes and strip lights
    around     north (behind the cage): bench with machines, small yellow robot, grey cabinet
               south (where you stand): wooden desk with blue drawers, laptop, second desk
               with the teach pendant and monitor
               west: fume extractor, then the robot area (green robot, white robot on a blue
               wrapped pallet, cobot table)

Coordinates: metres, x = east, y = north, z = up, cage centre on the floor = (0, 0, 0).
All positions are estimates from the video; change them in LAYOUT below.
"""
import argparse, math, os, sys

import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "tex")
SIM_TEX = os.path.join(HERE, "..", "textures")

CAGE_HALF, CAGE_H = 1.0, 2.0
ROOM = dict(xmin=-8.0, xmax=1.5, ymin=-7.0, ymax=2.3, h=4.8)   # east wall ~0.5 m behind the net (video)

# ---------------------------------------------------------------- textures (numpy)


def _save(name, arr):
    from PIL import Image
    os.makedirs(TEX, exist_ok=True)
    p = os.path.join(TEX, name)
    Image.fromarray(arr).save(p)
    return p


def make_net_texture(px=1024, cells=19, strand=0.055, seed=0):
    """RGBA square rope mesh: black strands, transparent holes. `cells` holes per texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:px, 0:px] / px * cells
    # slight wobble so the mesh is not perfectly regular
    x = x + 0.12 * np.sin(y * 0.9 + rng.uniform(0, 6)) + 0.05 * np.sin(y * 3.1) + 0.04 * np.sin(x * 2.3 + y * 1.7)
    y = y + 0.12 * np.sin(x * 0.8 + rng.uniform(0, 6)) + 0.05 * np.sin(x * 2.9) + 0.04 * np.sin(y * 2.1 + x * 1.3)
    d = np.minimum(np.abs(x - np.round(x)), np.abs(y - np.round(y)))
    knot = (np.abs(x - np.round(x)) < strand * 1.6) & (np.abs(y - np.round(y)) < strand * 1.6)
    a = ((d < strand) | knot).astype(np.float32)
    rgba = np.zeros((px, px, 4), np.uint8)
    rgba[..., :3] = 14
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return _save("net.png", rgba)


def make_mat_texture(px_per_m=400, tile=0.6, n=4, seed=1):
    """Mid-grey puzzle mats (n x n tiles of `tile` m) as in the Nov 2026 lab video, a little worn."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    N = int(n * tile * px_per_m)
    y, x = np.mgrid[0:N, 0:N] / px_per_m
    base = np.full((N, N), 0.50, np.float32)
    # fine foam grain
    base += rng.normal(0, 0.02, (N, N)).astype(np.float32)
    # wear: big soft lighter patches
    w = Image.fromarray((rng.random((6, 6)) * 255).astype(np.uint8)).resize((N, N), Image.BICUBIC)
    w = np.asarray(w.filter(ImageFilter.GaussianBlur(N / 20)), np.float32) / 255
    base *= 0.9 + 0.2 * w
    # scuffs
    for _ in range(140):
        cx, cy, L, ang = rng.uniform(0, N), rng.uniform(0, N), rng.uniform(10, 80), rng.uniform(0, math.pi)
        t = np.linspace(-L / 2, L / 2, int(L))
        xs = (cx + t * math.cos(ang)).astype(int) % N
        ys = (cy + t * math.sin(ang)).astype(int) % N
        base[ys, xs] -= 0.05
    # puzzle tiles: each pixel belongs to a tile, whose edges carry square teeth (as in the real
    # mats: ~10 cm period, ~3 cm deep). The seam is drawn where the tile changes, so the zig-zag
    # outline is continuous, and every tile gets its own slightly different tone.
    u, v = x / tile, y / tile
    tooth_u = 0.05 * np.sign(np.sin(v * math.pi * 12 + 0.3))
    tooth_v = 0.05 * np.sign(np.sin(u * math.pi * 12 + 0.3))
    iu = np.floor(u + tooth_u).astype(int) % n
    iv = np.floor(v + tooth_v).astype(int) % n
    tid = iu * n + iv
    tone = rng.uniform(0.9, 1.08, n * n).astype(np.float32)
    base *= tone[tid]
    edge = (tid != np.roll(tid, 1, 0)) | (tid != np.roll(tid, 1, 1))
    edge |= np.roll(edge, 1, 0) | np.roll(edge, 1, 1)                  # 2 px wide gap
    base[edge] *= 0.72
    g = np.clip(base, 0, 1)
    rgb = (np.stack([g, g * 0.93, g * 0.93], -1) * 255).astype(np.uint8)    # slightly warm grey
    return _save("mats.png", rgb), n * tile


def make_lab_floor_texture(px=1024, size=4.0, seed=2):
    """Light grey epoxy floor with faint marks."""
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    g = np.full((px, px), 0.78, np.float32) + rng.normal(0, 0.01, (px, px)).astype(np.float32)
    w = Image.fromarray((rng.random((8, 8)) * 255).astype(np.uint8)).resize((px, px), Image.BICUBIC)
    g += (np.asarray(w.filter(ImageFilter.GaussianBlur(40)), np.float32) / 255 - 0.5) * 0.06
    for _ in range(60):
        cx, cy, L, ang = rng.uniform(0, px), rng.uniform(0, px), rng.uniform(20, 160), rng.uniform(0, math.pi)
        t = np.linspace(-L / 2, L / 2, int(L))
        g[((cy + t * math.sin(ang)).astype(int) % px), ((cx + t * math.cos(ang)).astype(int) % px)] -= 0.12
    g = np.clip(g, 0, 1)
    return _save("labfloor.png", (np.stack([g, g, g * 0.98], -1) * 255).astype(np.uint8)), size


# ---------------------------------------------------------------- materials


def mat_color(name, rgb, rough=0.6, metal=0.0, emit=0.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    return m


def mat_image(name, path, scale=1.0, alpha=False, rough=0.8, gain=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / scale, 1 / scale, 1)
    nt.links.new(tc.outputs["Generated" if scale is None else "Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    if gain != 1.0:
        mul = nt.nodes.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        nt.links.new(tex.outputs["Color"], mul.inputs[6])
        mul.inputs[7].default_value = (gain, gain, gain, 1)
        nt.links.new(mul.outputs[2], b.inputs["Base Color"])
    else:
        nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    if alpha:
        nt.links.new(tex.outputs["Alpha"], b.inputs["Alpha"])
        pass  # Cycles uses the alpha directly
    return m


def mat_uv_image(name, path, rough=0.6, gain=1.0):
    """Image mapped once over the object's UVs (for the pad)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    nt.links.new(tex.outputs["Color"], [i for i in mul.inputs if i.identifier == "A_Color"][0])
    [i for i in mul.inputs if i.identifier == "B_Color"][0].default_value = (gain, gain, gain, 1)
    nt.links.new([o for o in mul.outputs if o.identifier == "Result_Color"][0], b.inputs["Base Color"])
    return m


# ---------------------------------------------------------------- shapes


def _finish(obj, mat, name):
    obj.name = name
    if mat:
        obj.data.materials.append(mat)
    return obj


def box(name, size, loc, mat, rot_z=0.0):
    """size = (sx, sy, sz); loc = centre of the bottom face."""
    sx, sy, sz = size
    bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0], loc[1], loc[2] + sz / 2))
    o = bpy.context.object
    o.scale = (sx, sy, sz)
    o.rotation_euler[2] = math.radians(rot_z)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return _finish(o, mat, name)


def cyl(name, r, h, loc, mat, rot=(0, 0, 0), verts=24):
    """loc = centre."""
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=loc, vertices=verts,
                                        rotation=tuple(math.radians(a) for a in rot))
    return _finish(bpy.context.object, mat, name)


def sphere(name, r, loc, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=20, ring_count=12)
    bpy.ops.object.shade_smooth()
    return _finish(bpy.context.object, mat, name)


def rod(name, a, b, r, mat, verts=12):
    """Cylinder from point a to point b."""
    a, b = Vector(a), Vector(b)
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, location=(a + b) / 2, vertices=verts)
    o = bpy.context.object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    bpy.ops.object.shade_smooth()
    return _finish(o, mat, name)


def tube_curve(name, pts, r, mat):
    """A bent hose through the points."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = r
    cu.bevel_resolution = 4
    sp = cu.splines.new("NURBS")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (*q, 1)
    sp.use_endpoint_u = True
    sp.order_u = 3
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    return o


def plane(name, size, loc, mat, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=tuple(math.radians(a) for a in rot))
    o = bpy.context.object
    o.scale = (size[0], size[1], 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return _finish(o, mat, name)


def grid(name, w, h, nx, ny, loc, rot, mat):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=nx, y_subdivisions=ny, size=1, location=loc,
                                    rotation=tuple(math.radians(a) for a in rot))
    o = bpy.context.object
    o.scale = (w, h, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return _finish(o, mat, name)


# ---------------------------------------------------------------- the scene


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_room(M):
    r = ROOM
    W, D = r["xmax"] - r["xmin"], r["ymax"] - r["ymin"]
    cx, cy = (r["xmin"] + r["xmax"]) / 2, (r["ymin"] + r["ymax"]) / 2
    path, size = make_lab_floor_texture()
    plane("lab_floor", (W, D), (cx, cy, -0.002), mat_image("labfloor", path, scale=size, rough=0.35))
    plane("ceiling", (W, D), (cx, cy, r["h"]), M["ceiling"], rot=(180, 0, 0))
    plane("wall_n", (W, r["h"]), (cx, r["ymax"], r["h"] / 2), M["wall"], rot=(90, 0, 0))
    plane("wall_s", (W, r["h"]), (cx, r["ymin"], r["h"] / 2), M["wall"], rot=(90, 0, 0))
    plane("wall_w", (D, r["h"]), (r["xmin"], cy, r["h"] / 2), M["wall"], rot=(90, 0, 90))
    plane("wall_e", (D, r["h"]), (r["xmax"], cy, r["h"] / 2), M["wall"], rot=(90, 0, 90))
    # windows in the north wall over the robot area, a door, the grey door to the east
    for i, x in enumerate((-6.5, -5.0, -3.5)):
        box(f"window_{i}", (1.3, 0.05, 1.0), (x, r["ymax"] - 0.03, 1.3), M["glass"])
        box(f"window_frame_{i}", (1.4, 0.04, 1.1), (x, r["ymax"] - 0.01, 1.25), M["white_frame"])
    box("big_window_s", (4.0, 0.05, 3.6), (-2.0, r["ymin"] + 0.03, 0.4), M["glass_dark"])
    # columns
    for x in (-2.0,):
        box(f"column_{x}", (0.35, 0.35, r["h"]), (x, r["ymax"] - 0.18, 0), M["wall"])
    # red / white tape lines on the floor
    for (x0, y0, x1, y1) in ((-1.9, -0.2, -1.9, -4.0), (-4.8, -0.9, -2.1, -0.9)):
        L = math.hypot(x1 - x0, y1 - y0)
        n = int(L / 0.1)
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
        for k in range(n):
            t = (k + 0.5) / n
            box(f"tape_{x0}_{y0}_{k}", (L / n, 0.05, 0.001), (x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0),
                M["tape_red"] if k % 2 else M["tape_white"], rot_z=ang)
    # ceiling: two big insulated ducts, pipes, strip lights
    for y, rad in ((-0.6, 0.32), (1.4, 0.28)):
        cyl(f"duct_{y}", rad, W, (cx, y, r["h"] - 0.55), M["duct"], rot=(0, 90, 0))
    for i, (y, z) in enumerate(((0.4, 4.0), (0.55, 4.0), (-1.6, 3.9), (2.0, 3.6))):
        cyl(f"pipe_{i}", 0.03, W, (cx, y, z), M["pipe"], rot=(0, 90, 0), verts=10)
    for x in np.arange(r["xmin"] + 1.0, r["xmax"], 2.5):
        for y in np.arange(r["ymin"] + 1.0, r["ymax"], 2.2):
            box(f"light_{x:.1f}_{y:.1f}", (1.4, 0.12, 0.05), (x, y, r["h"] - 0.35), M["lamp"])
            box(f"lightbox_{x:.1f}_{y:.1f}", (1.45, 0.16, 0.05), (x, y, r["h"] - 0.30), M["metal"])


def build_cage(M, pad_xy=(0.0, 0.0), pad_yaw=0.0):
    h = CAGE_HALF
    path, size = make_mat_texture()
    mats = plane("cage_mats", (2 * h, 2 * h), (0, 0, 0.004), mat_image("mats", path, scale=size, rough=0.85))
    pad = plane("pad", (0.5, 0.5), (pad_xy[0], pad_xy[1], 0.009), mat_uv_image("pad", os.path.join(SIM_TEX, "pad.png"), gain=0.25),
                rot=(0, 0, pad_yaw))
    # posts and top frame: silver aluminium profile with black corner connectors (lab photos)
    t = 0.045
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"post_{sx}{sy}", (t, t, CAGE_H), (sx * h, sy * h, 0), M["alu_profile"])
            box(f"top_corner_{sx}{sy}", (0.09, 0.09, 0.07), (sx * (h - 0.01), sy * (h - 0.01), CAGE_H - 0.075), M["black_plastic"])
            box(f"post_foot_{sx}{sy}", (0.08, 0.08, 0.012), (sx * h, sy * h, 0), M["black_plastic"])
    for s in (-1, 1):
        box(f"top_x_{s}", (2 * h, t, t), (0, s * h, CAGE_H - t), M["alu_profile"])
        box(f"top_y_{s}", (t, 2 * h, t), (s * h, 0, CAGE_H - t), M["alu_profile"])
    # net: subdivided planes with a rope-mesh alpha texture, sagging inwards a little
    netp = make_net_texture(cells=10, strand=0.035)
    net_m = mat_image("net", netp, scale=1.0, alpha=True, rough=0.9)
    sides = [("n", (0, h, CAGE_H / 2), (90, 0, 0)), ("s", (0, -h, CAGE_H / 2), (90, 0, 0)),
             ("e", (h, 0, CAGE_H / 2), (90, 0, 90)), ("w", (-h, 0, CAGE_H / 2), (90, 0, 90))]
    rng = np.random.default_rng(3)
    for nm, loc, rot in sides:
        o = grid(f"net_{nm}", 2 * h, CAGE_H, 24, 30, loc, rot, net_m)
        inward = -Vector(loc).normalized()
        inward.z = 0
        inward.normalize()
        ph = rng.uniform(0, 6, 3)
        for v in o.data.vertices:
            p = v.co
            a = (p.x if abs(rot[2]) < 1 else p.y) / h          # -1..1 along the side
            z = p.z / CAGE_H
            sag = 0.06 * (1 - a * a) * math.sin(math.pi * z) + 0.015 * math.sin(7 * a + ph[0]) * math.sin(5 * z + ph[1])
            v.co = p + inward * sag
        # extra net lying on the floor along the side
        skirt = grid(f"net_skirt_{nm}", 2 * h, 0.25, 12, 3,
                     (loc[0] * 0.88, loc[1] * 0.88, 0.012), (0, 0, rot[2]), net_m)
    top = grid("net_top", 2 * h, 2 * h, 24, 24, (0, 0, CAGE_H), (0, 0, 0), net_m)
    for v in top.data.vertices:      # the roof hangs down in the middle
        v.co.z -= 0.15 * (1 - (v.co.x / h) ** 2) * (1 - (v.co.y / h) ** 2)
    for o in bpy.data.objects:
        if o.name.startswith("net"):
            o.visible_shadow = False          # the real net leaves no visible shadow pattern
    return pad


def desk(M, name, x, y, w, d, rot=0.0, drawers=True):
    """Workbench: beech top, white frame, blue drawer unit."""
    top_h = 0.9
    box(f"{name}_top", (w, d, 0.04), (x, y, top_h - 0.04), M["wood"], rot_z=rot)
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (w / 2 - 0.04), sy * (d / 2 - 0.04)
            box(f"{name}_leg_{sx}{sy}", (0.04, 0.04, top_h - 0.04), (x + c * lx - s * ly, y + s * lx + c * ly, 0), M["white_frame"])
    box(f"{name}_shelf", (w - 0.1, d - 0.1, 0.02), (x, y, 0.15), M["white_frame"], rot_z=rot)
    if drawers:
        lx = -w / 2 + 0.35
        box(f"{name}_drawers", (0.55, d - 0.1, 0.45), (x + c * lx, y + s * lx, top_h - 0.5), M["blue"], rot_z=rot)


def monitor(M, name, x, y, z, rot=0.0, w=0.55):
    box(f"{name}_screen", (w, 0.03, w * 0.6), (x, y, z + 0.12), M["black_plastic"], rot_z=rot)
    box(f"{name}_stand", (0.06, 0.06, 0.14), (x, y, z), M["black_plastic"], rot_z=rot)
    box(f"{name}_foot", (0.22, 0.16, 0.01), (x, y, z), M["black_plastic"], rot_z=rot)


def arm_robot(M, name, base, colour, scale=1.0, yaw=0.0, pose=(30, -40, 60)):
    """Industrial 6-axis arm from simple shapes: base, turret, upper arm, forearm, wrist."""
    s = scale
    x, y = base
    cyl(f"{name}_base", 0.22 * s, 0.25 * s, (x, y, 0.125 * s), M["black_metal"])
    cyl(f"{name}_turret", 0.18 * s, 0.35 * s, (x, y, 0.42 * s), colour)
    yawr = math.radians(yaw)
    fwd = Vector((math.cos(yawr), math.sin(yawr), 0))
    shoulder = Vector((x, y, 0.62 * s))
    a1 = math.radians(pose[0])
    elbow = shoulder + (fwd * math.sin(a1) + Vector((0, 0, math.cos(a1)))) * 0.75 * s
    a2 = math.radians(pose[1]) + a1
    wrist = elbow + (fwd * math.cos(a2) + Vector((0, 0, math.sin(a2)))) * 0.65 * s
    a3 = math.radians(pose[2])
    tool = wrist + (fwd * math.cos(a3) - Vector((0, 0, math.sin(a3)))) * 0.18 * s
    rod(f"{name}_upper", shoulder, elbow, 0.11 * s, colour)
    sphere(f"{name}_elbow", 0.13 * s, elbow, colour)
    rod(f"{name}_fore", elbow, wrist, 0.08 * s, colour)
    sphere(f"{name}_wrist", 0.08 * s, wrist, colour)
    rod(f"{name}_tool", wrist, tool, 0.045 * s, M["black_metal"])
    sphere(f"{name}_shoulder", 0.16 * s, shoulder, colour)


def cobot(M, name, base, z0, yaw=0.0):
    """UR-style cobot: thin white/grey links with blue-grey joint caps."""
    x, y = base
    yawr = math.radians(yaw)
    fwd = Vector((math.cos(yawr), math.sin(yawr), 0))
    p0 = Vector((x, y, z0))
    p1 = p0 + Vector((0, 0, 0.18))
    p2 = p1 + fwd * 0.25 + Vector((0, 0, 0.35))
    p3 = p2 + fwd * 0.35 - Vector((0, 0, 0.05))
    for a, b, nm in ((p0, p1, "l0"), (p1, p2, "l1"), (p2, p3, "l2")):
        rod(f"{name}_{nm}", a, b, 0.045, M["cobot"])
    for i, p in enumerate((p1, p2, p3)):
        cyl(f"{name}_j{i}", 0.06, 0.12, tuple(p), M["cobot_joint"], rot=(90, 0, yaw))


def booster_k1(M, name, x, y):
    """a Booster K1 (small humanoid, about 0.95 m tall) sitting on the floor with its legs
    stretched out towards the cage, as the five in the video"""
    box(f"{name}_hips", (0.16, 0.24, 0.12), (x, y, 0.0), M["dark_grey"])
    box(f"{name}_torso", (0.16, 0.26, 0.3), (x - 0.02, y, 0.12), M["dark_grey"])
    box(f"{name}_chest", (0.03, 0.18, 0.14), (x + 0.07, y, 0.26), M["metal"])
    cyl(f"{name}_neck", 0.03, 0.06, (x - 0.02, y, 0.45), M["black_plastic"])
    sphere(f"{name}_head", 0.075, Vector((x - 0.02, y, 0.53)), M["black_plastic"])
    for sy in (-1, 1):
        hip = Vector((x + 0.05, y + sy * 0.07, 0.06))
        knee = Vector((x + 0.27, y + sy * 0.08, 0.07))
        foot = Vector((x + 0.47, y + sy * 0.08, 0.06))
        rod(f"{name}_thigh{sy}", hip, knee, 0.05, M["black_plastic"])
        rod(f"{name}_shin{sy}", knee, foot, 0.04, M["black_plastic"])
        box(f"{name}_foot{sy}", (0.04, 0.07, 0.12), (foot.x + 0.02, foot.y, 0.0), M["black_plastic"])
        sh = Vector((x - 0.02, y + sy * 0.16, 0.39))
        el = Vector((x + 0.0, y + sy * 0.18, 0.24))
        rod(f"{name}_upperarm{sy}", sh, el, 0.035, M["dark_grey"])
        rod(f"{name}_forearm{sy}", el, Vector((x + 0.08, y + sy * 0.17, 0.12)), 0.03, M["black_plastic"])


def humanoid(M, name, x, y, yaw=0.0):
    """black humanoid robot hanging on its wheeled stand (photos 8 and 9)"""
    box(f"{name}_stand_foot", (0.7, 0.6, 0.08), (x, y, 0.04), M["black_plastic"], rot_z=yaw)
    cyl(f"{name}_stand_pole", 0.04, 1.6, (x, y - 0.2, 0.85), M["black_metal"])
    box(f"{name}_torso", (0.38, 0.25, 0.55), (x, y, 1.0), M["dark_grey"], rot_z=yaw)
    sphere(f"{name}_head", 0.12, Vector((x, y, 1.7)), M["dark_grey"])
    for sx in (-1, 1):
        rod(f"{name}_arm{sx}", Vector((x + sx * 0.24, y, 1.48)), Vector((x + sx * 0.28, y + 0.05, 0.95)), 0.05, M["dark_grey"])
        rod(f"{name}_leg{sx}", Vector((x + sx * 0.1, y, 1.0)), Vector((x + sx * 0.12, y, 0.25)), 0.06, M["dark_grey"])


def euro_boxes(M, name, x, y, n):
    for i in range(n):
        box(f"{name}_{i}", (0.6, 0.4, 0.31), (x, y, 0.33 * i), M["euro_grey"])


def build_surroundings(M):
    """Everything around the cage, placed from the 9 lab photos (Nov 2026). North is the side with
    the corridor to the wall; west the robot area (ABB, UR table); south the desks."""
    h = CAGE_HALF
    W = ROOM["ymax"]                      # north wall
    # ---- north corridor (photos 2, 3, 4): grey cabinet with magnets and two screens on top,
    #      the Fanuc on its aluminium frame, its controller, grey control boxes, the lab sign
    cx, cw = -1.05, 1.0
    box("cabinet_grey", (cw, 0.55, 2.0), (cx, W - 0.3, 0), M["cabinet"])
    box("cabinet_line", (0.01, 0.56, 1.95), (cx, W - 0.3, 0.02), M["black_plastic"])
    box("cabinet_handle", (0.06, 0.03, 0.12), (cx + 0.15, W - 0.585, 0.95), M["black_plastic"])
    for i, (dx, dz, col) in enumerate(((-0.35, 1.55, "magnet_red"), (-0.22, 1.55, "magnet_red"),
                                       (-0.28, 1.45, "magnet_blue"), (0.2, 1.6, "magnet_red"),
                                       (0.3, 1.58, "magnet_blue"), (0.32, 1.35, "magnet_red"))):
        cyl(f"magnet_{i}", 0.025, 0.01, (cx + dx, W - 0.58, dz), M[col], rot=(90, 0, 0), verts=12)
    for i, (dx, w_) in enumerate(((-0.25, 0.5), (0.2, 0.62))):     # screens in aluminium frames
        box(f"cab_screen_frame_{i}", (w_ + 0.05, 0.3, 0.5), (cx + dx, W - 0.32 + 0.05 * i, 2.0), M["alu"])
        box(f"cab_screen_{i}", (w_, 0.02, 0.44), (cx + dx, W - 0.48 + 0.05 * i, 2.03), M["black_plastic"])
    box("lab_sign", (0.9, 0.01, 0.16), (cx + 1.05, W - 0.01, 2.15), M["glass"])
    # Fanuc: yellow arm on a floor frame of aluminium profiles, reaching towards the cage
    for i, (dx, dy, lx, ly) in enumerate(((0, 0, 0.9, 0.06), (0, 0.35, 0.9, 0.06), (0, -0.35, 0.9, 0.06),
                                         (0.42, 0, 0.06, 0.8), (-0.42, 0, 0.06, 0.8))):
        box(f"fanuc_frame_{i}", (lx, ly, 0.08), (0.05 + dx, W - 0.55 + dy, 0), M["alu"])
    box("fanuc_plate", (0.35, 0.35, 0.1), (0.05, W - 0.55, 0.08), M["black_metal"])
    arm_robot(M, "yellow_robot", (0.05, W - 0.55), M["yellow"], scale=0.7, yaw=-120, pose=(15, 10, 50))
    box("fanuc_controller", (0.55, 0.55, 0.85), (0.6, W - 0.4, 0), M["beige_machine"])
    box("fanuc_controller_label", (0.4, 0.01, 0.3), (0.6, W - 0.68, 0.45), M["yellow"])
    box("fanuc_controller_top", (0.55, 0.55, 0.05), (0.6, W - 0.4, 0.85), M["black_plastic"])
    tube_curve("fanuc_cables", [(0.45, W - 0.4, 0.9), (0.6, W - 0.5, 0.97), (0.75, W - 0.35, 0.92)], 0.03, M["tape_red"])
    box("teach_pendant_n", (0.25, 0.08, 0.3), (0.7, W - 0.5, 0.9), M["pendant"], rot_z=10)
    box("control_white", (0.5, 0.5, 0.75), (1.2, W - 0.35, 0), M["white_box"])
    box("control_white_top", (0.5, 0.5, 0.25), (1.2, W - 0.35, 0.75), M["grey_box"])
    box("duct_down", (0.6, 0.3, 2.4), (1.1, W - 0.15, 2.2), M["duct"])
    # ---- west (photos 1, 2, 7): grey mats with five Booster K1 sitting along the cage, the big white ABB
    #      on its wooden pallet with the blue-wrapped crate beside it, the UR cobot table
    box("mats_out_w", (1.0, 2.4, 0.012), (-1.6, -0.1, 0), M["mat_grey"])
    for i, yy in enumerate(np.linspace(-0.8, 0.8, 5)):      # five Booster K1 sitting in a row by the net
        booster_k1(M, f"booster_k1_{i}", -1.6, float(yy))
    box("abb_pallet", (1.3, 1.2, 0.14), (-3.1, 0.4, 0), M["wood"])
    arm_robot(M, "white_robot", (-3.1, 0.4), M["white_robot"], scale=2.1, yaw=10, pose=(-15, 20, 70))
    box("abb_controller", (0.7, 0.6, 1.3), (-3.9, 1.6, 0), M["grey_box"])
    box("abb_controller_screen", (0.3, 0.02, 0.22), (-3.9, 1.29, 0.95), M["laptop_screen"])
    box("crate_base", (1.2, 2.0, 0.12), (-3.1, -1.3, 0), M["black_plastic"])
    box("crate_wrap", (1.2, 2.0, 0.4), (-3.1, -1.3, 0.12), M["blue_wrap"])
    box("mobile_base", (0.7, 0.5, 0.3), (-3.0, -2.9, 0), M["dark_grey"])
    box("cobot_table", (1.0, 0.7, 0.85), (-2.4, 1.35, 0), M["alu"])
    box("cobot_table_glass", (1.0, 0.7, 0.02), (-2.4, 1.35, 0.85), M["glass"])
    cobot(M, "cobot", (-2.65, 1.35), 0.87, yaw=-30)
    box("green_print", (0.12, 0.08, 0.06), (-2.2, 1.2, 0.87), M["green"])
    box("spindle_stand", (0.25, 0.25, 1.0), (-1.75, W - 0.35, 0), M["alu"])
    box("spindle", (0.18, 0.18, 0.4), (-1.75, W - 0.45, 1.0), M["black_plastic"])
    tube_curve("orange_cable", [(-1.75, W - 0.45, 1.4), (-1.6, W - 0.5, 1.2), (-1.55, W - 0.4, 0.4), (-1.6, W - 0.3, 0.02)], 0.012, M["orange"])
    euro_boxes(M, "euro_a", -2.95, W - 0.25, 3)
    euro_boxes(M, "euro_b", -2.3, W - 0.25, 3)
    box("airtech_box_1", (0.4, 0.4, 0.5), (-1.65, 1.35, 0), M["cardboard"])
    box("airtech_box_2", (0.4, 0.4, 0.45), (-1.65, 1.35, 0.5), M["cardboard"])
    # ---- east (video): the wall is about 0.5 m behind the net. A cardboard box on the floor, a
    #      wooden board leaning on the wall, the green eyewash station, a black box up on a shelf
    box("carton_e", (0.4, 0.6, 0.35), (1.25, 0.05, 0), M["cardboard"])
    box("board_e", (0.06, 0.28, 1.35), (1.42, 0.5, 0), M["wood"])
    cyl("eyewash_pipe", 0.02, 0.5, (1.46, 0.3, 1.45), M["pipe"])
    for i, dz in enumerate((1.25, 1.55)):
        sphere(f"eyewash_{i}", 0.05, Vector((1.43, 0.3, dz)), M["green"])
    box("shelf_e", (0.3, 0.7, 0.02), (1.35, 0.85, 1.75), M["white_frame"])
    box("dji_box", (0.25, 0.6, 0.25), (1.35, 0.85, 1.77), M["cardboard"])
    box("wall_box_e", (0.08, 0.2, 0.2), (1.46, -0.2, 0.9), M["black_plastic"])
    box("socket_e", (0.06, 0.15, 0.1), (1.47, -0.6, 0.35), M["tape_red"])
    # ---- south (video): one long workbench row along the whole net, blue drawer units, PCs and
    #      monitors on it; a second row further away; the humanoids and the crane beyond
    desk(M, "desk_s", -0.15, -h - 0.5, 2.9, 0.8)
    box("desk_s_drawers2", (0.55, 0.7, 0.45), (0.6, -h - 0.5, 0.4), M["blue"])
    for i, (x, kind) in enumerate(((-1.2, "pc"), (-0.75, "mon"), (-0.15, "mon"), (0.35, "pc"), (0.85, "mon"))):
        if kind == "pc":
            box(f"pc_s{i}", (0.22, 0.48, 0.48), (x, -h - 0.55, 0.9), M["black_plastic"])
        else:
            monitor(M, f"mon_s{i}", x, -h - 0.7, 0.9, rot=180)
            box(f"keyboard_s{i}", (0.45, 0.15, 0.02), (x, -h - 0.4, 0.9), M["black_plastic"])
    box("parts_s", (0.3, 0.2, 0.12), (0.1, -h - 0.35, 0.9), M["cardboard"])
    box("red_tool_s", (0.15, 0.1, 0.1), (-0.5, -h - 0.35, 0.9), M["tape_red"])
    desk(M, "desk_s2", -0.4, -3.0, 2.9, 0.8)
    for i, x in enumerate((-1.3, -0.4, 0.5)):
        monitor(M, f"mon_s2_{i}", x, -2.9, 0.9)
    box("chair_s", (0.5, 0.5, 0.5), (0.3, -2.2, 0), M["black_plastic"])
    box("chair_s_back", (0.45, 0.08, 0.55), (0.3, -2.42, 0.5), M["black_plastic"])
    humanoid(M, "humanoid_1", -0.9, -4.2)
    humanoid(M, "humanoid_2", -2.3, -4.6)
    box("crane_foot", (1.0, 0.8, 0.08), (-3.6, -5.6, 0), M["black_metal"])
    box("crane_post", (0.1, 0.1, 2.3), (-3.6, -5.6, 0.08), M["black_metal"])
    rod("crane_boom", Vector((-3.6, -5.6, 2.3)), Vector((-3.6, -4.7, 2.0)), 0.05, M["black_metal"])
    for i, x in enumerate((-5.2, -3.4)):
        desk(M, f"desk_w{i}", x, -4.0, 1.6, 0.75)
        monitor(M, f"mon_w{i}", x, -3.9, 0.9)
    box("cases", (0.8, 0.5, 0.9), (0.9, -4.6, 0), M["black_plastic"])


def materials():
    c = mat_color
    return dict(
        wall=c("wall", (0.82, 0.82, 0.8), 0.9), ceiling=c("ceiling", (0.55, 0.55, 0.56), 0.9),
        glass=c("glass", (0.12, 0.16, 0.22), 0.1), glass_dark=c("glass_dark", (0.08, 0.1, 0.16), 0.1),
        white_frame=c("white_frame", (0.85, 0.85, 0.85), 0.5), door_grey=c("door_grey", (0.6, 0.62, 0.62), 0.6),
        tape_red=c("tape_red", (0.7, 0.05, 0.05), 0.5), tape_white=c("tape_white", (0.9, 0.9, 0.9), 0.5),
        duct=c("duct", (0.75, 0.75, 0.74), 0.5, 0.3), pipe=c("pipe", (0.8, 0.8, 0.8), 0.4, 0.5),
        lamp=c("lamp", (1, 1, 0.97), 0.5, 0, emit=8.0), metal=c("metal", (0.6, 0.6, 0.62), 0.35, 0.8),
        black_metal=c("black_metal", (0.02, 0.02, 0.02), 0.5, 0.3),
        alu_profile=c("alu_profile", (0.72, 0.73, 0.75), 0.35, 0.8), alu=c("alu", (0.65, 0.66, 0.68), 0.4, 0.7),
        alu_top=c("alu_top", (0.75, 0.75, 0.76), 0.4, 0.6), beige_machine=c("beige_machine", (0.75, 0.68, 0.5), 0.6),
        grey_box=c("grey_box", (0.35, 0.36, 0.38), 0.6), white_box=c("white_box", (0.85, 0.85, 0.83), 0.5),
        yellow=c("yellow", (0.85, 0.65, 0.02), 0.4), cabinet=c("cabinet", (0.68, 0.69, 0.68), 0.5),
        cardboard=c("cardboard", (0.55, 0.4, 0.25), 0.9), black_plastic=c("black_plastic", (0.03, 0.03, 0.03), 0.6),
        wood=c("wood", (0.72, 0.52, 0.3), 0.6), blue=c("blue", (0.08, 0.18, 0.6), 0.4),
        laptop=c("laptop", (0.6, 0.6, 0.62), 0.4, 0.5), laptop_screen=c("laptop_screen", (0.7, 0.8, 0.95), 0.3, 0, emit=1.5),
        dark_grey=c("dark_grey", (0.15, 0.15, 0.15), 0.5), pendant=c("pendant", (0.3, 0.32, 0.35), 0.5),
        green=c("green", (0.3, 0.85, 0.05), 0.4), controller=c("controller", (0.08, 0.08, 0.08), 0.5),
        blue_wrap=c("blue_wrap", (0.08, 0.2, 0.5), 0.3), white_robot=c("white_robot", (0.88, 0.88, 0.86), 0.4),
        cobot=c("cobot", (0.8, 0.82, 0.85), 0.4), cobot_joint=c("cobot_joint", (0.35, 0.45, 0.55), 0.4),
        orange=c("orange", (0.9, 0.35, 0.05), 0.5),
        magnet_red=c("magnet_red", (0.75, 0.05, 0.05), 0.4), magnet_blue=c("magnet_blue", (0.05, 0.15, 0.6), 0.4),
        euro_grey=c("euro_grey", (0.5, 0.52, 0.54), 0.6), mat_grey=c("mat_grey", (0.12, 0.12, 0.13), 0.8),
    )


def setup_render(samples=24, res=(640, 400)):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = 24
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = -0.6
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.6, 0.62, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    # soft fill from above, like the many ceiling lamps
    bpy.ops.object.light_add(type="AREA", location=(0, -1, ROOM["h"] - 0.4))
    L = bpy.context.object
    L.data.energy = 900
    L.data.size = 10
    bpy.ops.object.light_add(type="AREA", location=(-4, -1, ROOM["h"] - 0.4))
    L = bpy.context.object
    L.data.energy = 500
    L.data.size = 6


def camera(name, loc, look_at=None, hfov=90.0, rot=None):
    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(hfov)
    cam_data.clip_start = 0.02
    o = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(o)
    o.location = loc
    if look_at is not None:
        d = Vector(look_at) - Vector(loc)
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = d.to_track_quat("-Z", "Y")
    elif rot is not None:
        o.rotation_mode = "XYZ"
        o.rotation_euler = rot
    return o


def drone_camera(x, y, z, heading_deg, gimbal_deg=-50.0, hfov=62.0):
    """Same convention as sim/render.py: heading 0 = north (+y), 90 = east; gimbal negative = down."""
    # Blender camera looks along -Z; rotate X by (90 + gimbal) to tilt, then about Z.
    rx = math.radians(90 + gimbal_deg)
    rz = math.radians(-heading_deg)
    return camera("drone_cam", (x, y, z), rot=(rx, 0, rz), hfov=hfov)


def render_to(cam, path, res=None, samples=None):
    sc = bpy.context.scene
    sc.camera = cam
    if res:
        sc.render.resolution_x, sc.render.resolution_y = res
    if samples:
        sc.cycles.samples = samples
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


WHITE_BALANCE = np.array([1.1, 0.97, 0.86], np.float32)   # the Mini 3 frames are warm


def drone_frame(x, y, z, heading_deg, res=(320, 180), samples=16, target=0.42, tmp="/tmp/_drone.exr"):
    """The drone camera's picture as an RGB uint8 array, with auto-exposure like the Mini 3:
    the scene is rendered in linear light, then brightened until the mean reaches `target`
    (the floor is black, so the gain is large), then a filmic curve and sRGB."""
    sc = bpy.context.scene
    cam = drone_camera(x, y, z, heading_deg)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.cycles.samples = samples
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    sc.render.image_settings.file_format = "PNG"
    img = bpy.data.images.load(tmp)
    a = np.array(img.pixels[:], np.float32).reshape(res[1], res[0], 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    bpy.data.objects.remove(cam)
    lum = float((a @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean())
    a = a * min(8.0, target / max(lum, 1e-4)) * WHITE_BALANCE
    a = a / (1 + 0.35 * a)                                   # soft highlights
    a = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(np.clip(a, 0, None), 1 / 2.4) - 0.055)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- light probes for the fast renderer
#
# sim/render.py draws the inside of the cage itself (mats, pad, net, posts) in a few ms. For
# everything seen through the net it looks up "probes": the lab rendered here by Cycles as six
# 90-degree pictures (a cube) plus a depth map, from a grid of points inside the cage. The depth
# lets render.py correct the parallax between the drone and the nearest probes.

FACES = {   # name: (forward, up); right = forward x up
    "px": ((1, 0, 0), (0, 0, 1)), "nx": ((-1, 0, 0), (0, 0, 1)),
    "py": ((0, 1, 0), (0, 0, 1)), "ny": ((0, -1, 0), (0, 0, 1)),
    "pz": ((0, 0, 1), (0, 1, 0)), "nz": ((0, 0, -1), (0, 1, 0)),
}
PROBE_GRID = dict(x=(-0.7, 0.0, 0.7), y=(-0.7, 0.0, 0.7), z=(0.4, 1.2, 2.0, 2.8))
if os.environ.get("PROBE_TEST"):
    PROBE_GRID = dict(x=(0.0,), y=(0.0,), z=(1.2,))
HIDE_FOR_PROBES = ("net", "post_", "top_", "pad")


def _face_camera(pos, face, res):
    from mathutils import Matrix
    f, u = (Vector(v) for v in FACES[face])
    r = f.cross(u)
    cam = camera(f"probe_{face}", pos, hfov=90.0)
    cam.data.sensor_fit = "HORIZONTAL"
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = Matrix((r, u, -f)).transposed().to_quaternion()
    return cam


def _face_dirs(face, n):
    f, u = (np.array(v, float) for v in FACES[face])
    r = np.cross(f, u)
    c = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(c, -c)                     # row 0 = top of the picture
    d = f + X[..., None] * r + Y[..., None] * u
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _hide_cage():
    for o in bpy.data.objects:
        if o.name.startswith(HIDE_FOR_PROBES):
            o.hide_render = True
            o.hide_viewport = True        # also for scene.ray_cast (the depth maps)
    bpy.context.view_layer.update()


def probe_depths(pos, depth_res=64):
    sc = bpy.context.scene
    deps = bpy.context.evaluated_depsgraph_get()
    depth = np.zeros((len(pos), 6, depth_res, depth_res), np.float16)
    for i, p in enumerate(pos):
        P = Vector(tuple(float(c) for c in p))
        for k, face in enumerate(FACES):
            dirs = _face_dirs(face, depth_res).reshape(-1, 3)
            dd = np.empty(len(dirs), np.float32)
            for j, v in enumerate(dirs):
                hit, loc, *_ = sc.ray_cast(deps, P, Vector(tuple(v)))
                dd[j] = (loc - P).length if hit else 30.0
            depth[i, k] = dd.reshape(depth_res, depth_res)
    return depth


def redo_depths(path):
    """Recompute only the depth maps of an existing probe file (fast, no rendering)."""
    _hide_cage()
    z = dict(np.load(path))
    z["depth"] = probe_depths(z["pos"], z["depth"].shape[-1])
    np.savez_compressed(path, **z)
    print("depths ->", path, float(z["depth"].max()))


def bake_probes(out, res=320, depth_res=64, samples=32):
    sc = bpy.context.scene
    _hide_cage()
    deps = bpy.context.evaluated_depsgraph_get()
    g = PROBE_GRID
    pos = np.array([(x, y, z) for x in g["x"] for y in g["y"] for z in g["z"]], np.float32)
    rgb = np.zeros((len(pos), 6, res, res, 3), np.float16)
    depth = np.zeros((len(pos), 6, depth_res, depth_res), np.float16)
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.cycles.samples = samples
    sc.render.image_settings.file_format = "OPEN_EXR"
    tmp = "/tmp/_probe.exr"
    for i, p in enumerate(pos):
        for k, face in enumerate(FACES):
            cam = _face_camera(tuple(p), face, res)
            sc.camera = cam
            sc.render.filepath = tmp
            bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(tmp)
            rgb[i, k] = np.array(im.pixels[:], np.float32).reshape(res, res, 4)[::-1, :, :3]
            bpy.data.images.remove(im)
            bpy.data.objects.remove(cam)
        print(f"probe {i + 1}/{len(pos)}", flush=True)
    depth[:] = probe_depths(pos, depth_res)
    # brightness of the black mats under the same light, so render.py can match its own floor
    cam = _face_camera((0.0, 0.0, 1.2), "nz", res)
    sc.camera = cam
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(tmp)
    mats = np.array(im.pixels[:], np.float32).reshape(res, res, 4)[res // 3: 2 * res // 3, res // 3: 2 * res // 3, :3]
    bpy.data.images.remove(im)
    np.savez_compressed(out, pos=pos, rgb=rgb, depth=depth, faces=np.array(list(FACES)),
                        mat_radiance=np.float32(mats.mean()))
    print("probes ->", out)


# ---------------------------------------------------------------- export for Gazebo (ROS 2 simulator)

EXPORT_SKIP = ("net", "pad")      # Gazebo gets its own 3D net and a movable pad


def _tile_uvs(obj, size):
    """UVs in metres / size, so an image texture repeats every `size` m (OBJ has no mapping nodes)."""
    me = obj.data
    if not me.uv_layers:
        me.uv_layers.new()
    uv = me.uv_layers.active.data
    mw = obj.matrix_world
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = mw @ me.vertices[me.loops[li].vertex_index].co
            uv[li].uv = (co.x / size, co.y / size)


def export_obj(out_dir):
    """The lab and cage frame (no net, no pad) as one OBJ + MTL + textures, z up, metres."""
    os.makedirs(out_dir, exist_ok=True)
    for o in list(bpy.data.objects):
        if o.name.startswith(EXPORT_SKIP) or o.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(o)
    for o in list(bpy.data.objects):
        if o.type == "CURVE":                     # hoses -> meshes
            bpy.ops.object.select_all(action="DESELECT")
            o.select_set(True)
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.convert(target="MESH")
    _tile_uvs(bpy.data.objects["cage_mats"], 2.4)
    _tile_uvs(bpy.data.objects["lab_floor"], 4.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.wm.obj_export(filepath=os.path.join(out_dir, "lab.obj"), export_selected_objects=True,
                          forward_axis="Y", up_axis="Z", export_materials=True, path_mode="COPY",
                          export_uv=True, export_normals=True, apply_modifiers=True)
    # Gazebo does not read texture options like "-s 0.25 0.25 1" (the tiling is in the UVs already)
    mtl = os.path.join(out_dir, "lab.mtl")
    import re
    text = re.sub(r"^(map_\w+)\s+(?:-\w+(?:\s+[-\d.]+)+\s+)+", r"\1 ", open(mtl).read(), flags=re.M)
    # newer Blender leaves out "Kd" for textured materials; Gazebo then draws them black
    blocks = text.split("newmtl ")
    for i, b in enumerate(blocks):
        if "map_Kd" in b and "\nKd " not in b:
            name, rest = b.split("\n", 1)
            blocks[i] = f"{name}\nKd 1.000000 1.000000 1.000000\n{rest}"
    open(mtl, "w").write("newmtl ".join(blocks))
    print("obj ->", out_dir)


def build(pad_xy=(0.0, 0.0), pad_yaw=0.0):
    clear()
    M = materials()
    build_room(M)
    build_cage(M, pad_xy, pad_yaw)
    build_surroundings(M)
    setup_render()


OUTSIDE_VIEWS = [
    ("se_high", (3.5, -4.0, 3.6), (0, 0, 1.0)),
    ("nw_high", (-3.8, 1.9, 3.8), (0, 0, 0.8)),
    ("desk_view", (0.0, -2.6, 1.6), (0, 0.6, 0.8)),
    ("west_view", (-2.8, -2.5, 1.7), (0, 0.2, 1.0)),
    ("east_view", (3.2, -1.0, 1.7), (0, 0.2, 1.0)),
    ("top", (0.5, 0.4, 3.9), (0, 0, 0)),
]

DRONE_VIEWS = [(0, -0.75, 0.9, 0), (0.3, -0.2, 1.4, 30), (-0.6, 0.3, 1.2, 90),
               (0.0, 0.5, 1.6, 180), (0.5, 0.5, 2.0, 225), (-0.7, -0.7, 1.0, 45)]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="write the .blend file here")
    ap.add_argument("--previews", help="render outside and drone views into this folder")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--only", nargs="*", help="names of views to render")
    ap.add_argument("--probes", help="bake the light probes for sim/render.py into this .npz")
    ap.add_argument("--probe-res", type=int, default=320)
    ap.add_argument("--redo-depths", help="recompute only the depth maps of this probe file")
    ap.add_argument("--export-obj", help="write the lab as OBJ for Gazebo into this folder")
    a = ap.parse_args(argv)
    build()
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
    if a.export_obj:
        export_obj(os.path.abspath(a.export_obj))
        return
    if a.redo_depths:
        redo_depths(os.path.abspath(a.redo_depths))
    if a.probes:
        bake_probes(os.path.abspath(a.probes), res=a.probe_res, samples=a.samples)
    if a.previews:
        os.makedirs(a.previews, exist_ok=True)
        for name, loc, tgt in OUTSIDE_VIEWS:
            if a.only and name not in a.only:
                continue
            render_to(camera(name, loc, tgt, hfov=80), os.path.join(os.path.abspath(a.previews), f"out_{name}.png"),
                      res=(640, 400), samples=a.samples)
            print(name, flush=True)
        for i, (x, y, z, hd) in enumerate(DRONE_VIEWS):
            if a.only and f"drone{i}" not in a.only:
                continue
            from PIL import Image
            Image.fromarray(drone_frame(x, y, z, hd, samples=a.samples)).save(
                os.path.join(os.path.abspath(a.previews), f"drone_{i}.png"))
            print("drone", i, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
