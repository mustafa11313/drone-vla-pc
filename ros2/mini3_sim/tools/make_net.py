"""
make_net.py  -  the cage net as a real 3D mesh for Gazebo (models/cage_lab/meshes/net.obj).

Thin black rope strips on the 4 sides and the top of the 2 x 2 x 3 m cage, a square mesh of
about 5.5 cm, sagging a little inwards and slightly irregular like the real net, plus a strip of
extra net lying on the floor along each side. Both faces of every strip are written, so it is
visible from inside and outside.

    python3 tools/make_net.py
"""
import math, os

import numpy as np

HALF, HEIGHT = 1.0, 3.0
CELL = 0.055          # m, mesh size
STRAND = 0.004        # m, rope width
SEGMENTS = 24         # pieces per rope (for the sag)
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "cage_lab", "meshes", "net.obj")


def side_point(face, a, b, rng_phase):
    """a along the face (-HALF..HALF), b up (0..HEIGHT) for walls, or b across for the top."""
    if face == "top":
        sag = 0.06 * (1 - (a / HALF) ** 2) * (1 - (b / HALF) ** 2)
        return np.array([a, b, HEIGHT - sag])
    inward = 0.06 * (1 - (a / HALF) ** 2) * math.sin(math.pi * b / HEIGHT) \
        + 0.01 * math.sin(7 * a + rng_phase[0]) * math.sin(5 * b / HEIGHT + rng_phase[1])
    if face == "n":
        return np.array([a, HALF - inward, b])
    if face == "s":
        return np.array([a, -HALF + inward, b])
    if face == "e":
        return np.array([HALF - inward, a, b])
    return np.array([-HALF + inward, a, b])


def build():
    rng = np.random.default_rng(3)
    verts, quads, normals = [], [], []

    def strip(points, normal_hint):
        """A flat ribbon of width STRAND along the polyline, both sides."""
        for p, q in zip(points[:-1], points[1:]):
            d = q - p
            side = np.cross(d, normal_hint)
            n = np.linalg.norm(side)
            if n < 1e-9:
                continue
            side = side / n * STRAND / 2
            nrm = np.cross(side, d)
            nrm /= np.linalg.norm(nrm)
            i = len(verts)
            verts.extend([p - side, p + side, q + side, q - side])
            normals.extend([nrm, -nrm])
            quads.append((i, i + 1, i + 2, i + 3, len(normals) - 2))
            quads.append((i + 3, i + 2, i + 1, i, len(normals) - 1))

    for face in ("n", "s", "e", "w", "top"):
        ph = rng.uniform(0, 6, 2)
        if face == "top":
            normal = np.array([0, 0, 1.0])
            b_max, b_min = HALF, -HALF
        else:
            normal = {"n": np.array([0, 1.0, 0]), "s": np.array([0, 1.0, 0]),
                      "e": np.array([1.0, 0, 0]), "w": np.array([1.0, 0, 0])}[face]
            b_max, b_min = HEIGHT, 0.0
        # ropes running along a (horizontal) and along b (vertical)
        for b0 in np.arange(b_min + CELL / 2, b_max, CELL):
            jitter = rng.normal(0, 0.004)
            pts = [side_point(face, a, b0 + jitter + 0.006 * math.sin(3 * a + ph[0]), ph)
                   for a in np.linspace(-HALF, HALF, SEGMENTS)]
            strip(np.array(pts), normal)
        for a0 in np.arange(-HALF + CELL / 2, HALF, CELL):
            jitter = rng.normal(0, 0.004)
            pts = [side_point(face, a0 + jitter + 0.006 * math.sin(2 * b + ph[1]), b, ph)
                   for b in np.linspace(b_min, b_max, SEGMENTS)]
            strip(np.array(pts), normal)
    # extra net lying on the floor along each side (25 cm wide)
    for face in ("n", "s", "e", "w"):
        inward = {"n": (0, -1, 0), "s": (0, 1, 0), "e": (-1, 0, 0), "w": (1, 0, 0)}[face]
        for k in np.arange(0.0, 0.25, CELL):
            pts = [side_point(face, a, 0.0, (0, 0)) + np.array(inward) * k + np.array([0, 0, 0.006])
                   for a in np.linspace(-HALF + 0.02, HALF - 0.02, SEGMENTS)]
            strip(np.array(pts), np.array([0, 0, 1.0]))
    return np.array(verts), quads, np.array(normals)


def main():
    v, q, nr = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT.replace(".obj", ".mtl"), "w") as f:
        f.write("newmtl net_rope\nKa 0 0 0\nKd 0.03 0.03 0.03\nKs 0.02 0.02 0.02\nNs 10\nd 1\nillum 1\n")
    with open(OUT, "w") as f:
        f.write("mtllib net.mtl\no net\n")
        for x, y, z in v:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for x, y, z in nr:
            f.write(f"vn {x:.3f} {y:.3f} {z:.3f}\n")
        f.write("usemtl net_rope\n")
        for a, b, c, d, n in q:
            f.write(f"f {a + 1}//{n + 1} {b + 1}//{n + 1} {c + 1}//{n + 1} {d + 1}//{n + 1}\n")
    print(OUT, len(v), "vertices", len(q), "faces")


if __name__ == "__main__":
    main()
