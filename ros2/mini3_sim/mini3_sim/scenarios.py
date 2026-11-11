"""
scenarios  -  the yaw and roll scenario set (setups:=scenarios) and the camera geometry it uses.

Why: the 140 real episodes teach throttle and pitch well, but yaw only partly and roll hardly
(report of Nov 2026). In the real flights the roll pushes are small, short and often reversed, and
big turns go towards a pad the camera cannot see. These scenarios add many clear yaw and roll
examples, each with a reason the camera can see, and every layout twice (pad left and pad right).

The rule the autopilot follows (mini3_sim/autopilot.py, ScenarioPilot), from where the red pad is
in the camera picture (camera 50 deg down, 62 x 37 deg):

    pad not in the picture           ->  turn clockwise (yaw right) until it shows up
    far (upper part), well to a side ->  yaw towards it until it is about centred
    near (lower part), well to a side ->  roll towards it, heading kept
    about centred                     ->  pitch forward (with roll for a small sideways offset)
    over the pad                      ->  throttle down: DJI auto landing

Yaw and roll are never used at the same time (as in the real flights).

The eight scenarios (number of episodes in a full set of 660, half pad-left, half pad-right):

    A  yaw_small     far pad, 12-22 deg to the side                        120
    B  yaw_edge      far pad at the edge of the picture (22-33 deg)         80
    C  search        pad not in the picture (70-180 deg to the side)        60
    D  roll_side     near pad well to the side: roll                       120
    E  diagonal      pad a little to the side: pitch and roll together     100
    F  yaw_then_roll far pad to the side; the yaw stops early, roll does the rest  80
    G  close         pad close in front: small corrections, then landing    60
    H  straight      pad straight ahead: pitch only                         40

Setup i is always the same (seeded), so a failed episode is flown again with the same layout.
Setups 2k and 2k+1 are mirror images (same distances and height, pad on the other side).
The pair order is shuffled once, so the first 300 already have every scenario in proportion.
"""
import math
import random

CAM_FWD, CAM_DOWN = 0.1, 0.015              # camera position on the drone (m)
CAM_PITCH, HFOV, VFOV = 50.0, 62.0, 37.3    # deg, as in models/mini3/model.sdf (640 x 360)
FLOOR = 0.03                                # drone centre height when standing on the floor

YAW_X = 12.0        # deg: the pad further to the side than this is "well to the side"
NEAR_Y = 4.0        # deg below the picture centre: from here down the pad counts as near
DRONE_LIM, PAD_LIM = 0.65, 0.68             # m from the cage centre (2 x 2 m cage)

SCENARIOS = {        # name: (episodes in the full set, description)
    "yaw_small": (120, "far pad 12-22 deg to the side: yaw to face it"),
    "yaw_edge": (80, "far pad at the edge of the picture: strong yaw"),
    "search": (60, "pad not in the picture: clockwise search turn"),
    "roll_side": (120, "near pad well to the side: roll, heading kept"),
    "diagonal": (100, "pad a little to the side: pitch and roll together"),
    "yaw_then_roll": (80, "far pad to the side: short yaw, then roll"),
    "close": (60, "pad close in front: small corrections and landing"),
    "straight": (40, "pad straight ahead: pitch only"),
}
FULL_SET = sum(n for n, _ in SCENARIOS.values())     # 660


def pad_angles(fwd, right, z):
    """Where the pad centre is in the camera picture: (x, y) in degrees, x > 0 right of the
    centre, y > 0 below it; None if it is behind the camera. fwd/right: pad offset from the drone
    (m), z: drone centre height above the floor (m)."""
    f, d = fwd - CAM_FWD, z - CAM_DOWN
    p = math.radians(CAM_PITCH)
    depth = f * math.cos(p) + d * math.sin(p)
    if depth <= 0.05:
        return None
    x = math.degrees(math.atan(right / depth))
    y = math.degrees(math.atan((d * math.cos(p) - f * math.sin(p)) / depth))
    return x, y


def in_picture(a, margin=0.0):
    """a from pad_angles: inside the picture, `margin` deg in from the edges (negative: a centre
    just outside, with most of the 50 cm pad still showing)."""
    return a is not None and abs(a[0]) < HFOV / 2 - margin and abs(a[1]) < VFOV / 2 - margin


def _pair_order(seed=2026):
    names = [n for n, (count, _) in SCENARIOS.items() for _ in range(count // 2)]
    random.Random(seed).shuffle(names)
    return names


PAIRS = _pair_order()


def _sample_offset(name, rnd):
    """(fwd, right >= 0, hover height) in the drone's frame that fits the scenario."""
    for _ in range(20000):
        h = rnd.uniform(0.8, 1.4)                 # hover height above the floor (m)
        z = h + FLOOR
        if name == "search":
            b, dist = rnd.uniform(70, 180), rnd.uniform(0.6, 1.5)
        elif name == "close":
            b, dist = rnd.uniform(0, 25), rnd.uniform(0.25, 0.65)
        elif name == "straight":
            b, dist = rnd.uniform(0, 4), rnd.uniform(0.6, 1.7)
        elif name == "diagonal":
            b, dist = rnd.uniform(4, 16), rnd.uniform(0.8, 1.7)
        elif name == "roll_side":
            b, dist = rnd.uniform(15, 60), rnd.uniform(0.35, 1.0)
        else:                                     # the far yaw scenarios
            b, dist = rnd.uniform(8, 65), rnd.uniform(0.9, 1.8)
        fwd, right = dist * math.cos(math.radians(b)), dist * math.sin(math.radians(b))
        a = pad_angles(fwd, right, z)
        vis = in_picture(a, -2.0)
        x, y = (abs(a[0]), a[1]) if a else (0.0, 0.0)
        # far pads not near the top edge: a small height change must not push them out of view
        # (the height can end up to ~10 cm lower than asked: check there too)
        low = pad_angles(fwd, right, z - 0.1)
        vis = vis and y > -VFOV / 2 + 5 and in_picture(low, -2.0) and low[1] > -VFOV / 2 + 4
        ok = {
            "yaw_small": vis and YAW_X <= x < 22 and y < NEAR_Y,
            "yaw_edge": vis and x >= 22 and y < NEAR_Y,
            # not in the picture now, but well inside it once the turn faces it
            "search": not in_picture(a, -6.0) and in_picture(pad_angles(dist, 0.0, z), 3.0),
            "roll_side": vis and x >= YAW_X and y >= NEAR_Y,
            "diagonal": vis and 4 <= x < YAW_X and right >= 0.1,
            "yaw_then_roll": vis and 14 <= x < 28 and y < NEAR_Y,
            "close": vis and x < 20 and y >= 6,
            "straight": vis and x < 4,
        }[name]
        if ok:
            return fwd, right, h
    raise RuntimeError(f"no layout found for {name}")


def make_scenario(i, seed=9000):
    """Setup i of the scenario set (i from 0). Same i -> same setup."""
    pair, mirror = divmod(i, 2)
    name = PAIRS[pair % len(PAIRS)]
    rnd = random.Random(seed + pair)                         # shared by the two mirror images
    fwd, right, h = _sample_offset(name, rnd)
    yaw_stop = rnd.uniform(7, 10) if name == "yaw_then_roll" else rnd.uniform(2, 4)
    if mirror:
        right = -right
    rnd2 = random.Random(seed * 7 + i)                       # own placement and camera look
    for _ in range(20000):
        dx, dy = rnd2.uniform(-DRONE_LIM, DRONE_LIM), rnd2.uniform(-DRONE_LIM, DRONE_LIM)
        hd = rnd2.uniform(0, 360)
        r = math.radians(hd)
        px = dx + fwd * math.sin(r) + right * math.cos(r)
        py = dy + fwd * math.cos(r) - right * math.sin(r)
        if abs(px) <= PAD_LIM and abs(py) <= PAD_LIM:
            break
    else:
        raise RuntimeError(f"setup {i}: the layout does not fit in the cage")
    look = {   # camera variation per episode (camera_exposure.py)
        "target": round(rnd2.uniform(145, 185), 1),
        "wb": [round(1.10 * rnd2.uniform(0.95, 1.05), 3), 1.0, round(0.93 * rnd2.uniform(0.95, 1.05), 3)],
        "soften": round(rnd2.uniform(0.35, 0.8), 2),
        "noise": round(rnd2.uniform(0.0, 3.0), 2),
    }
    return {"number": i + 1, "repeat": 1, "scenario": name, "mirror": bool(mirror),
            "pad_side": "left" if right < 0 else "right",
            "drone_xy": (dx, dy), "pad_xy": (px, py), "heading": hd, "height": round(h, 3),
            "yaw_stop": round(yaw_stop, 2), "pad_yaw": round(rnd2.uniform(0, 360), 1), "look": look}


if __name__ == "__main__":       # print the first setups and the scenario counts
    import collections
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else FULL_SET
    c = collections.Counter(make_scenario(i)["scenario"] for i in range(n))
    print(f"first {n} setups:", dict(c))
    for i in range(6):
        s = make_scenario(i)
        print(i + 1, s["scenario"], s["pad_side"], [round(v, 2) for v in s["drone_xy"]],
              [round(v, 2) for v in s["pad_xy"]], round(s["heading"]), s["height"])
