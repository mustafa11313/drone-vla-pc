"""
run.py  -  run many simulated missions in the cage and compare brains.

    # no API, no GPU: rule-based brain + a fake "VLA" that steers toward red pixels
    python3 -m sim.run --pilot scripted --policy seek --missions 20

    # Claude as the brain (ANTHROPIC_API_KEY set in your terminal), real VLA on the cluster
    python3 -m sim.run --pilot claude --policy http://<gpu-host>:8000 --missions 20 --video

    # the VLA alone, no brain (the baseline arm)
    python3 -m sim.run --pilot none --policy http://<gpu-host>:8000 --missions 20

--policy:  inference server URL (OpenVLA, CNN, ...), "servo" (classic controller as the VLA),
           "seek" (fake model steering to red pixels), "mock" (fixed forward-and-down command).
Same --seed gives the same pads and start positions, so arms are compared on the same course.

Writes <out>/mission_<seed>/{steps.csv, decisions.jsonl, result.json, flight.gif with --video}
and <out>/summary.csv, and prints the success rate.

A VLA trained on real camera images will probably fly worse here than on the drone: the
picture is drawn from simple shapes. This first version tests the brain, the skills and the
safety rules; the realistic copy of the cage (from the lab video) comes next.
"""
import argparse, csv, os, time
from datetime import datetime

import numpy as np

from brain.pilot import Decision
from sim.env import CageEnv
from sim.mission import MISSION, run_mission


class SeekPolicy:
    """Stand-in for a VLA: flies toward red pixels (like inference_server.py --mock-seek)."""
    last_confidence = None

    def act(self, jpeg, instruction=None):
        from brain.phases import pad_in_frame
        visible, x, y, _ = pad_in_frame(jpeg)
        if not visible:
            return [15.0, 0.0, 0.0, 0.0], 0.0, 0.0
        pitch = max(0.0, 30.0 * (0.9 - y))
        throttle = -40.0 if abs(x) < 0.25 and y > 0.6 else 0.0
        return [round(pitch, 1), round(30.0 * x, 1), throttle, 0.0], 0.0, 0.0


class MockPolicy:
    last_confidence = None

    def act(self, jpeg, instruction=None):
        return [10.0, 0.0, -20.0, 0.0], 0.0, 0.0


class NoPilot:
    """VLA alone: the model flies the whole mission, no brain."""
    last_ms = 0.0

    def decide(self, obs, context):
        if context.get("current"):
            return Decision(observation="", pad_visible=False, skill="continue", reason="VLA only")
        return Decision(observation="", pad_visible=False, skill="vla", seconds=20, reason="VLA only")


def make_policy(name):
    if name == "seek":
        return SeekPolicy()
    if name == "mock":
        return MockPolicy()
    if name == "servo":
        from brain.servo import ServoPolicy
        return ServoPolicy()
    from brain.policy_client import PolicyClient
    return PolicyClient(name, timeout_s=10)


def make_pilot(args):
    if args.pilot == "claude":
        from brain.pilot import ClaudePilot
        return ClaudePilot(model=args.model, effort=args.effort)
    if args.pilot == "scripted":
        from brain.pilot import ScriptedPilot
        return ScriptedPilot(approach=args.approach)
    return NoPilot()


def main(argv=None):
    ap = argparse.ArgumentParser(description="simulated cage missions: brain + VLA")
    ap.add_argument("--pilot", choices=["claude", "scripted", "none"], default="scripted")
    ap.add_argument("--policy", default="seek", help="inference server URL, servo, seek or mock")
    ap.add_argument("--missions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="first seed; mission i uses seed + i")
    ap.add_argument("--mission", default=MISSION)
    ap.add_argument("--decide-every", type=float, default=2.0, help="ask the brain at least this often (s)")
    ap.add_argument("--brain-latency", type=float, help="fixed brain answer time in simulated s "
                    "(default: the real measured time; scripted pilot: 1.5)")
    ap.add_argument("--approach", choices=["vla", "servo", "move"], default="vla", help="scripted pilot only")
    ap.add_argument("--servo-target-y", type=float, help="servo skill: where the pad should sit in the "
                    "picture (-1 top .. 1 bottom) before descending; default brain/servo.py TARGET_Y")
    ap.add_argument("--model", default="claude-opus-5-5", help="Claude model for --pilot claude")
    ap.add_argument("--effort", default="low", help="Claude effort: low is fastest")
    ap.add_argument("--max-time", type=float, default=90.0)
    ap.add_argument("--video", action="store_true", help="save flight.gif per mission")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.brain_latency is None and args.pilot != "claude":
        args.brain_latency = 1.5 if args.pilot == "scripted" else 0.0
    out = args.out or os.path.join("logs", f"sim_{datetime.now():%Y%m%d_%H%M%S}_{args.pilot}")
    os.makedirs(out, exist_ok=True)
    policy = make_policy(args.policy)
    if args.servo_target_y is not None:
        from brain.skills import ServoSkill
        ServoSkill.target_y = args.servo_target_y
    rows = []
    for i in range(args.missions):
        seed = args.seed + i
        env = CageEnv(seed=seed, max_time=args.max_time)
        pilot = make_pilot(args)
        print(f"mission {i + 1}/{args.missions} (seed {seed}), pad at {env.scene.pad_x:+.2f}, {env.scene.pad_y:+.2f}")
        t0 = time.time()
        res = run_mission(env, pilot, policy, mission=args.mission, decide_every=args.decide_every,
                          brain_latency=args.brain_latency, out_dir=os.path.join(out, f"mission_{seed:03d}"),
                          save_frames=args.video)
        res["wall_s"] = round(time.time() - t0, 1)
        print(f"  -> {res['outcome']}, {res['time_s']} s, {res['pad_distance_m']} m from the pad centre, "
              f"closest to the net {res['min_net_m']} m\n")
        rows.append(res)
    keys = ["seed", "outcome", "time_s", "pad_distance_m", "min_net_m", "decisions",
            "brain_latency_mean_s", "skills_used", "wall_s"]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "skills_used": " ".join(r["skills_used"])})
    n = len(rows)
    ok = sum(r["outcome"] == "landed_on_pad" for r in rows)
    print(f"pilot={args.pilot} policy={args.policy}: {ok}/{n} landed on the pad ({100 * ok / max(n, 1):.0f} %)")
    for o in ("landed_off_pad", "hit_net", "timeout"):
        c = sum(r["outcome"] == o for r in rows)
        if c:
            print(f"  {o}: {c}")
    if ok:
        print(f"  mean time to land on the pad: {np.mean([r['time_s'] for r in rows if r['outcome'] == 'landed_on_pad']):.1f} s")
    print(f"logs: {out}")
    return rows


if __name__ == "__main__":
    main()
