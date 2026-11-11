"""
mission.py  -  one closed-loop mission in the simulated cage: pilot (brain) + skills (hands) +
governor + drone model + camera.

Timing is simulated honestly:
  - the drone and the skills run at 10 Hz of simulated time;
  - when the pilot is asked for a decision at time t, its real answer time L (the API call,
    measured on the wall clock, or --brain-latency for the scripted pilot) passes in the
    simulation as well: the running skill keeps flying until t + L, and only then does the
    new decision take over. A slow brain therefore really costs flight time, like on the drone;
  - if the running skill finishes while the pilot is still thinking, the drone hovers;
  - with no decision for BRAIN_TIMEOUT_S, the drone hovers, after BRAIN_GIVE_UP_S it lands
    (the same fallback the real loop must have when the network drops).

The pilot is asked when the running skill has finished, and every decide_every seconds
while a skill runs (it may answer "continue").
"""
import csv
import json
import os
import time

from brain.skills import Hands, Hover, make_skill, pad_info
from sim.env import DT, CageEnv

MISSION = "Find the red landing pad and land on it."
BRAIN_TIMEOUT_S = 4.0
BRAIN_GIVE_UP_S = 15.0


def run_mission(env, pilot, policy, mission=MISSION, decide_every=2.0, brain_latency=None,
                out_dir=None, save_frames=False, max_decisions=80, log=print):
    """Returns the result dict of env.result() plus brain statistics. Writes logs to out_dir."""
    obs = env.observe()
    hands = Hands()
    hands.set(Hover(0.5))
    history, decisions = [], []
    pending = None                 # (apply_at_sim_t, decision, latency_s, requested_at)
    last_decision_t = 0.0
    last_request_t = -99.0
    prev_jpeg = None
    current_entry = None
    steps = []
    frames = []
    while not env.done:
        t = env.t
        # 1. apply a decision whose thinking time has passed
        if pending and t >= pending[0]:
            _, d, lat, t_req = pending
            pending = None
            last_decision_t = t
            entry = {"t_request": round(t_req, 1), "t_apply": round(t, 1), "latency_s": round(lat, 2),
                     **{k: v for k, v in d.model_dump().items() if v is not None}}
            decisions.append(entry)
            log(f"  t={t:5.1f}s  {d.skill:8s} {_args(d):28s} ({lat:.1f} s) {d.reason}")
            if d.skill == "done":
                hands.set(Hover(1.0))
            elif d.skill != "continue":
                if current_entry is not None:
                    current_entry["ended"] = "replaced" if not hands.skill.finished else hands.skill.status
                skill = make_skill(d, policy)
                hands.set(skill)
                current_entry = {"skill": d.skill, "what": skill.describe(), "t": round(t, 1),
                                 "pad_y_before": pad_info(obs["jpeg"]).get("y", -1)}
                history.append(current_entry)
        # 2. ask the pilot (finished skill, or periodic check)
        need = hands.skill is None or hands.skill.finished or t - last_request_t >= decide_every
        if pending is None and need and len(decisions) < max_decisions:
            if current_entry is not None and hands.skill.finished and "ended" not in current_entry:
                current_entry["ended"] = hands.skill.status
            ctx = {"mission": mission, "history": history,
                   "current": None if hands.skill is None or hands.skill.finished else hands.skill.describe(),
                   "previous_jpeg": prev_jpeg}
            w0 = time.perf_counter()
            try:
                d = pilot.decide(obs, ctx)
                lat = brain_latency if brain_latency is not None else time.perf_counter() - w0
            except Exception as e:                  # brain unreachable: treat like no answer
                log(f"  t={t:5.1f}s  pilot error: {e}")
                d, lat = None, BRAIN_TIMEOUT_S
            last_request_t = t
            prev_jpeg = obs["jpeg"]
            if d is not None:
                pending = (t + lat, d, lat, t)
        # 3. safety fallback when the brain is silent
        silent = t - last_decision_t
        if hands.skill.finished and silent > BRAIN_GIVE_UP_S:
            env.auto_land()
        # 4. fly one step
        cmd, source = hands.step(obs)
        if cmd == "auto_land":
            env.auto_land()
            cmd = [0.0, 0.0, 0.0, 0.0]
        obs = env.step(cmd)
        s = env.drone.s
        steps.append([round(env.t, 1), round(s.x, 3), round(s.y, 3), round(s.z, 3), round(s.heading, 1),
                      *cmd, source, hands.skill.describe() if hands.skill else "", int(env.auto_landing)])
        if save_frames:
            frames.append(env.last_rgb)
    if current_entry is not None and "ended" not in current_entry:
        current_entry["ended"] = env.outcome
    res = env.result()
    lats = [d["latency_s"] for d in decisions]
    res.update({"decisions": len(decisions),
                "brain_latency_mean_s": round(sum(lats) / len(lats), 2) if lats else 0.0,
                "skills_used": sorted({h["skill"] for h in history})})
    if out_dir:
        _write(out_dir, res, steps, decisions, frames)
    return res


def _args(d):
    parts = []
    for k in ("degrees", "meters", "forward_m", "right_m", "seconds"):
        v = getattr(d, k)
        if v is not None:
            parts.append(f"{k}={v:g}")
    if d.instruction:
        parts.append(f"'{d.instruction[:20]}'")
    return " ".join(parts)


def _write(out_dir, res, steps, decisions, frames):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "steps.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "heading", "pitch", "roll", "throttle", "yaw", "source", "skill", "auto_landing"])
        w.writerows(steps)
    with open(os.path.join(out_dir, "decisions.jsonl"), "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    if frames:
        from PIL import Image
        imgs = [Image.fromarray(fr) for fr in frames[::2]]
        imgs[0].save(os.path.join(out_dir, "flight.gif"), save_all=True, append_images=imgs[1:],
                     duration=200, loop=0)
