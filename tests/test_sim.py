"""Tests of the cage simulator, the skills and the pilots (no GPU, no API key needed)."""
import math
from types import SimpleNamespace

import numpy as np

from brain.phases import pad_in_frame
from brain.pilot import ClaudePilot, Decision, ScriptedPilot
from brain.skills import Climb, Hands, Move, Rotate, ServoSkill, make_skill
from sim.dynamics import Mini3, Params
from sim.env import CageEnv
from sim.mission import run_mission
from sim.render import Scene, render, to_jpeg
from sim.run import SeekPolicy, main as run_main


def test_forward_stick_flies_along_heading():
    d = Mini3(Params(noise=0))
    d.s.heading = 90.0                       # facing east
    for _ in range(30):
        d.step([40, 0, 0, 0], 0.1)
    assert d.s.x > 0.3 and abs(d.s.y) < 0.01
    tel = d.to_telemetry()
    assert tel["vy"] > 0.1 and abs(tel["vx"]) < 0.01          # NED: east speed in "vy"


def test_throttle_down_descends():
    d = Mini3(Params(noise=0))
    z0 = d.s.z
    for _ in range(20):
        d.step([0, 0, -100, 0], 0.1)
    assert d.s.z < z0 - 0.3


def test_pad_seen_ahead_not_behind():
    sc = Scene(pad_x=0.0, pad_y=0.5)
    ahead = to_jpeg(render(0.0, -0.3, 1.2, 0.0, sc))          # pad 0.8 m north, facing north
    behind = to_jpeg(render(0.0, -0.3, 1.2, 180.0, sc))
    assert pad_in_frame(ahead)[0]
    # behind: only the small red tape marks on the lab floor (they are in the video too)
    assert pad_in_frame(behind)[3] < 0.01 < pad_in_frame(ahead)[3]


def test_pad_left_appears_left():
    sc = Scene(pad_x=-0.5, pad_y=0.5)
    vis, x, y, _ = pad_in_frame(to_jpeg(render(0.0, -0.4, 1.2, 0.0, sc)))
    assert vis and x < -0.2


def test_net_hit_ends_flight():
    env = CageEnv(seed=1, noise=False)
    env.reset(pad=(0.5, 0.5, 0), start=(0.0, 0.0), heading=0)
    for _ in range(100):
        env.step([60, 0, 0, 0])
        if env.done:
            break
    assert env.outcome == "hit_net"


def test_auto_landing_on_pad():
    env = CageEnv(seed=2, noise=False)
    env.reset(pad=(0.1, 0.1, 0), start=(0.1, 0.1), heading=0)
    env.auto_land()
    while not env.done:
        env.step([0, 0, 0, 0])
    assert env.outcome == "landed_on_pad"


def test_rotate_and_climb_and_move_skills():
    env = CageEnv(seed=3, noise=False)
    env.reset(pad=(0.5, 0.5, 0), start=(0.0, 0.0), heading=10)
    hands = Hands()
    for skill, check in ((Rotate(90), lambda s: abs(((s.heading - 100) + 180) % 360 - 180) < 6),
                         (Climb(0.4), lambda s: abs(s.z - 1.6) < 0.1),
                         (Move(0.4, 0.0), lambda s: abs(s.x - 0.4) < 0.12)):
        hands.set(skill)
        obs = env.observe()
        for _ in range(150):
            cmd, _ = hands.step(obs)
            obs = env.step(cmd)
            if skill.finished:
                break
        assert skill.status == "done", skill.describe()
        assert check(env.drone.s), (skill.describe(), env.drone.s)


def test_make_skill_from_decision():
    d = Decision(observation="", pad_visible=False, skill="rotate", degrees=45, reason="")
    assert isinstance(make_skill(d, None), Rotate)
    assert make_skill({"skill": "land"}, None).name == "land"


def test_scripted_mission_lands_on_pad():
    ServoSkill.target_y = 0.75
    try:
        env = CageEnv(seed=4)
        res = run_mission(env, ScriptedPilot(approach="servo"), SeekPolicy(), brain_latency=1.5, log=lambda *a: None)
    finally:
        ServoSkill.target_y = None
    assert res["outcome"] == "landed_on_pad", res
    assert res["decisions"] > 1


class FakeMessages:
    def __init__(self, decision):
        self.decision = decision
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(stop_reason="end_turn", parsed_output=self.decision)


def test_claude_pilot_request_shape():
    dec = Decision(observation="pad ahead", pad_visible=True, skill="vla", seconds=4, reason="approach")
    fake = FakeMessages(dec)
    pilot = ClaudePilot(client=SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    env = CageEnv(seed=5)
    obs = env.observe()
    out = pilot.decide(obs, {"mission": "land on the pad", "history": [], "current": None,
                             "previous_jpeg": obs["jpeg"]})
    assert out.skill == "vla"
    k = fake.kwargs
    content = k["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert k["system"][0]["cache_control"]["type"] == "ephemeral"
    assert k["output_format"] is Decision and k["fallbacks"] == "default"


def test_claude_pilot_refusal_hovers():
    fake = FakeMessages(None)
    fake.parse = lambda **kw: SimpleNamespace(stop_reason="refusal", parsed_output=None)
    pilot = ClaudePilot(client=SimpleNamespace(beta=SimpleNamespace(messages=fake)))
    obs = CageEnv(seed=6).observe()
    assert pilot.decide(obs, {"history": []}).skill == "hover"


def test_brain_silence_lands_the_drone():
    class Silent:
        def decide(self, obs, ctx):
            raise TimeoutError("no network")
    env = CageEnv(seed=7, max_time=60)
    res = run_mission(env, Silent(), SeekPolicy(), log=lambda *a: None)
    assert res["outcome"] in ("landed_on_pad", "landed_off_pad")
    assert res["time_s"] < 40


def test_run_cli(tmp_path):
    rows = run_main(["--pilot", "none", "--policy", "seek", "--missions", "1", "--seed", "0",
                     "--max-time", "30", "--out", str(tmp_path)])
    assert len(rows) == 1 and (tmp_path / "summary.csv").exists()
