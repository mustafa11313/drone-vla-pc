"""Round 4: hybrid controller, unfamiliar scenes, Arabic commands, judge helpers, failure library, tracking."""
import csv, io, json, os

import numpy as np
from PIL import Image, ImageDraw

from brain import failures, judge, language, ood, servo, tracking
from brain.governor import Governor


def frame(pad_xy=None, size=(320, 180), floor=(120, 120, 120)):
    img = Image.new("RGB", size, floor)
    if pad_xy:
        x, y = pad_xy
        ImageDraw.Draw(img).ellipse([x - 25, y - 15, x + 25, y + 15], fill=(220, 20, 20))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_servo_steers_toward_pad_and_descends_when_aligned():
    s = servo.Servo()
    right = s.command(frame((280, 99)), now=0.0)        # pad right of centre
    assert right is not None and right[1] > 5           # roll right
    s.reset()
    aligned = s.command(frame((160, 140)), now=0.0)     # centred, near the target row
    assert aligned[2] < 0                               # descending
    assert all(abs(v) <= servo.MAX_PCT for v in aligned[:2])
    assert s.command(frame(None), now=1.0) is not None  # just lost: keeps descending
    assert s.command(frame(None), now=5.0) is None      # lost for long: model takes over


def test_servo_policy_hovers_without_pad():
    cmd, _, _ = servo.ServoPolicy().act(frame(None))
    assert cmd == [0.0, 0.0, 0.0, 0.0]


def test_governor_authority_shrinks_limits():
    g = Governor([(-40, 40)] * 4, scale=1.0, max_step=100, allow_yaw=True)
    assert g.apply([40, 0, 0, 0], 0.5)[0] == 20


def test_ood_flags_a_new_floor(tmp_path):
    rng = np.random.default_rng(0)
    feats = np.stack([ood.features(frame((rng.integers(60, 260), rng.integers(40, 140)))) for _ in range(30)])
    ref = tmp_path / "ref.npz"
    np.savez(ref, feats=feats, threshold=1.0)
    f = ood.Familiarity(str(ref))
    states = [f.update(frame((160, 90), floor=(30, 140, 40))) for _ in range(ood.RUN)]
    assert states[-1][1] is True and states[-1][2] is True   # flips after RUN frames
    assert states[0][1] is False


def test_arabic_word_list_and_english_passthrough(monkeypatch):
    monkeypatch.setattr(language, "with_claude", lambda t: None)    # no network in tests
    assert language.to_instruction("انزل على المنصة الحمراء") == (language.TRAINING_SENTENCE, "arabic word list")
    assert language.to_instruction("land on the pad") == ("land on the pad", "as said")
    assert language.to_instruction("مرحبا")[1].startswith("not understood")


def test_judge_pick_keeps_last_frame_and_reads_flight():
    frames = list(range(100))
    p = judge.pick(frames)
    assert p[-1] == 99 and len(p) == judge.N_FRAMES
    assert judge.event_flight({"detail": json.dumps({"flight": 10})}) == 10
    assert judge.event_flight({"detail": "plain text"}) is None


def test_failures_classify():
    ev = lambda name, text="": {"event": name, "text": text}
    steps = [{"unsure": "0", "pad_visible": "1", "pad_x": "0.8", "pad_y": "0", "sent_pitch": "0",
              "model_yaw": "0"}] * 30
    assert failures.classify(steps, [ev("feedback", "up")]) == (False, "", "")
    failed, cause, note = failures.classify(steps, [ev("feedback", "down"), ev("feedback_reason", "too far left")])
    assert failed and cause == "drift" and note == "too far left"
    assert failures.classify(steps, [ev("takeover"), ev("ood_on")])[1] == "unfamiliar"
    assert failures.classify(steps, [ev("stop")], {"on_pad": "0", "category": "overshoot"})[1] == "overshoot"


def test_failures_collect_from_session(tmp_path):
    sdir = tmp_path / "session_20261101_120000"
    sdir.mkdir()
    with open(sdir / "steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "flight", "unsure", "pad_visible", "pad_x", "pad_y", "sent_pitch", "model_yaw"])
        for i in range(30):
            w.writerow([i, 1, 1, 1, 0, 0, 0, 0])
    with open(sdir / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "seq", "event", "detail"])
        w.writerow(["t", 5, "feedback", json.dumps({"name": "feedback", "detail": "down", "flight": 1})])
    res = failures.collect(str(tmp_path))
    assert res == [{"session": sdir.name, "flight": 1, "failed": True, "cause": "unsure", "note": ""}]


def test_manifest(tmp_path):
    class A:
        policy, policy_b, stats, ood, mode, scale = "mock", None, None, None, "assist", 0.5
    m = tracking.write_manifest(str(tmp_path), A())
    assert m["model"] == {"name": "mock"} and "commit" in m["pc_code"]
    tracking.add_to_manifest(str(tmp_path), "app", {"version": "1.4"})
    assert json.load(open(tmp_path / "manifest.json"))["app"]["version"] == "1.4"
