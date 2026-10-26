"""Round 2 tools. Run with:  python3 -m pytest tests/"""
import csv, json, os, sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import augment
from brain import ab_test, phases
from brain.bridge_server import BlindAB


def test_shifts_symmetric_and_inside_frame():
    s = augment.shifts(4)
    assert len(s) == 4 and 0 not in s
    assert sorted(s) == sorted(-x for x in s)
    assert max(abs(x) for x in s) <= (1 - augment.CROP) / 2 + 1e-9


def test_shift_right_means_roll_left():
    a = np.zeros((3, 4), np.float32)
    out = augment.corrected_actions(a, 0.05)
    assert np.allclose(out[:, 1], -0.10) and np.allclose(out[:, [0, 2, 3]], 0)
    assert np.all(augment.corrected_actions(np.full((1, 4), -0.95, np.float32), 0.05)[:, 1] >= -1.0)


def test_paraphrase_is_deterministic():
    phrases = ["a", "b", "c"]
    assert augment.paraphrase_for("ep1", phrases) == augment.paraphrase_for("ep1", phrases)


def test_split_val_is_real_episodes_only():
    pytest.importorskip("h5py")
    import drone_dataset_dataset_builder as b
    files = [f"/x/episode_{i:04d}.h5" for i in range(1, 21)]
    train, val = b.split_jobs(files, recovery_views=2, paraphrases=True)
    assert [j[0] for j in val] == files[-2:]                      # same val as before augmentation
    assert all(j[1] == 0 and j[2] is None for j in val)           # never shifted, original words
    assert len(train) == 18 * 3 and {j[0] for j in train} == set(files[:-2])
    assert len({j[3] for j in train + val}) == len(train) + len(val)   # unique TFDS keys


def test_phase_labels():
    assert phases.label([0, 0, 30, 0], True) == "climb"
    assert phases.label([30, 0, -30, 0], True) == "descend"
    assert phases.label([30, 0, 0, 0], False) == "search"
    assert phases.label([30, 0, 0, 0], True) == "approach"
    assert phases.label([3, -2, 0, 0], True) == "align"


def test_token_confidence_matches_openvla_mapping():
    from inference_server import token_confidence
    vocab, centers = 32000, (np.linspace(-1, 1, 256)[:-1] + np.linspace(-1, 1, 256)[1:]) / 2
    q01, q99 = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    logits = np.full((2, 32064), -50.0)
    logits[0, vocab - 1 - 200] = 50.0                  # certain: one bin
    logits[1, [vocab - 1 - 10, vocab - 1 - 244]] = 50.0   # torn between full left and full right
    c = token_confidence(logits, vocab, centers, q01, q99)
    assert c["p"][0] > 0.99 and c["spread_pct"][0] < 1
    assert c["p"][1] == pytest.approx(0.5, abs=0.01) and c["spread_pct"][1] > 80


def test_blind_ab_is_balanced_and_keyed(tmp_path):
    key = tmp_path / "ab_key.csv"
    ab = BlindAB({"A": "a", "B": "b"}, str(key), seed=1)
    arms = []
    for _ in range(6):
        ab.new_flight(str(tmp_path / "session_x"))
        arms.append(ab.current)
    assert all(sorted(arms[i:i + 2]) == ["A", "B"] for i in range(0, 6, 2))
    assert BlindAB({}, str(key))._last_flight() == 6               # numbering continues after a restart


def test_ab_report(tmp_path):
    (tmp_path / "session_1").mkdir()
    with open(tmp_path / "session_1" / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "seq", "event", "detail"])
        w.writerow(["t", 5, "takeover", json.dumps({"name": "takeover", "flight": 2})])
    key = [{"flight": str(i), "arm": a} for i, a in enumerate("ABABAB", 1)]
    scores = [{"flight": str(i), "distance_cm": d, "on_pad": o}
              for i, (d, o) in enumerate([(5, 1), (40, 0), (8, 1), (35, 0), (6, 1), (50, 0)], 1)]
    r = ab_test.summarise(key, scores, ab_test.interventions(str(tmp_path)))
    assert r["A"]["on_pad"] == 3 and r["B"]["on_pad"] == 0
    assert r["A"]["mean_cm"] == pytest.approx(6.3, abs=0.1) and r["B"]["takeovers"] == 1
    assert r["p_value"] < 0.2
    assert ab_test.permutation_p([1, 2, 3], [1, 2, 3]) > 0.5
