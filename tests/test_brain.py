"""Run with:  python3 -m pytest tests/"""
import json

from brain.governor import Governor, limits_from_stats
from brain.protocol import pack_frame, unpack_frame

LIMITS = [(0.0, 45.0), (-38.0, 35.0), (-100.0, 0.0), (-37.0, 54.0)]


def test_frame_round_trip():
    header = {"type": "frame", "seq": 7, "t_ms": 1234, "telemetry": {"height_m": 1.2}}
    h, jpeg = unpack_frame(pack_frame(header, b"\xff\xd8jpegdata\xff\xd9"))
    assert h == header and jpeg == b"\xff\xd8jpegdata\xff\xd9"


def test_governor_clamps_to_training_range():
    g = Governor(LIMITS, max_step=1000)
    assert g.apply([90.0, -90.0, 50.0, 0.0]) == [45.0, -38.0, 0.0, 0.0]


def test_governor_limits_change_per_step():
    g = Governor(LIMITS, max_step=25)
    assert g.apply([45.0, 0.0, 0.0, 0.0])[0] == 25.0
    assert g.apply([45.0, 0.0, 0.0, 0.0])[0] == 45.0


def test_governor_yaw_off_unless_allowed():
    assert Governor(LIMITS, max_step=1000).apply([0, 0, 0, 40.0])[3] == 0.0
    assert Governor(LIMITS, max_step=1000, allow_yaw=True).apply([0, 0, 0, 40.0])[3] == 40.0


def test_governor_scale_and_reset():
    g = Governor(LIMITS, scale=0.5, max_step=1000)
    assert g.limits_abs() == [22.5, 19.0, 50.0, 27.0]
    g.apply([20.0, 0, 0, 0])
    g.reset()
    assert g.last == [0.0, 0.0, 0.0, 0.0]


def test_limits_from_openvla_statistics(tmp_path):
    stats = {"drone_dataset": {"action": {"q01": [0.0, -0.38, -1.0, -0.37], "q99": [0.45, 0.35, 0.0, 0.54]}}}
    p = tmp_path / "dataset_statistics.json"
    p.write_text(json.dumps(stats))
    assert limits_from_stats(str(p)) == [(0.0, 45.0), (-38.0, 35.0), (-100.0, 0.0), (-37.0, 54.0)]
