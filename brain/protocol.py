"""
Messages between the phone app and the PC brain (one WebSocket, ws://<pc-ip>:8765).

Phone -> PC, one BINARY message per camera frame (10 per second):

    [4 bytes: header length, big-endian][header: UTF-8 JSON][JPEG bytes]

    header = {"type": "frame", "seq": 17, "t_ms": 123456,       # phone clock, only echoed back
              "telemetry": {"height_m": 1.8, "vx": 0.1, "vy": 0.0, "vz": -0.2, "heading": 41.0,
                            "pilot": [pitch, roll, throttle, yaw],   # physical sticks, percent
                            "flying": true, "vs_enabled": true}}

Phone -> PC, TEXT messages:  {"type": "hello", "app": "...", "stick_mode": 2}
                             {"type": "event", "name": "takeover" | "stop" | "landed" | ..., "seq": 17}

PC -> phone, TEXT messages:

    {"type": "config", "mode": "shadow" | "active", "limits_pct": [p, r, t, y], "land_height_m": 0.6}
    {"type": "cmd", "seq": 17, "t_ms": 123456, "sticks": [pitch, roll, throttle, yaw],   # percent
     "mode": "shadow", "model_ms": 182.4, "pc_ms": 190.1}

Sticks are always stick PERCENT (-100..100) in the order pitch, roll, throttle, yaw, exactly
like the *_cmd columns of the telemetry CSV. The phone converts them to tilt / turn rate /
climb speed with the same curve as replay.py and applies its own limits on top.
The phone measures the full delay itself: now - t_ms of the frame the command answers.
"""
import json
import struct

STICKS = ("pitch", "roll", "throttle", "yaw")


def pack_frame(header, jpeg):
    h = json.dumps(header).encode()
    return struct.pack(">I", len(h)) + h + jpeg


def unpack_frame(data):
    if len(data) < 4:
        raise ValueError("frame message too short")
    (n,) = struct.unpack(">I", data[:4])
    if n > len(data) - 4:
        raise ValueError("header length larger than message")
    header = json.loads(data[4:4 + n].decode())
    return header, data[4 + n:]
