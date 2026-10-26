"""Talks to inference_server.py. The model answers -1..1 per stick; this returns percent."""
import time

import requests


class PolicyClient:
    def __init__(self, url, timeout_s=2.0, instruction=None):
        self.url = url.rstrip("/") + "/act"
        self.timeout = timeout_s
        self.params = {"instruction": instruction} if instruction else None
        self.session = requests.Session()
        self.last_confidence = None      # {"p": [...], "spread_pct": [...]} of the last answer, or None

    def act(self, jpeg):
        """Returns (sticks in percent [pitch, roll, throttle, yaw], model_ms, round_trip_ms)."""
        t0 = time.perf_counter()
        r = self.session.post(self.url, data=jpeg, params=self.params, timeout=self.timeout,
                              headers={"Content-Type": "image/jpeg"})
        r.raise_for_status()
        j = r.json()
        rt = (time.perf_counter() - t0) * 1000
        self.last_confidence = j.get("confidence")
        return [100.0 * a for a in j["action"]], j.get("model_ms", 0.0), rt
