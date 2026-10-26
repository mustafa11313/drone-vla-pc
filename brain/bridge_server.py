"""
bridge_server.py  -  the PC brain. The phone app connects here; the model runs elsewhere.

    phone --frames + telemetry--> bridge_server --JPEG--> inference_server (GPU)
    phone <--stick command------- bridge_server <--4 values--

Run on the laptop (same Wi-Fi as the phone):

    python3 -m brain.bridge_server --policy http://<gpu-host>:8000 --stats <ckpt>/dataset_statistics.json
    python3 -m brain.bridge_server --policy mock          # no model at all, for tests

--mode shadow (default): the model's commands go to the phone but the phone only SHOWS them;
                         you fly. The log then compares the model with your sticks.
--mode active          : the phone may apply them, but only after you press MODEL FLY on the
                         phone as well (two keys: the PC and the pilot must both agree).

Only the newest frame is used: if the model is still busy, older frames are dropped (and
counted), so commands never queue up and get stale.

Every session writes logs/session_<date>_<time>/steps.csv (one row per command, with the
latency of each part and the model's confidence) and events.csv (takeover, stop, landed, ...).

--max-spread 15 : when the model is unsure (OpenVLA's spread of pitch or roll above 15 stick
                  percent, see inference_server.py), send hover instead of its command (idea M1).

Blind A/B test of two checkpoints (idea E2), see brain/ab_test.py:

    python3 -m brain.bridge_server --policy http://<gpu>:8000 --policy-b http://<gpu>:8001 --mode active

Every flight (from MODEL FLY to landed / takeover / stop) secretly uses A or B, balanced in
blocks of two. The choice goes only to logs/ab_key.csv; the screen and steps.csv just say
"flight 7". Score the landings blind, then unblind with brain/ab_test.py.
"""
import argparse, asyncio, csv, json, os, random, time
from datetime import datetime

import websockets

from brain.governor import Governor, limits_from_stats
from brain.protocol import unpack_frame

# Used when no dataset_statistics.json is given: roughly the 1 %-99 % range of the 50 pilot
# episodes (project report, section 5), in stick percent. Throttle was never positive.
DEFAULT_LIMITS = [(0.0, 45.0), (-38.0, 35.0), (-100.0, 0.0), (-37.0, 54.0)]


class MockPolicy:
    last_confidence = {"p": [0.9, 0.9, 0.9, 0.9], "spread_pct": [3.0, 3.0, 3.0, 0.0]}

    def act(self, jpeg):
        time.sleep(0.05)
        return [10.0, 0.0, -20.0, 0.0], 50.0, 50.0


class BlindAB:
    """Picks A or B per flight, balanced in shuffled blocks of two; only ab_key.csv knows which."""

    def __init__(self, policies, key_path, seed=None):
        self.policies = policies                    # {"A": client, "B": client}
        self.key_path = key_path
        self.rng = random.Random(seed)
        self.block = []
        self.flight = self._last_flight()
        self.current = None

    def _last_flight(self):
        if not os.path.exists(self.key_path):
            return 0
        rows = list(csv.reader(open(self.key_path)))[1:]
        return max((int(r[0]) for r in rows if r), default=0)

    def new_flight(self, session_dir):
        if not self.block:
            self.block = ["A", "B"]
            self.rng.shuffle(self.block)
        self.current = self.block.pop()
        self.flight += 1
        new = not os.path.exists(self.key_path)
        with open(self.key_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["flight", "arm", "session", "time"])
            w.writerow([self.flight, self.current, os.path.basename(session_dir),
                        time.strftime("%Y-%m-%d %H:%M:%S")])
        return self.flight

    def policy(self):
        return self.policies[self.current or "A"]


class Session:
    def __init__(self, args, policy, governor, ab=None):
        self.args = args
        self.policy = policy
        self.ab = ab                  # BlindAB or None
        self.gov = governor
        self.latest = None            # (header, jpeg, t_recv)
        self.new_frame = asyncio.Event()
        self.dropped = 0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(args.log_dir, f"session_{stamp}")
        os.makedirs(os.path.join(self.dir, "frames") if args.save_frames else self.dir, exist_ok=True)
        self.steps = csv.writer(open(os.path.join(self.dir, "steps.csv"), "w", newline=""))
        self.steps.writerow(["seq", "flight", "mode", "height_m",
                             "pilot_pitch", "pilot_roll", "pilot_throttle", "pilot_yaw",
                             "model_pitch", "model_roll", "model_throttle", "model_yaw",
                             "sent_pitch", "sent_roll", "sent_throttle", "sent_yaw",
                             "wait_ms", "policy_ms", "model_ms", "pc_ms", "dropped_before",
                             "p_pitch", "p_roll", "p_throttle", "p_yaw",
                             "spread_pitch", "spread_roll", "spread_throttle", "spread_yaw", "unsure"])
        self.events = csv.writer(open(os.path.join(self.dir, "events.csv"), "w", newline=""))
        self.events.writerow(["time", "seq", "event", "detail"])
        self.flight = ""              # blind A/B flight number, "" when not testing

    def event(self, name, seq="", detail=""):
        self.events.writerow([datetime.now().isoformat(timespec="milliseconds"), seq, name, detail])
        print(f"[EVENT] {name} {detail}")


async def worker(ws, s):
    loop = asyncio.get_running_loop()
    while True:
        await s.new_frame.wait()
        s.new_frame.clear()
        header, jpeg, t_recv = s.latest
        dropped, s.dropped = s.dropped, 0
        t_start = time.perf_counter()
        policy = s.ab.policy() if s.ab else s.policy
        try:
            raw, model_ms, _ = await loop.run_in_executor(None, policy.act, jpeg)
        except Exception as e:
            s.event("policy_error", header.get("seq"), str(e))
            continue                    # no answer: the phone's watchdog will hover the drone
        policy_ms = (time.perf_counter() - t_start) * 1000
        conf = getattr(policy, "last_confidence", None) or {}
        p, spread = conf.get("p") or [""] * 4, conf.get("spread_pct") or [""] * 4
        unsure = bool(s.args.max_spread and conf.get("spread_pct")
                      and max(spread[0], spread[1]) > s.args.max_spread)
        sent = s.gov.apply([0.0, 0.0, 0.0, 0.0] if unsure else raw)
        pc_ms = (time.perf_counter() - t_recv) * 1000
        cmd = {"type": "cmd", "seq": header.get("seq"), "t_ms": header.get("t_ms"), "sticks": sent,
               "mode": s.args.mode, "model_ms": round(model_ms, 1), "pc_ms": round(pc_ms, 1),
               "confidence": conf or None, "unsure": unsure}
        await ws.send(json.dumps(cmd))
        tel = header.get("telemetry", {})
        pilot = tel.get("pilot", ["", "", "", ""])
        s.steps.writerow([header.get("seq"), s.flight, s.args.mode, tel.get("height_m", ""), *pilot,
                          *[round(v, 1) for v in raw], *sent,
                          round((t_start - t_recv) * 1000, 1), round(policy_ms, 1),
                          round(model_ms, 1), round(pc_ms, 1), dropped, *p, *spread, int(unsure)])
        if s.args.save_frames:
            with open(os.path.join(s.dir, "frames", f"{header.get('seq', 0):06d}.jpg"), "wb") as f:
                f.write(jpeg)


async def handle(ws, s):
    await ws.send(json.dumps({"type": "config", "mode": s.args.mode, "limits_pct": s.gov.limits_abs(),
                              "land_height_m": s.args.land_height}))
    s.event("connected", detail=str(ws.remote_address))
    task = asyncio.create_task(worker(ws, s))
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                header, jpeg = unpack_frame(msg)
                if s.new_frame.is_set():
                    s.dropped += 1           # the previous frame was never processed
                s.latest = (header, jpeg, time.perf_counter())
                s.new_frame.set()
            else:
                m = json.loads(msg)
                if m.get("type") == "event":
                    if m.get("name") in ("takeover", "stop", "landed"):
                        s.gov.reset()
                    if m.get("name") == "armed" and s.ab:
                        s.flight = s.ab.new_flight(s.dir)
                        print(f"[A/B] flight {s.flight} (arm hidden in {s.ab.key_path})")
                    s.event(m.get("name", "?"), m.get("seq", ""),
                            json.dumps({**m, "flight": s.flight}) if s.ab else json.dumps(m))
                elif m.get("type") == "hello":
                    s.event("hello", detail=json.dumps(m))
    except websockets.ConnectionClosed:
        pass
    finally:
        task.cancel()
        s.event("disconnected")


async def main_async(args):
    if args.policy == "mock":
        policy = MockPolicy()
    else:
        from brain.policy_client import PolicyClient
        policy = PolicyClient(args.policy)
    ab = None
    if args.policy_b:
        from brain.policy_client import PolicyClient
        b = MockPolicy() if args.policy_b == "mock" else PolicyClient(args.policy_b)
        os.makedirs(args.log_dir, exist_ok=True)
        ab = BlindAB({"A": policy, "B": b}, os.path.join(args.log_dir, "ab_key.csv"))
    limits = limits_from_stats(args.stats) if args.stats else DEFAULT_LIMITS
    governor = Governor(limits, scale=args.scale, allow_yaw=args.allow_yaw)
    print(f"[BRIDGE] mode={args.mode} policy={args.policy}{' + blind B' if ab else ''} "
          f"limits%={governor.limits_abs()} "
          f"listening on ws://0.0.0.0:{args.port}")

    async def handler(ws, *_):
        s = Session(args, policy, governor, ab)
        governor.reset()
        print(f"[BRIDGE] logging to {s.dir}")
        await handle(ws, s)

    async with websockets.serve(handler, "0.0.0.0", args.port, max_size=8 * 1024 * 1024):
        await asyncio.Future()


def parse(argv=None):
    ap = argparse.ArgumentParser(description="PC brain: phone <-> model bridge")
    ap.add_argument("--policy", required=True, help="inference server URL, or 'mock'")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mode", choices=["shadow", "active"], default="shadow")
    ap.add_argument("--stats", help="dataset_statistics.json of the checkpoint (stick limits)")
    ap.add_argument("--scale", type=float, default=1.0, help="shrink the stick limits, e.g. 0.5")
    ap.add_argument("--allow-yaw", action="store_true", help="let the model turn (off: yaw = 0)")
    ap.add_argument("--land-height", type=float, default=0.6,
                    help="phone starts DJI auto-landing below this height (m) while descending")
    ap.add_argument("--max-spread", type=float, default=0.0,
                    help="hover when the model's pitch or roll spread is above this (stick %%); 0 = off")
    ap.add_argument("--policy-b", help="second inference server URL: blind A/B test per flight")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--save-frames", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main_async(parse()))
