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
latency of each part) and events.csv (takeover, stop, landed, ...).
"""
import argparse, asyncio, csv, json, os, time
from datetime import datetime

import websockets

from brain.governor import Governor, limits_from_stats
from brain.protocol import unpack_frame

# Used when no dataset_statistics.json is given: roughly the 1 %-99 % range of the 50 pilot
# episodes (project report, section 5), in stick percent. Throttle was never positive.
DEFAULT_LIMITS = [(0.0, 45.0), (-38.0, 35.0), (-100.0, 0.0), (-37.0, 54.0)]


class MockPolicy:
    def act(self, jpeg):
        time.sleep(0.05)
        return [10.0, 0.0, -20.0, 0.0], 50.0, 50.0


class Session:
    def __init__(self, args, policy, governor):
        self.args = args
        self.policy = policy
        self.gov = governor
        self.latest = None            # (header, jpeg, t_recv)
        self.new_frame = asyncio.Event()
        self.dropped = 0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(args.log_dir, f"session_{stamp}")
        os.makedirs(os.path.join(self.dir, "frames") if args.save_frames else self.dir, exist_ok=True)
        self.steps = csv.writer(open(os.path.join(self.dir, "steps.csv"), "w", newline=""))
        self.steps.writerow(["seq", "mode", "height_m",
                             "pilot_pitch", "pilot_roll", "pilot_throttle", "pilot_yaw",
                             "model_pitch", "model_roll", "model_throttle", "model_yaw",
                             "sent_pitch", "sent_roll", "sent_throttle", "sent_yaw",
                             "wait_ms", "policy_ms", "model_ms", "pc_ms", "dropped_before"])
        self.events = csv.writer(open(os.path.join(self.dir, "events.csv"), "w", newline=""))
        self.events.writerow(["time", "seq", "event", "detail"])

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
        try:
            raw, model_ms, _ = await loop.run_in_executor(None, s.policy.act, jpeg)
        except Exception as e:
            s.event("policy_error", header.get("seq"), str(e))
            continue                    # no answer: the phone's watchdog will hover the drone
        policy_ms = (time.perf_counter() - t_start) * 1000
        sent = s.gov.apply(raw)
        pc_ms = (time.perf_counter() - t_recv) * 1000
        cmd = {"type": "cmd", "seq": header.get("seq"), "t_ms": header.get("t_ms"), "sticks": sent,
               "mode": s.args.mode, "model_ms": round(model_ms, 1), "pc_ms": round(pc_ms, 1)}
        await ws.send(json.dumps(cmd))
        tel = header.get("telemetry", {})
        pilot = tel.get("pilot", ["", "", "", ""])
        s.steps.writerow([header.get("seq"), s.args.mode, tel.get("height_m", ""), *pilot,
                          *[round(v, 1) for v in raw], *sent,
                          round((t_start - t_recv) * 1000, 1), round(policy_ms, 1),
                          round(model_ms, 1), round(pc_ms, 1), dropped])
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
                    s.event(m.get("name", "?"), m.get("seq", ""), json.dumps(m))
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
    limits = limits_from_stats(args.stats) if args.stats else DEFAULT_LIMITS
    governor = Governor(limits, scale=args.scale, allow_yaw=args.allow_yaw)
    print(f"[BRIDGE] mode={args.mode} policy={args.policy} limits%={governor.limits_abs()} "
          f"listening on ws://0.0.0.0:{args.port}")

    async def handler(ws, *_):
        s = Session(args, policy, governor)
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
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--save-frames", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main_async(parse()))
