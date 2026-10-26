"""
rehearsal.py  -  the ground rehearsal: run recorded episodes through the live chain, no drone.

Two ways:

1. Model only (is the server right and fast enough?):

       python3 -m brain.rehearsal --policy http://<gpu-host>:8000 episode_0001_* episode_0002_*

   Sends every frame of the flown part (same trimming as csv_to_hdf5.py) to the inference
   server and compares the answer with the pilot's sticks: per-stick MAE in percent, and the
   model time. For validation episodes the MAE should match eval_val.py.

2. Whole chain (pretends to be the phone):

       python3 -m brain.bridge_server --policy http://<gpu-host>:8000      # in one terminal
       python3 -m brain.rehearsal --bridge ws://127.0.0.1:8765 episode_0001_*

   Streams the frames to the bridge at the recorded 10 Hz with the recorded telemetry, exactly
   like the app will, and measures for every frame how long the command took to come back
   (p50 / p95), how many frames got a command, and the command rate. If the rate is well below
   10 Hz, every command lasts longer than in the training data: see the project notes.

Episode folders are the ones the phone uploads (telemetry CSV + frames/).
"""
import argparse, asyncio, glob, json, os, statistics, time

import pandas as pd

CMD_COLS = ["pitch_cmd", "roll_cmd", "throttle_cmd", "yaw_cmd"]
FT_TO_M = 0.3048
MPH_TO_MS = 0.44704


def trim(df):
    """Same as csv_to_hdf5.trim: first stick input .. start of AUTO_LANDING (exclusive)."""
    moving = (df[CMD_COLS] != 0).any(axis=1)
    start = int(moving.idxmax()) if moving.any() else 0
    landing = df.index[df["flight_mode"] == "AUTO_LANDING"]
    end = int(landing[0]) if len(landing) else len(df)
    return df.iloc[start:end]


def load_episode(ep_dir):
    """[(row index, telemetry row, jpeg bytes)] of the flown part of one episode folder."""
    csvs = sorted(glob.glob(os.path.join(ep_dir, "*telemetry_*.csv")))
    if not csvs:
        raise SystemExit(f"no telemetry CSV in {ep_dir}")
    df = pd.read_csv(csvs[0])
    frames = sorted(glob.glob(os.path.join(ep_dir, "frames", "*.jpg")))
    if len(frames) != len(df):
        raise SystemExit(f"{ep_dir}: {len(df)} rows but {len(frames)} frames")
    out = []
    for i, row in trim(df).iterrows():
        with open(frames[i], "rb") as f:
            out.append((i, row, f.read()))
    return out


def telemetry_of(row):
    """CSV row (Airdata units) -> the telemetry block the app sends (metres, m/s)."""
    return {"height_m": round(row["height_above_takeoff"] * FT_TO_M, 3),
            "vx": round(row["x_speed"] * MPH_TO_MS, 3), "vy": round(row["y_speed"] * MPH_TO_MS, 3),
            "vz": round(row["z_speed"] * MPH_TO_MS, 3), "heading": float(row["heading"]),
            "pilot": [int(row[c]) for c in CMD_COLS], "flying": True, "vs_enabled": True}


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def run_model_only(policy_url, episodes):
    from brain.policy_client import PolicyClient
    from brain.phases import PHASES, label_episode
    client = PolicyClient(policy_url, timeout_s=10)
    err = [[], [], [], []]
    by_phase = {p: [] for p in PHASES}          # mean |error| over the 4 sticks, per step
    model_ms, total_ms = [], []
    for ep in episodes:
        steps = load_episode(ep)
        labels = label_episode(ep, steps)
        for (_, row, jpeg), lab in zip(steps, labels):
            sticks, m_ms, rt = client.act(jpeg)
            e = [abs(sticks[k] - row[c]) for k, c in enumerate(CMD_COLS)]
            for k in range(4):
                err[k].append(e[k])
            by_phase[lab["phase"]].append(statistics.mean(e))
            model_ms.append(m_ms)
            total_ms.append(rt)
        print(f"{os.path.basename(ep.rstrip('/'))}: {len(steps)} frames")
    print(f"\n{len(total_ms)} frames")
    print("stick      MAE (stick %)")
    for k, c in enumerate(CMD_COLS):
        print(f"{c[:-4]:9s} {statistics.mean(err[k]):8.1f}")
    print("phase      MAE (all sticks)   steps   (labels from brain/phases.py)")
    for p, v in by_phase.items():
        if v:
            print(f"{p:9s} {statistics.mean(v):8.1f}        {len(v):6d}")
    print(f"model time    p50 {pct(model_ms, .5):6.0f} ms   p95 {pct(model_ms, .95):6.0f} ms")
    print(f"with network  p50 {pct(total_ms, .5):6.0f} ms   p95 {pct(total_ms, .95):6.0f} ms"
          f"   -> at most {1000 / max(pct(total_ms, .5), 1):.1f} commands per second")


async def run_bridge(url, episodes, rate_hz):
    import websockets
    from brain.protocol import pack_frame
    sent_at, delays, n_sent = {}, [], 0
    first_cmd = last_cmd = None
    async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
        print("bridge says:", await ws.recv())
        await ws.send(json.dumps({"type": "hello", "app": "rehearsal.py", "stick_mode": 2}))

        async def reader():
            nonlocal first_cmd, last_cmd
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") == "cmd" and m.get("seq") in sent_at:
                    now = time.perf_counter()
                    delays.append((now - sent_at.pop(m["seq"])) * 1000)
                    first_cmd = first_cmd or now
                    last_cmd = now

        task = asyncio.create_task(reader())
        seq, t0 = 0, time.perf_counter()
        for ep in episodes:
            for _, row, jpeg in load_episode(ep):
                seq += 1
                header = {"type": "frame", "seq": seq, "t_ms": int((time.perf_counter() - t0) * 1000),
                          "telemetry": telemetry_of(row)}
                sent_at[seq] = time.perf_counter()
                await ws.send(pack_frame(header, jpeg))
                n_sent += 1
                await asyncio.sleep(max(0.0, t0 + seq / rate_hz - time.perf_counter()))
        await asyncio.sleep(2.0)           # let the last answers arrive
        await ws.send(json.dumps({"type": "event", "name": "landed", "seq": seq}))
        task.cancel()
    span = (last_cmd - first_cmd) if delays and last_cmd > first_cmd else 0
    print(f"\nframes sent {n_sent}, commands back {len(delays)} ({100 * len(delays) / max(n_sent, 1):.0f} %)")
    if delays:
        print(f"frame -> command  p50 {pct(delays, .5):6.0f} ms   p95 {pct(delays, .95):6.0f} ms"
              f"   max {max(delays):6.0f} ms")
        if span:
            print(f"command rate      {(len(delays) - 1) / span:.1f} per second (data was recorded at {rate_hz:g})")


def main():
    ap = argparse.ArgumentParser(description="ground rehearsal with recorded episodes")
    ap.add_argument("episodes", nargs="+", help="episode folders (telemetry CSV + frames/)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy", help="inference server URL: test the model only")
    g.add_argument("--bridge", help="bridge WebSocket URL: pretend to be the phone")
    ap.add_argument("--rate", type=float, default=10.0, help="frames per second sent to the bridge")
    args = ap.parse_args()
    if args.policy:
        run_model_only(args.policy, args.episodes)
    else:
        asyncio.run(run_bridge(args.bridge, args.episodes, args.rate))


if __name__ == "__main__":
    main()
