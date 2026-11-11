"""
fit_dynamics.py  -  fit the Mini 3 motion model (dynamics.py) to recorded episodes.

    python3 -m sim.fit_dynamics ~/drone_database/incoming/episode_*        # all 50
    python3 -m sim.fit_dynamics <episodes> --save                          # write sim/mini3_params.json

For each axis the recorded stick is pushed through the model's first-order response for a
range of time constants; for each time constant the gain is a least-squares fit, and the
time constant with the smallest error wins. The telemetry is coarse (speeds in steps of
0.1 m/s, height in steps of 0.1 m), so more episodes give a better fit. The script prints
the fitted constants and how much of the speed variation they explain (R^2).
"""
import argparse, glob, math, os

import numpy as np
import pandas as pd

from sim.dynamics import Params, curve

MPH = 0.44704
FT = 0.3048
TAUS = np.round(np.arange(0.05, 2.01, 0.05), 2)


def load(ep):
    df = pd.read_csv(sorted(glob.glob(os.path.join(ep, "*telemetry_*.csv")))[0])
    df = df[df["flight_mode"] != "AUTO_LANDING"].reset_index(drop=True)
    t = df["time_ms"].to_numpy() / 1000.0
    h = np.radians(df["heading"].to_numpy())
    vn, ve = df["x_speed"].to_numpy() * MPH, df["y_speed"].to_numpy() * MPH
    fwd = vn * np.cos(h) + ve * np.sin(h)
    right = -vn * np.sin(h) + ve * np.cos(h)
    # climb speed from the height (z_speed is mostly 0 in these logs); smoothed over 0.5 s
    hgt = pd.Series(df["height_above_takeoff"].to_numpy() * FT).rolling(5, center=True, min_periods=1).mean()
    up = np.gradient(hgt.to_numpy(), t) if len(t) > 2 else np.zeros_like(t)
    head = np.unwrap(h)
    yaw_rate = np.degrees(np.gradient(head, t)) if len(t) > 2 else np.zeros_like(t)
    c = lambda col: np.array([curve(v) for v in df[col]])
    return {"t": t, "fwd": (c("pitch_cmd"), fwd), "right": (c("roll_cmd"), right),
            "up": (df["throttle_cmd"].to_numpy() / 100.0, up), "yaw": (c("yaw_cmd"), yaw_rate)}


def response(u, t, tau):
    """First-order response of unit gain to input u (sampled at times t)."""
    y = np.zeros_like(u)
    for i in range(1, len(u)):
        a = min(1.0, (t[i] - t[i - 1]) / tau)
        y[i] = y[i - 1] + (u[i - 1] - y[i - 1]) * a
    return y


def fit_axis(episodes, axis):
    best = None
    for tau in TAUS:
        xs, ys = [], []
        for e in episodes:
            u, y = e[axis]
            xs.append(response(u, e["t"], tau))
            ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
        if (x ** 2).sum() < 1e-9:
            return None
        k = float((x * y).sum() / (x ** 2).sum())
        err = float(((y - k * x) ** 2).mean())
        if best is None or err < best[2]:
            r2 = 1 - err / max(float(y.var()), 1e-12)
            best = (k, float(tau), err, r2)
    return best


def main():
    ap = argparse.ArgumentParser(description="fit the Mini 3 motion model to recorded episodes")
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--save", action="store_true", help="write sim/mini3_params.json")
    args = ap.parse_args()
    eps = [load(e) for e in args.episodes]
    p = Params()
    res = {}
    for axis in ("fwd", "right", "up", "yaw"):
        r = fit_axis(eps, axis)
        res[axis] = r
        if r is None:
            print(f"{axis:6s}: no stick input on this axis, default kept")
            continue
        k, tau, err, r2 = r
        print(f"{axis:6s}: gain {k:7.3f}  tau {tau:.2f} s  rms error {math.sqrt(err):.3f}  R^2 {r2:.2f}")
    if res["fwd"] and res["right"]:
        p.k_fwd, p.k_right = res["fwd"][0], res["right"][0]
        p.tau = round((res["fwd"][1] + res["right"][1]) / 2, 2)
    elif res["fwd"]:
        p.k_fwd, p.tau = res["fwd"][0], res["fwd"][1]
    if res["up"]:
        p.k_up = abs(res["up"][0])
    if res["yaw"] and res["yaw"][0] > 0:
        p.k_yaw, p.tau_yaw = res["yaw"][0], res["yaw"][1]
    p.k_fwd, p.k_right, p.k_up, p.k_yaw = (round(v, 3) for v in (p.k_fwd, p.k_right, p.k_up, p.k_yaw))
    print(f"\nparameters: {p}")
    if args.save:
        p.save()
        print("saved to sim/mini3_params.json")


if __name__ == "__main__":
    main()
