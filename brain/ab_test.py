"""
ab_test.py  -  unblind a blind A/B test of two checkpoints (idea E2).

Why blind: whoever scores the landings, and the pilot deciding when to take over, must not
know which model is flying, or their expectations leak into the numbers.

1. Two inference servers (two checkpoints), one bridge with both:
       python3 inference_server.py ~/runs/model_A --port 8000
       python3 inference_server.py ~/runs/model_B --port 8001
       python3 -m brain.bridge_server --policy http://<gpu>:8000 --policy-b http://<gpu>:8001 --mode active
   Each MODEL FLY press starts a new flight; the bridge prints only "flight 7". A or B is
   written to logs/ab_key.csv (balanced: every 2 flights hold one A and one B). Don't open it.
2. After each landing, score it blind (overhead camera, before / after photo):
       python3 -m brain.landing_score before.jpg after.jpg --pad-cm 50 --flight 7
   (or type flight,distance_cm,on_pad into logs/ab_scores.csv by hand.)
3. When all flights are done:
       python3 -m brain.ab_test
   Per model: flights, landings on the pad, distance (mean / median), takeovers and STOPs,
   and a permutation test: the chance that a difference this large in mean distance appears
   when both models are in truth equally good (p < 0.05: the difference is probably real).
"""
import argparse, csv, glob, json, os, statistics

import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def interventions(log_dir):
    """{flight: {"takeover": n, "stop": n, "watchdog_handback": n}} from all events.csv files."""
    out = {}
    for path in glob.glob(os.path.join(log_dir, "session_*", "events.csv")):
        for row in read_csv(path):
            try:
                detail = json.loads(row.get("detail") or "{}")
            except json.JSONDecodeError:
                continue
            flight = detail.get("flight") if isinstance(detail, dict) else None
            if flight in (None, ""):
                continue
            d = out.setdefault(int(flight), {"takeover": 0, "stop": 0, "watchdog_handback": 0})
            if row["event"] in d:
                d[row["event"]] += 1
    return out


def permutation_p(a, b, n=10000, seed=0):
    """Two-sided p-value for the difference in means of a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    observed = abs(a.mean() - b.mean())
    both = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        rng.shuffle(both)
        hits += abs(both[:len(a)].mean() - both[len(a):].mean()) >= observed - 1e-12
    return (hits + 1) / (n + 1)


def summarise(key_rows, score_rows, inter):
    arm_of = {int(r["flight"]): r["arm"] for r in key_rows}
    scores = {int(r["flight"]): r for r in score_rows}
    result = {}
    for arm in ("A", "B"):
        flights = sorted(f for f, a in arm_of.items() if a == arm)
        scored = [scores[f] for f in flights if f in scores]
        dist = [float(r["distance_cm"]) for r in scored]
        result[arm] = {
            "flights": len(flights), "scored": len(scored),
            "on_pad": sum(int(r["on_pad"]) for r in scored),
            "mean_cm": round(statistics.mean(dist), 1) if dist else None,
            "median_cm": round(statistics.median(dist), 1) if dist else None,
            "takeovers": sum(inter.get(f, {}).get("takeover", 0) for f in flights),
            "stops": sum(inter.get(f, {}).get("stop", 0) for f in flights),
            "watchdog": sum(inter.get(f, {}).get("watchdog_handback", 0) for f in flights),
            "distances": dist,
        }
    result["p_value"] = permutation_p(result["A"]["distances"], result["B"]["distances"])
    return result


def main():
    ap = argparse.ArgumentParser(description="unblind the A/B checkpoint test")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--key", help="default <log-dir>/ab_key.csv")
    ap.add_argument("--scores", help="default <log-dir>/ab_scores.csv")
    args = ap.parse_args()
    key = read_csv(args.key or os.path.join(args.log_dir, "ab_key.csv"))
    scores_path = args.scores or os.path.join(args.log_dir, "ab_scores.csv")
    scores = read_csv(scores_path) if os.path.exists(scores_path) else []
    r = summarise(key, scores, interventions(args.log_dir))
    print("model  flights  scored  on pad   mean cm  median cm  takeovers  STOPs  watchdog")
    for arm in ("A", "B"):
        x = r[arm]
        print(f"  {arm}    {x['flights']:5d}   {x['scored']:5d}   {x['on_pad']:3d}/{x['scored']:<3d}"
              f"  {x['mean_cm'] if x['mean_cm'] is not None else '-':>7}  {x['median_cm'] if x['median_cm'] is not None else '-':>9}"
              f"  {x['takeovers']:9d}  {x['stops']:5d}  {x['watchdog']:8d}")
    print(f"\npermutation test on mean distance: p = {r['p_value']:.3f}"
          + ("  (probably a real difference)" if r["p_value"] < 0.05 else "  (could be chance; fly more pairs)"))
    print("A = --policy, B = --policy-b of the bridge.")


if __name__ == "__main__":
    main()
