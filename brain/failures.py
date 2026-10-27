"""
failures.py  -  the failure library (idea L6): failed model flights, grouped by what went wrong.

    python3 -m brain.failures logs/

Reads every bridge session in logs/ (steps.csv, events.csv) plus, when present, your
thumbs up / down from the phone (L4) and the AI judge's verdicts (logs/judge.csv, X13).
A flight counts as failed when you gave it a thumbs down, the judge says it missed the pad,
or it ended with a takeover, STOP or lost link. Each failure gets ONE cause, first match wins:

    judge          the AI judge's category, when it named one
    unfamiliar     the scene detector (X12) flagged the room during the flight
    link           watchdog: commands stopped arriving
    unsure         the model was unsure (hover instead of obey) in > 20 % of the steps
    pad_lost       the pad was in view, then out of view for the last 2 s before the end
    overshoot      the pad left the picture at the bottom while the model still pitched forward
    drift          the pad stayed far to one side (|x| > 0.5) for more than 2 s
    wrong_yaw      the model asked for a turn (|yaw| > 20 %) in > 20 % of the steps
    other          none of the above (your spoken reason, if any, is listed)

Output: a table of causes with counts and the flights in each, and the advice which
episodes to record next (the biggest group first), so new data targets real weaknesses.
"""
import argparse, csv, glob, json, os
from collections import defaultdict


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def flight_of(row):
    try:
        d = json.loads(row.get("detail") or "{}")
        f = d.get("flight") if isinstance(d, dict) else None
        return int(f) if f not in (None, "") else None
    except (ValueError, TypeError):
        return None


def read_judge(log_dir):
    path = os.path.join(log_dir, "judge.csv")
    if not os.path.exists(path):
        return {}
    return {r["flight"]: r for r in csv.DictReader(open(path))}


def classify(steps, events, judge=None):
    """(failed, cause, note) for one flight. steps: rows of steps.csv; events: names + details."""
    names = [e["event"] for e in events]
    feedback = [e for e in events if e["event"] == "feedback"]
    thumbs = feedback[-1]["text"] if feedback else ""
    reasons = [e["text"] for e in events if e["event"] == "feedback_reason"]
    judged_miss = judge is not None and judge.get("on_pad") == "0"
    failed = (thumbs == "down" or judged_miss
              or any(n in names for n in ("takeover", "stop", "watchdog_handback")))
    if thumbs == "up" and not judged_miss:
        failed = False
    if not failed:
        return False, "", ""
    note = "; ".join(reasons)
    if judge and judge.get("category") not in (None, "", "unknown", "landed_on_pad"):
        return True, judge["category"], note or judge.get("reason", "")
    n = max(len(steps), 1)
    if any(n_ == "ood_on" for n_ in names):
        return True, "unfamiliar", note
    if "watchdog_handback" in names:
        return True, "link", note
    if sum(r.get("unsure") == "1" for r in steps) / n > 0.2:
        return True, "unsure", note
    vis = [r.get("pad_visible") == "1" for r in steps]
    if any(vis) and len(vis) >= 20 and not any(vis[-20:]):
        last = max(i for i, v in enumerate(vis) if v)
        if num(steps[last].get("pad_y")) > 0.7 and num(steps[last].get("sent_pitch")) > 5:
            return True, "overshoot", note
        return True, "pad_lost", note
    if sum(1 for r in steps if r.get("pad_visible") == "1" and abs(num(r.get("pad_x"))) > 0.5) > 20:
        return True, "drift", note
    if sum(abs(num(r.get("model_yaw"))) > 20 for r in steps) / n > 0.2:
        return True, "wrong_yaw", note
    return True, "other", note


ADVICE = {
    "pad_lost": "record episodes where the pad starts near the edge of the view and is re-found",
    "overshoot": "record slower final approaches that stop right above the pad",
    "drift": "record episodes with sideways corrections (pad left or right of the path)",
    "wrong_yaw": "keep yaw off, or record episodes that need a turn",
    "unfamiliar": "record a few episodes in this room / light, or fly where training was recorded",
    "unsure": "record more of the situations where the model hesitated (see steps.csv)",
    "link": "check Wi-Fi and the inference server speed (rehearsal --bridge)",
    "too_early_descent": "record episodes that hold height until the pad is under the drone",
}


def collect(log_dir):
    judge = read_judge(log_dir)
    results = []
    for sdir in sorted(glob.glob(os.path.join(log_dir, "session_*"))):
        try:
            steps = list(csv.DictReader(open(os.path.join(sdir, "steps.csv"))))
            events = list(csv.DictReader(open(os.path.join(sdir, "events.csv"))))
        except OSError:
            continue
        per_steps, per_events = defaultdict(list), defaultdict(list)
        for r in steps:
            if r.get("flight"):
                per_steps[int(r["flight"])].append(r)
        for e in events:
            f = flight_of(e)
            if f is not None:
                d = json.loads(e["detail"])
                per_events[f].append({"event": e["event"], "text": d.get("detail", "")})
        for f in sorted(set(per_steps) | set(per_events)):
            failed, cause, note = classify(per_steps[f], per_events[f], judge.get(str(f)))
            results.append({"session": os.path.basename(sdir), "flight": f, "failed": failed,
                            "cause": cause, "note": note})
    return results


def main():
    ap = argparse.ArgumentParser(description="failed model flights grouped by cause")
    ap.add_argument("log_dir", nargs="?", default="logs")
    args = ap.parse_args()
    res = collect(args.log_dir)
    fails = [r for r in res if r["failed"]]
    print(f"{len(res)} model flights, {len(fails)} failed")
    groups = defaultdict(list)
    for r in fails:
        groups[r["cause"]].append(r)
    for cause, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        where = ", ".join(f"{r['session'][8:]}#{r['flight']}" for r in rows[:8])
        print(f"  {cause:18s} {len(rows):3d}   {where}{' ...' if len(rows) > 8 else ''}")
        for r in rows:
            if r["note"]:
                print(f"      {r['session'][8:]}#{r['flight']}: {r['note']}")
    if groups:
        top = max(groups, key=lambda c: len(groups[c]))
        print(f"\nnext recordings: {ADVICE.get(top, 'look at the flights listed above')}")


if __name__ == "__main__":
    main()
