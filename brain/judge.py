"""
judge.py  -  an AI judge for every flight (idea X13).

Claude looks at the end of a flight and answers: did it land on the red pad, and if not,
what went wrong? You get a label and a failure reason for every flight without typing, to
check against the landing score (E1) and your thumbs up / down (L4), and to feed the failure
library (brain/failures.py).

Needs on the laptop:  pip install anthropic   and your API key in ANTHROPIC_API_KEY
(set it in your own terminal; it never goes into the repo or a log).

    # a bridge session (start the bridge with --save-frames); one line per model flight
    python3 -m brain.judge --session logs/session_20261101_143000

    # a recorded episode folder (telemetry CSV + frames/)
    python3 -m brain.judge --episode ~/drone_database/incoming/episode_0046_*

    # optional overhead photo after landing (the E1 camera) makes the verdict much safer
    python3 -m brain.judge --session logs/session_... --after after_flight7.jpg --flight 7

Results are appended to <log dir>/judge.csv: flight, on_pad, confidence, category, reason.
The camera looks 50 deg down, so the pad leaves the picture just before touchdown: the
judge is told this and says "unknown" rather than guess. An overhead photo removes that gap.
"""
import argparse, base64, csv, glob, io, json, os
from typing import Literal

from PIL import Image
from pydantic import BaseModel

MODEL = "claude-opus-5-5"
N_FRAMES = 8           # last frames of the flight that are shown

PROMPT = """You judge an autonomous drone landing experiment. The drone (DJI Mini 3) must land on a
red landing pad on the floor of an indoor lab. The images are the drone's own camera, which
points about 50 degrees down, in time order (0.1 s to 1 s apart), ending when the flight ended.
Because of the camera angle, the pad slides out of the bottom of the picture just before
touchdown even on a good landing. If an overhead photo is included, it is the most reliable
evidence of where the drone ended up.

Flight facts from the logs: {facts}

Decide whether the drone landed on the pad. If it clearly did not, pick the most likely failure
category: pad_lost (pad left the view and the drone never found it again), overshoot (flew past
the pad), drift (ended beside the pad), wrong_yaw (turned away), too_early_descent (came down
before reaching the pad), pilot_took_over (the logs say the pilot took over), or unknown.
Only say on_pad=true or false when the images support it; otherwise use null and category unknown."""


class Verdict(BaseModel):
    on_pad: bool | None
    confidence: Literal["low", "medium", "high"]
    category: Literal["landed_on_pad", "pad_lost", "overshoot", "drift", "wrong_yaw",
                      "too_early_descent", "pilot_took_over", "unknown"]
    reason: str


def image_block(path_or_bytes, width=640):
    data = open(path_or_bytes, "rb").read() if isinstance(path_or_bytes, str) else path_or_bytes
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def pick(frames, n=N_FRAMES):
    """The last n frames, denser near the end (the end decides the landing)."""
    if len(frames) <= n:
        return frames
    tail = frames[-int(len(frames) * 0.4):] if len(frames) > 3 * n else frames
    step = max(1, len(tail) // n)
    return tail[::-1][::step][:n][::-1]        # always includes the very last frame


def ask(frames, facts, after=None):
    import anthropic
    client = anthropic.Anthropic()
    content = [image_block(f) for f in frames]
    if after:
        content += [{"type": "text", "text": "Overhead photo after landing:"}, image_block(after)]
    content.append({"type": "text", "text": PROMPT.format(facts=json.dumps(facts))})
    r = client.beta.messages.parse(
        model=MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # another model answers if this one declines
        messages=[{"role": "user", "content": content}],
        output_format=Verdict)
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return Verdict(on_pad=None, confidence="low", category="unknown", reason=f"no verdict ({r.stop_reason})")
    return r.parsed_output


def event_flight(row):
    """Flight number stored in an events.csv detail, or None."""
    try:
        d = json.loads(row.get("detail") or "{}")
        return int(d["flight"]) if isinstance(d, dict) and d.get("flight") not in (None, "") else None
    except (ValueError, KeyError, TypeError):
        return None


def session_flights(session_dir):
    """{flight: (frames, facts)} from a bridge session saved with --save-frames."""
    steps = list(csv.DictReader(open(os.path.join(session_dir, "steps.csv"))))
    events = list(csv.DictReader(open(os.path.join(session_dir, "events.csv"))))
    flights = {}
    for row in steps:
        f = row.get("flight")
        if f:
            flights.setdefault(int(f), []).append(int(row["seq"]))
    out = {}
    for f, seqs in flights.items():
        frames = [p for p in (os.path.join(session_dir, "frames", f"{s:06d}.jpg") for s in seqs) if os.path.exists(p)]
        ends = [e["event"] for e in events if event_flight(e) == f
                and e["event"] in ("takeover", "stop", "watchdog_handback", "auto_landing", "landed")]
        out[f] = (frames, {"flight": f, "ended_by": ends[-1] if ends else "unknown", "frames": len(seqs)})
    return out


def write(path, flight, v):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["flight", "on_pad", "confidence", "category", "reason"])
        w.writerow([flight, "" if v.on_pad is None else int(v.on_pad), v.confidence, v.category, v.reason])


def main():
    ap = argparse.ArgumentParser(description="AI judge for landings")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="bridge session folder (saved with --save-frames)")
    g.add_argument("--episode", help="recorded episode folder")
    ap.add_argument("--flight", type=int, help="only this flight of the session")
    ap.add_argument("--after", help="overhead photo after landing (E1 camera)")
    ap.add_argument("--out", help="default: judge.csv next to the session / episode")
    args = ap.parse_args()
    if args.episode:
        frames = sorted(glob.glob(os.path.join(args.episode, "frames", "*.jpg")))
        name = os.path.basename(os.path.normpath(args.episode))
        jobs = {name: (frames, {"episode": name, "frames": len(frames), "ended_by": "recorded episode"})}
        out = args.out or os.path.join(os.path.dirname(os.path.normpath(args.episode)), "judge.csv")
    else:
        jobs = session_flights(args.session)
        if args.flight is not None:
            jobs = {args.flight: jobs[args.flight]}
        out = args.out or os.path.join(os.path.dirname(os.path.normpath(args.session)), "judge.csv")
    for flight, (frames, facts) in jobs.items():
        if not frames:
            print(f"flight {flight}: no saved frames (start the bridge with --save-frames)")
            continue
        v = ask(pick(frames), facts, args.after)
        write(out, flight, v)
        print(f"flight {flight}: on_pad={v.on_pad} ({v.confidence}) {v.category}: {v.reason}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
