"""
whatif.py  -  the what-if lab: change the picture or the words, watch the command (idea M4).

    python3 -m brain.whatif --policy http://<gpu-host>:8000 episode_0046_* episode_0047_*
    python3 -m brain.whatif --policy http://<gpu-host>:8000 episode_0046_* --every 2 --csv whatif.csv

For every 5th flown frame the model is asked again with one thing changed, and the answer is
compared with its answer to the untouched frame. Works with any server (OpenVLA, CNN, mock).

    mirror       picture flipped left-right. A model that understood the scene answers with
                 the same pitch and the opposite roll and yaw. Reported: roll sign agreement.
    no_pad       red pad painted over with the floor colour. If the commands barely change,
                 the model is not looking at the pad (it learned the room, not the task).
    shift_left / shift_right
                 view moved 5 % sideways (like the recovery views of augment.py). Reported:
                 roll change per 1 % shift; augment.py assumes about -2 % roll per 1 % shift.
    dark / bright
                 brightness x0.6 / x1.4. Commands should hardly change.
    new_words    a held-out instruction (paraphrases_test.txt). Should hardly change.
    wrong_task   "fly away from the red landing pad". If nothing changes, the words are ignored.

Output: one line per variant, mean absolute change per stick in stick percent, plus the
special check of that variant. --csv writes every single answer for plots.
"""
import argparse, csv, io, os, statistics

import numpy as np
from PIL import Image, ImageEnhance

from brain.landing_score import red_mask
from brain.policy_client import PolicyClient
from brain.rehearsal import load_episode

STICKS = ["pitch", "roll", "throttle", "yaw"]
SHIFT = 0.05
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jpeg_of(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def shifted(img, shift):
    """Same 90 % window as augment.shifted_image, kept at full size."""
    w, h = img.size
    cw, ch = w * 0.9, h * 0.9
    left = min(max((w - cw) / 2 + shift * w, 0), w - cw)
    top = (h - ch) / 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def without_pad(img):
    a = np.asarray(img).copy()
    m = red_mask(img)
    if m.any():
        grown = m.copy()                    # grow the mask a little to cover the pad's edge
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            grown |= np.roll(np.roll(m, dy, 0), dx, 1)
        a[grown] = np.median(a[~grown], axis=0).astype(np.uint8)
    return Image.fromarray(a)


def image_variants(img):
    return {
        "mirror": img.transpose(Image.FLIP_LEFT_RIGHT),
        "no_pad": without_pad(img),
        "shift_left": shifted(img, -SHIFT),
        "shift_right": shifted(img, SHIFT),
        "dark": ImageEnhance.Brightness(img).enhance(0.6),
        "bright": ImageEnhance.Brightness(img).enhance(1.4),
    }


def held_out_instruction():
    path = os.path.join(HERE, "paraphrases_test.txt")
    try:
        lines = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        return lines[0]
    except OSError:
        return "Please put the drone down on the red pad."


def run(policy_url, episodes, every, csv_path=None):
    base = PolicyClient(policy_url, timeout_s=10)
    words = {"new_words": PolicyClient(policy_url, timeout_s=10, instruction=held_out_instruction()),
             "wrong_task": PolicyClient(policy_url, timeout_s=10, instruction="Fly away from the red landing pad.")}
    rows = []
    for ep in episodes:
        for i, _, jpeg in load_episode(ep)[::every]:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            ref, _, _ = base.act(jpeg)
            rows.append({"episode": os.path.basename(os.path.normpath(ep)), "row": i, "variant": "original",
                         **dict(zip(STICKS, ref)), "ref": ref})
            asks = [(name, base, jpeg_of(v)) for name, v in image_variants(img).items()]
            asks += [(name, client, jpeg) for name, client in words.items()]
            for name, client, data in asks:
                out, _, _ = client.act(data)
                rows.append({"episode": rows[-1]["episode"], "row": i, "variant": name,
                             **dict(zip(STICKS, out)), "ref": ref})
        print(f"{os.path.basename(os.path.normpath(ep))}: done", flush=True)
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["episode", "row", "variant", *STICKS])
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(r[k], 2) if k in STICKS else r[k]) for k in w.fieldnames})
    return rows


def report(rows):
    variants = [v for v in dict.fromkeys(r["variant"] for r in rows) if v != "original"]
    print(f"\n{sum(r['variant'] == 'original' for r in rows)} frames. Mean |change| vs. the untouched frame, stick %:")
    print("variant        pitch   roll  throttle   yaw   check")
    for v in variants:
        sel = [r for r in rows if r["variant"] == v]
        if v == "mirror":     # expected: pitch, throttle same; roll, yaw mirrored
            diff = [[abs(r["pitch"] - r["ref"][0]), abs(r["roll"] + r["ref"][1]),
                     abs(r["throttle"] - r["ref"][2]), abs(r["yaw"] + r["ref"][3])] for r in sel]
            turned = [r for r in sel if abs(r["ref"][1]) >= 5]
            agree = sum(np.sign(r["roll"]) == -np.sign(r["ref"][1]) for r in turned)
            check = f"roll flips sign in {agree}/{len(turned)} frames with |roll| >= 5 %  (vs. mirrored answer)"
        else:
            diff = [[abs(r[s] - r["ref"][k]) for k, s in enumerate(STICKS)] for r in sel]
            check = ""
            if v == "shift_right":
                left = {(r["episode"], r["row"]): r["roll"] for r in rows if r["variant"] == "shift_left"}
                slope = [(r["roll"] - left[(r["episode"], r["row"])]) / (2 * SHIFT * 100) for r in sel
                         if (r["episode"], r["row"]) in left]
                check = f"roll change per 1 % shift: {statistics.mean(slope):+.2f} % (augment.py assumes -2.00)"
            elif v == "no_pad":
                check = "large = the model looks at the pad (good)"
            elif v == "wrong_task":
                check = "near 0 = the words are ignored"
        m = np.mean(diff, axis=0)
        print(f"{v:12s} {m[0]:6.1f} {m[1]:6.1f} {m[2]:8.1f} {m[3]:6.1f}   {check}")


def main():
    ap = argparse.ArgumentParser(description="what-if lab: perturb frames / words, compare commands")
    ap.add_argument("episodes", nargs="+", help="episode folders (telemetry CSV + frames/)")
    ap.add_argument("--policy", required=True, help="inference server URL")
    ap.add_argument("--every", type=int, default=5, help="use every n-th flown frame")
    ap.add_argument("--csv", help="write all answers to this CSV")
    args = ap.parse_args()
    report(run(args.policy, args.episodes, args.every, args.csv))


if __name__ == "__main__":
    main()
