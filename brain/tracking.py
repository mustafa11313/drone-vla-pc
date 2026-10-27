"""
tracking.py  -  every flight can be traced back to exactly what produced it (idea X16).

The bridge writes logs/session_<time>/manifest.json when a session starts, and adds the
phone's details when the app says hello:

    pc_code        git commit of this repo (+ "dirty" if files were changed and not committed)
    model          what the inference server says it runs (checkpoint folder, crop, ...)
    dataset        sha256 of the checkpoint's dataset_statistics.json = which dataset it saw
    ood_reference  sha256 of the unfamiliar-scene reference, when used
    app            app version and build from the phone
    settings       every bridge option (mode, limits, scale, hybrid height, ...)

    python3 -m brain.tracking logs/        # one line per session: date, model, app, settings

So a thesis table row "checkpoint 2000 steps, 9/10 on the pad" can always be checked against
the session folders it came from.
"""
import argparse, glob, hashlib, json, os, subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    if not path or not os.path.exists(os.path.expanduser(path)):
        return None
    with open(os.path.expanduser(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def git_state():
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return {"commit": commit or None, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def server_info(url):
    if not url or not url.startswith("http"):
        return {"name": url}
    try:
        import requests
        return requests.get(url.rstrip("/") + "/", timeout=3).json()
    except Exception as e:
        return {"url": url, "error": str(e)}


def write_manifest(session_dir, args, extra=None):
    m = {"created": datetime.now().isoformat(timespec="seconds"),
         "pc_code": git_state(),
         "model": server_info(args.policy),
         "model_b": server_info(args.policy_b) if getattr(args, "policy_b", None) else None,
         "dataset": {"stats_file": args.stats, "stats_sha256": sha256(args.stats)},
         "ood_reference": {"file": getattr(args, "ood", None), "sha256": sha256(getattr(args, "ood", None))},
         "settings": {k: v for k, v in vars(args).items()},
         "app": None}
    m.update(extra or {})
    path = os.path.join(session_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(m, f, indent=1, default=str)
    return m


def add_to_manifest(session_dir, key, value):
    path = os.path.join(session_dir, "manifest.json")
    try:
        m = json.load(open(path))
    except (OSError, ValueError):
        m = {}
    m[key] = value
    with open(path, "w") as f:
        json.dump(m, f, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description="list sessions with what produced them")
    ap.add_argument("log_dir", nargs="?", default="logs")
    args = ap.parse_args()
    for path in sorted(glob.glob(os.path.join(args.log_dir, "session_*", "manifest.json"))):
        m = json.load(open(path))
        model = m.get("model") or {}
        app = m.get("app") or {}
        s = m.get("settings") or {}
        code = m.get("pc_code") or {}
        print(f"{os.path.basename(os.path.dirname(path))}  model={model.get('model') or model.get('name')}  "
              f"data={(m.get('dataset') or {}).get('stats_sha256')}  app={app.get('version')}  "
              f"pc={code.get('commit')}{'+dirty' if code.get('dirty') else ''}  mode={s.get('mode')} "
              f"scale={s.get('scale')} hybrid={s.get('hybrid')} ood={bool(s.get('ood'))}")


if __name__ == "__main__":
    main()
