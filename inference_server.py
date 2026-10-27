"""
inference_server.py  -  runs the fine-tuned OpenVLA model behind a small HTTP API.

    POST /act      body = one JPEG frame (raw bytes)
                   optional query ?instruction=...   (default: the training instruction)
                   answer: {"action": [pitch, roll, throttle, yaw],    # -1..1, same as the dataset
                            "model_ms": 182.4, "step": 17,
                            "confidence": {"p": [4 values], "spread_pct": [4 values]} or null}
    GET  /         health check: model folder, device, mode, steps served

Confidence (idea M1, OpenVLA only): each stick value is ONE token picked from 256 bins. The
model's probabilities over those bins say how sure it is:
    p           probability of the bin it picked (1.0 = certain)
    spread_pct  standard deviation of the stick value over all bins, in stick percent
                (small = sure; large = it hesitates between very different commands)
The bridge logs both and can hover instead of obeying an unsure model (--max-spread).

Other models:
    python3 inference_server.py --cnn runs/cnn/best.pt       # the small CNN of cnn_baseline.py

Run on a GPU node (env openvla):

    python3 inference_server.py ~/runs/model_2000steps --port 8000

Run WITHOUT a GPU or model (for testing the PC brain and the app):

    python3 inference_server.py --mock --port 8000

The frame is prepared exactly like drone_dataset_dataset_builder.py does it
(RGB, resized to 224 x 224, bilinear). Training used --image_aug True, which
crops images randomly; OpenVLA's own evaluation code then centre-crops at test
time (90 % of the area). --center-crop does the same and is on by default.
eval_val.py does NOT crop, so its numbers may be slightly pessimistic.
"""
import argparse, hashlib, io, json, os, threading, time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

INSTRUCTION = "Take off, locate the red landing pad, and land on it."
IMAGE_SIZE = 224
CROP_AREA = 0.9          # OpenVLA's centre crop for models trained with image augmentation

app = Flask(__name__)
policy = None            # set in main()
lock = threading.Lock()  # one GPU, one request at a time
served = 0


def prepare_image(jpeg_bytes, center_crop=True):
    """JPEG bytes -> 224 x 224 RGB PIL image, same as the dataset builder (+ optional centre crop)."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    if center_crop:
        side = IMAGE_SIZE * CROP_AREA ** 0.5
        off = (IMAGE_SIZE - side) / 2
        img = img.crop((off, off, off + side, off + side)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return img


class OpenVLAPolicy:
    """The fine-tuned model. Loading takes about a minute and ~15 GB of GPU memory (bf16)."""

    def __init__(self, ckpt, center_crop=True):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.torch = torch
        self.ckpt = os.path.expanduser(ckpt)
        self.center_crop = center_crop
        self.processor = AutoProcessor.from_pretrained(self.ckpt, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
        stats = json.load(open(os.path.join(self.ckpt, "dataset_statistics.json")))
        self.vla.norm_stats["drone_dataset"] = stats["drone_dataset"]
        self.name = self.ckpt

    def act(self, jpeg_bytes, instruction):
        """(action -1..1, confidence dict). The action is exactly predict_action's answer; the
        token scores are only read on the side (generate is wrapped, its result unchanged)."""
        img = prepare_image(jpeg_bytes, self.center_crop)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inp = self.processor(prompt, img).to("cuda:0", dtype=self.torch.bfloat16)
        captured = {}
        generate = type(self.vla).generate.__get__(self.vla)

        def generate_with_scores(*a, **k):
            try:
                out = generate(*a, **k, output_scores=True, return_dict_in_generate=True)
            except Exception:                   # scores not supported: plain answer, no confidence
                return generate(*a, **k)
            captured["scores"] = out.scores
            return out.sequences

        self.vla.generate = generate_with_scores
        try:
            with self.torch.no_grad():
                a = self.vla.predict_action(**inp, unnorm_key="drone_dataset", do_sample=False)
        finally:
            del self.vla.generate
        conf = None
        if "scores" in captured:
            stats = self.vla.norm_stats["drone_dataset"]["action"]
            logits = np.stack([s[0].float().cpu().numpy() for s in captured["scores"]])
            conf = token_confidence(logits, self.vla.vocab_size, np.asarray(self.vla.bin_centers),
                                    np.asarray(stats["q01"]), np.asarray(stats["q99"]))
        return [float(x) for x in np.asarray(a, dtype=float)], conf


def token_confidence(logits, vocab_size, bin_centers, q01, q99):
    """logits: (n_sticks, vocab) scores of the generated action tokens.
    Same token -> value mapping as OpenVLA's predict_action. Returns
    {"p": [prob of the chosen bin], "spread_pct": [std of the stick value, percent]}."""
    n_bins = len(bin_centers)
    token_ids = np.arange(vocab_size - 256, vocab_size)
    bins = np.clip(vocab_size - token_ids - 1, 0, n_bins - 1)
    p_out, spread = [], []
    for k, row in enumerate(logits):
        z = row.astype(np.float64)
        prob = np.exp(z - z.max())
        prob = prob[token_ids] / prob.sum()
        prob = prob / max(prob.sum(), 1e-12)                  # only the action tokens
        values = 0.5 * (bin_centers[bins] + 1) * (q99[k] - q01[k]) + q01[k]
        mean = float((prob * values).sum())
        p_out.append(round(float(prob.max()), 3))
        spread.append(round(100 * float(np.sqrt((prob * (values - mean) ** 2).sum())), 1))
    return {"p": p_out, "spread_pct": spread}


class CNNPolicy:
    """The small baseline network of cnn_baseline.py (idea X15). Runs on CPU or GPU, ~5 ms."""

    def __init__(self, path):
        import torch
        from cnn_baseline import load_model
        self.torch = torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.meta = load_model(path, self.device)
        self.name = f"cnn:{path}"

    def act(self, jpeg_bytes, instruction):
        from cnn_baseline import to_tensor
        img = prepare_image(jpeg_bytes, center_crop=True)
        with self.torch.no_grad():
            a = self.model(to_tensor(np.asarray(img))[None].to(self.device))[0].cpu().numpy()
        return [float(x) for x in a], None


class MockPolicy:
    """No model: a gentle forward-and-descend command, with a fake model time. For tests only."""

    def __init__(self, delay_ms=150, seek=False):
        self.delay = delay_ms / 1000.0
        self.seek = seek
        self.name = "mock-seek" if seek else "mock"

    def act(self, jpeg_bytes, instruction):
        img = prepare_image(jpeg_bytes)    # still checks that the frame is a valid JPEG
        time.sleep(self.delay)
        conf = {"p": [0.9, 0.9, 0.9, 0.9], "spread_pct": [3.0, 3.0, 3.0, 0.0]}
        if not self.seek:
            return [0.10, 0.0, -0.20, 0.0], conf     # pitch, roll, throttle, yaw (-1..1)
        # --mock-seek: a hand-made "model" that flies toward the red pixels (tests of the tools)
        a = np.asarray(img).astype(int)
        red = (a[..., 0] >= 120) & (a[..., 0] - a[..., 1] >= 50) & (a[..., 0] - a[..., 2] >= 50)
        if red.mean() < 0.002:
            return [0.15, 0.0, 0.0, 0.0], conf      # nothing red: creep forward
        ys, xs = np.nonzero(red)
        x, y = xs.mean() / a.shape[1] * 2 - 1, ys.mean() / a.shape[0] * 2 - 1
        return [round(float(-0.3 * y), 3), round(float(0.3 * x), 3), -0.2 if abs(x) + abs(y) < 0.4 else 0.0, 0.0], conf


@app.route("/act", methods=["POST"])
def act():
    global served
    data = request.get_data()
    if not data:
        return jsonify({"error": "empty body, send one JPEG frame"}), 400
    instruction = request.args.get("instruction", INSTRUCTION)
    try:
        with lock:
            t0 = time.perf_counter()
            action, confidence = policy.act(data, instruction)
            model_ms = (time.perf_counter() - t0) * 1000
            served += 1
            step = served
    except Exception as e:
        return jsonify({"error": f"model failed: {e}"}), 500
    return jsonify({"action": action, "model_ms": round(model_ms, 1), "step": step,
                    "confidence": confidence})


@app.route("/", methods=["GET"])
def home():
    info = {"status": "inference server running", "model": policy.name, "steps_served": served}
    ckpt = getattr(policy, "ckpt", None)
    if ckpt:          # which checkpoint and dataset produced a flight (idea X16, brain/tracking.py)
        stats = os.path.join(ckpt, "dataset_statistics.json")
        if os.path.exists(stats):
            info["dataset_stats_sha256"] = hashlib.sha256(open(stats, "rb").read()).hexdigest()[:16]
        info["center_crop"] = getattr(policy, "center_crop", None)
    return jsonify(info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="checkpoint folder (with dataset_statistics.json)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true", help="no model, fixed command (for tests)")
    ap.add_argument("--cnn", help="serve the small CNN baseline (.pt from cnn_baseline.py) instead")
    ap.add_argument("--mock-delay-ms", type=float, default=150)
    ap.add_argument("--mock-seek", action="store_true", help="mock steers toward red pixels instead of a fixed command")
    ap.add_argument("--no-center-crop", action="store_true", help="feed the plain 224 x 224 frame")
    args = ap.parse_args()

    global policy
    if args.mock:
        policy = MockPolicy(args.mock_delay_ms, seek=args.mock_seek)
    elif args.cnn:
        policy = CNNPolicy(args.cnn)
    elif args.checkpoint:
        policy = OpenVLAPolicy(args.checkpoint, center_crop=not args.no_center_crop)
    else:
        ap.error("give a checkpoint folder or --mock")
    print(f"[SERVER] model: {policy.name}, listening on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
