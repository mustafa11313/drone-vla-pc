"""
inference_server.py  -  runs the fine-tuned OpenVLA model behind a small HTTP API.

    POST /act      body = one JPEG frame (raw bytes)
                   optional query ?instruction=...   (default: the training instruction)
                   answer: {"action": [pitch, roll, throttle, yaw],    # -1..1, same as the dataset
                            "model_ms": 182.4, "step": 17}
    GET  /         health check: model folder, device, mode, steps served

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
import argparse, io, json, os, threading, time

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
        img = prepare_image(jpeg_bytes, self.center_crop)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inp = self.processor(prompt, img).to("cuda:0", dtype=self.torch.bfloat16)
        with self.torch.no_grad():
            a = self.vla.predict_action(**inp, unnorm_key="drone_dataset", do_sample=False)
        return [float(x) for x in np.asarray(a, dtype=float)]


class MockPolicy:
    """No model: a gentle forward-and-descend command, with a fake model time. For tests only."""

    def __init__(self, delay_ms=150):
        self.delay = delay_ms / 1000.0
        self.name = "mock"

    def act(self, jpeg_bytes, instruction):
        prepare_image(jpeg_bytes)          # still checks that the frame is a valid JPEG
        time.sleep(self.delay)
        return [0.10, 0.0, -0.20, 0.0]     # pitch, roll, throttle, yaw (-1..1)


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
            action = policy.act(data, instruction)
            model_ms = (time.perf_counter() - t0) * 1000
            served += 1
            step = served
    except Exception as e:
        return jsonify({"error": f"model failed: {e}"}), 500
    return jsonify({"action": action, "model_ms": round(model_ms, 1), "step": step})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "inference server running", "model": policy.name, "steps_served": served})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="checkpoint folder (with dataset_statistics.json)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true", help="no model, fixed command (for tests)")
    ap.add_argument("--mock-delay-ms", type=float, default=150)
    ap.add_argument("--no-center-crop", action="store_true", help="feed the plain 224 x 224 frame")
    args = ap.parse_args()

    global policy
    if args.mock:
        policy = MockPolicy(args.mock_delay_ms)
    elif args.checkpoint:
        policy = OpenVLAPolicy(args.checkpoint, center_crop=not args.no_center_crop)
    else:
        ap.error("give a checkpoint folder or --mock")
    print(f"[SERVER] model: {policy.name}, listening on port {args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
