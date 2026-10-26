"""eval_val.py <checkpoint folder> -- compares the trained model's actions with the pilot's on the validation episodes."""
import json, os, sys
import numpy as np
import torch
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ckpt = os.path.expanduser(sys.argv[1])
processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
stats = json.load(open(os.path.join(ckpt, "dataset_statistics.json")))
vla.norm_stats["drone_dataset"] = stats["drone_dataset"]

b = tfds.builder_from_directory(os.path.expanduser("~/tensorflow_datasets/drone_dataset/1.0.0"))
P, T = [], []
for i, ep in enumerate(b.as_dataset(split="val")):
    for st in ep["steps"]:
        img = Image.fromarray(st["observation"]["image"].numpy())
        lang = st["language_instruction"].numpy().decode()
        prompt = f"In: What action should the robot take to {lang.lower()}?\nOut:"
        inp = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            a = vla.predict_action(**inp, unnorm_key="drone_dataset", do_sample=False)
        P.append(np.asarray(a, dtype=float)); T.append(st["action"].numpy().astype(float))
    print("episode", i + 1, "done, steps so far", len(P), flush=True)

P, T = np.array(P), np.array(T)
names = ["pitch", "roll", "throttle", "yaw"]
train_mean = np.array([stats["drone_dataset"]["action"]["mean"]])[0]
print("\n%d validation steps" % len(P))
print("%-9s %8s %8s %8s %6s" % ("stick", "MAE", "MAE_mean", "better%", "corr"))
for k, n in enumerate(names):
    mae = np.abs(P[:, k] - T[:, k]).mean()
    base = np.abs(train_mean[k] - T[:, k]).mean()
    corr = np.corrcoef(P[:, k], T[:, k])[0, 1] if P[:, k].std() > 0 and T[:, k].std() > 0 else float("nan")
    print("%-9s %8.3f %8.3f %7.0f%% %6.2f" % (n, mae, base, 100 * (1 - mae / base), corr))
np.save(os.path.expanduser("~/val_pred.npy"), P); np.save(os.path.expanduser("~/val_true.npy"), T)
print("MAE = mean absolute error of the stick value (-1..1); MAE_mean = error if the model always output the average stick; corr = correlation with the pilot.")
