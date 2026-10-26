"""
cnn_baseline.py  -  a small image -> 4 sticks network, the yardstick for OpenVLA (idea X15).

The thesis question this answers: does a 7-billion-parameter VLA beat a plain 11-million-
parameter CNN trained on the same frames and the same split? Either answer is a result.

    # train (GPU node, env openvla: it has torch + torchvision; ~10 min for 50 episodes)
    python3 cnn_baseline.py train ~/drone_database/episodes --out ~/runs/cnn
    python3 cnn_baseline.py train ~/drone_database/episodes --out ~/runs/cnn --recovery-views 2

    # the same validation table as eval_val.py, for this network
    python3 cnn_baseline.py eval ~/drone_database/episodes ~/runs/cnn/best.pt

    # serve it exactly like OpenVLA (the bridge and the app do not see a difference)
    python3 inference_server.py --cnn ~/runs/cnn/best.pt --port 8000

Data: the .h5 files of csv_to_hdf5.py and the SAME train / val split as the dataset builder
(drone_dataset_dataset_builder.split_jobs), so the numbers compare 1:1 with eval_val.py.
Network: ResNet-18 (ImageNet weights when they can be downloaded, else from scratch), last
layer -> 4 values, tanh -> -1..1. Loss: L1 (= the MAE that is reported).
Images: 224 x 224; training uses random 90 % crops (like OpenVLA's --image_aug), evaluation
and serving use the centre 90 % crop, so all three see the same thing.
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import drone_dataset_dataset_builder as builder  # noqa: E402

NAMES = ["pitch", "roll", "throttle", "yaw"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)    # ImageNet
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_AREA = 0.9


def to_tensor(img):
    """uint8 HxWx3 -> normalised float tensor 3xHxW."""
    import torch
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def crop(img, rng=None):
    """90 % area crop, back to 224: random position when rng is given, else the centre."""
    from PIL import Image
    h, w = img.shape[:2]
    side = int(round(h * CROP_AREA ** 0.5))
    if rng is None:
        top = left = (h - side) // 2
    else:
        top, left = rng.integers(0, h - side + 1), rng.integers(0, w - side + 1)
    out = Image.fromarray(img[top:top + side, left:left + side]).resize((w, h), Image.BILINEAR)
    return np.asarray(out)


def load_split(h5_dir, recovery_views=0):
    """(train images, train actions, val images, val actions); images uint8 N x 224 x 224 x 3."""
    train_jobs, val_jobs = builder.split_jobs(builder.find_h5_files(h5_dir), recovery_views)

    def load(jobs):
        imgs, acts = [], []
        for path, shift, _, _ in jobs:
            steps, _, _ = builder.load_episode(path, shift=shift)
            imgs += [s["observation"]["image"] for s in steps]
            acts += [s["action"] for s in steps]
        return np.stack(imgs), np.stack(acts).astype(np.float32)

    tx, ty = load(train_jobs)
    vx, vy = load(val_jobs) if val_jobs else (tx[:0], ty[:0])
    print(f"train {len(train_jobs)} episodes / {len(tx)} steps, val {len(val_jobs)} episodes / {len(vx)} steps")
    return tx, ty, vx, vy


def build_model(pretrained=True):
    import torch.nn as nn
    import torchvision
    net = None
    if pretrained:
        try:
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            print("ResNet-18 with ImageNet weights")
        except Exception as e:                      # no internet on the node: train from scratch
            print(f"ImageNet weights not available ({e.__class__.__name__}), training from scratch")
    if net is None:
        net = torchvision.models.resnet18(weights=None)
    net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.fc.in_features, 4), nn.Tanh())
    return net


def load_model(path, device="cpu"):
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = build_model(pretrained=False)
    net.load_state_dict(ckpt["model"])
    return net.to(device).eval(), ckpt.get("meta", {})


def predict(net, imgs, device, batch=64):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = torch.stack([to_tensor(crop(im)) for im in imgs[i:i + batch]]).to(device)
            out.append(net(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)


def report(pred, true, train_mean):
    """Same table and meaning as eval_val.py (values -1..1)."""
    print(f"\n{len(pred)} validation steps")
    print("%-9s %8s %8s %8s %6s" % ("stick", "MAE", "MAE_mean", "better%", "corr"))
    rows = {}
    for k, n in enumerate(NAMES):
        mae = float(np.abs(pred[:, k] - true[:, k]).mean())
        base = float(np.abs(train_mean[k] - true[:, k]).mean())
        corr = (float(np.corrcoef(pred[:, k], true[:, k])[0, 1])
                if pred[:, k].std() > 0 and true[:, k].std() > 0 else float("nan"))
        better = 100 * (1 - mae / base) if base > 0 else float("nan")
        print("%-9s %8.3f %8.3f %7.0f%% %6.2f" % (n, mae, base, better, corr))
        rows[n] = {"mae": round(mae, 4), "mae_mean": round(base, 4), "corr": round(corr, 3)}
    return rows


def train(args):
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tx, ty, vx, vy = load_split(args.h5_dir, args.recovery_views)
    net = build_model(not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    train_mean = ty.mean(0)
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(len(tx))
        t0, losses = time.time(), []
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            x = torch.stack([to_tensor(crop(tx[j], rng)) for j in idx]).to(device)
            y = torch.from_numpy(ty[idx]).to(device)
            loss = torch.nn.functional.l1_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        net.eval()
        val = float(np.abs(predict(net, vx, device) - vy).mean()) if len(vx) else float("nan")
        history.append({"epoch": epoch, "train_l1": round(float(np.mean(losses)), 4), "val_mae": round(val, 4)})
        print(f"epoch {epoch:3d}  train L1 {np.mean(losses):.4f}  val MAE {val:.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
        meta = {"epoch": epoch, "val_mae": val, "train_mean": train_mean.tolist(),
                "recovery_views": args.recovery_views, "h5_dir": os.path.abspath(args.h5_dir)}
        torch.save({"model": net.state_dict(), "meta": meta}, os.path.join(args.out, "last.pt"))
        if len(vx) == 0 or val < best:
            best = val
            torch.save({"model": net.state_dict(), "meta": meta}, os.path.join(args.out, "best.pt"))
    json.dump(history, open(os.path.join(args.out, "history.json"), "w"), indent=1)
    net, _ = load_model(os.path.join(args.out, "best.pt"), device)
    if len(vx):
        rows = report(predict(net, vx, device), vy, train_mean)
        json.dump(rows, open(os.path.join(args.out, "val_report.json"), "w"), indent=1)
    print(f"\nsaved {args.out}/best.pt (lowest val MAE {best:.4f})")


def evaluate(args):
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net, meta = load_model(args.checkpoint, device)
    _, ty, vx, vy = load_split(args.h5_dir)
    report(predict(net, vx, device), vy, np.array(meta.get("train_mean", ty.mean(0))))


def main():
    ap = argparse.ArgumentParser(description="small CNN baseline for the drone landing task")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("h5_dir", help="folder with the episode .h5 files")
    t.add_argument("--out", default="runs/cnn")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--recovery-views", type=int, default=0, help="extra shifted views per episode (augment.py)")
    t.add_argument("--no-pretrained", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("h5_dir")
    e.add_argument("checkpoint")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
