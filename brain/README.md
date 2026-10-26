# PC brain: the safe loop and the Round 2 tools

```
phone app ──frames + telemetry (WebSocket, 10/s)──▶ bridge_server.py ──JPEG (HTTP)──▶ inference_server.py
          ◀──stick command (percent)───────────────                 ◀──4 values─────    (GPU, OpenVLA)
```

The phone stays the only thing that talks to the drone, and it keeps its own limits,
watchdog and STOP. The PC never flies the drone directly.

## Install (laptop)

    pip install -r brain/requirements.txt

On the cluster, the `openvla` environment only needs `pip install flask` extra.

## 1. Model server

    python3 inference_server.py ~/runs/model_2000steps --port 8000     # GPU node
    python3 inference_server.py --mock --port 8000                     # anywhere, no model

## 2. Ground rehearsal (no drone)

    # model only: MAE against the pilot + model time
    python3 -m brain.rehearsal --policy http://<server>:8000 <episode folders>

    # whole chain: the script pretends to be the phone
    python3 -m brain.bridge_server --policy http://<server>:8000 --stats <ckpt>/dataset_statistics.json
    python3 -m brain.rehearsal --bridge ws://127.0.0.1:8765 <episode folders>

The second run prints the delay from frame to command and the command rate. The data was
recorded at 10 Hz; if the loop is much slower, each command lasts longer than in training.

## 3. With the phone, shadow mode first

    python3 -m brain.bridge_server --policy http://<server>:8000 --stats <ckpt>/dataset_statistics.json

You fly; the phone shows the model's command next to your sticks; `logs/session_*/steps.csv`
records both, with the delay of every part. Only after shadow flights look sensible:
`--mode active --scale 0.5`, and press MODEL FLY on the phone.

## 4. Landing score (overhead camera)

A phone on a tripod or taped to the ceiling looks straight down at the pad and is not moved.
Per flight: one photo before (pad only), one after landing.

    python3 -m brain.landing_score before.jpg after.jpg --pad-cm 50 --debug check.png

Prints the distance from the pad centre in cm and whether the drone is on the pad. Open
check.png the first few times to see that it found the right things.

## 5. Round 2 tools (data, model insight, evaluation)

| Idea | Command | What it gives you |
|---|---|---|
| X10 recovery views | `DRONE_RECOVERY_VIEWS=2 tfds build` | Each train episode 2 more times, view shifted 5 % left / right, roll corrected back. Val unchanged. |
| V2 paraphrases | `DRONE_PARAPHRASES=1 tfds build` | Train episodes get varied wordings (`paraphrases.txt`); `paraphrases_test.txt` is never trained on. |
| D5 phase labels | `python3 -m brain.phases <episodes> --csv` | climb / search / approach / align / descend per step; the rehearsal now prints the error per phase. |
| D1 coverage map | `python3 -m brain.coverage <episodes> --svg coverage.svg` | Which pad-vs-drone layouts have data (train / val), which are missing. |
| X15 CNN baseline | `python3 cnn_baseline.py train <h5 folder> --out runs/cnn` | ResNet-18 on the same split; same table as `eval_val.py`; serve with `inference_server.py --cnn runs/cnn/best.pt`. |
| M1 confidence | automatic (OpenVLA) | `/act` returns `confidence` (p and spread per stick); logged in `steps.csv`; `--max-spread 15` on the bridge hovers when the model is unsure. |
| M4 what-if lab | `python3 -m brain.whatif --policy <url> <val episodes>` | Mirror, pad removed, shifted, dark / bright, other words, wrong task: how the command changes. |
| E2 blind A/B | `--policy-b <url>` on the bridge, then `brain.ab_test` | Each MODEL FLY flight secretly uses A or B; score landings blind; unblind with a permutation test. |

Both build options are off by default, so `tfds build` without them gives exactly the old
dataset. Use a new dataset version folder (or delete the old one) when you switch them on.

`python3 inference_server.py --mock --mock-seek` is a fake model that steers toward red
pixels: handy to try the tools above without a GPU.

## Safety layers

| Layer | Where | What |
|---|---|---|
| Training range | `brain/governor.py` | Every stick clamped to the 1 %–99 % range of the pilot data; at most 25 % change per command; yaw off unless `--allow-yaw`. |
| Phone limits | app `StickController` | Own absolute caps, never looser than the PC's. |
| Watchdog | app | No fresh command for 0.5 s: hover. |
| Takeover | app | Moving a real stick switches the model off at once. |
| STOP | app | Button (and later the voice word) stops the model and hovers. |
| Landing | app | Below `--land-height` while descending: DJI auto-landing (the training data ends where auto-landing starts). |

## Tests

    python3 -m pytest tests/
