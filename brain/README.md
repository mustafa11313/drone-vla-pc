# PC brain: step 1, the safe loop

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
