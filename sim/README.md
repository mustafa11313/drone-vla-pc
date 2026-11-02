# Cage simulator: Claude as the brain, the VLA as the hands

A first, simple copy of the 2 × 2 × 3 m flight cage to test the whole high-level loop before
flying: a pilot (Claude, or a rule-based stand-in) picks a skill every 1–2 s, the skill flies at
10 Hz through the governor, and a drone model fitted from the recorded flights moves the
drone. No GPU is needed; only numpy and Pillow (plus `anthropic` for the Claude pilot).

```
camera frame + telemetry ──▶ pilot (brain/pilot.py)  every 1–2 s ──▶ skill choice
                         └─▶ skill (brain/skills.py) 10 Hz ──▶ governor ──▶ sticks ──▶ drone model (sim/dynamics.py)
                                                                                         └─▶ new frame (sim/render.py)
```

## Run

    # no API key, no GPU: rule-based brain + a fake "VLA" that steers toward red pixels
    python3 -m sim.run --pilot scripted --policy seek --missions 20 --servo-target-y 0.75

    # Claude as the brain (ANTHROPIC_API_KEY set in your own terminal)
    python3 -m sim.run --pilot claude --policy seek --missions 5 --video

    # the real VLA (inference_server.py on the cluster) as the hands, with and without a brain
    python3 -m sim.run --pilot claude --policy http://<gpu-host>:8000 --missions 20
    python3 -m sim.run --pilot none   --policy http://<gpu-host>:8000 --missions 20

The same `--seed` gives the same pad and start positions, so arms are compared on the same
course. Each mission writes `logs/sim_*/mission_<seed>/` with `steps.csv` (pose, sticks, which
skill flew), `decisions.jsonl` (every brain decision with its answer time and reason),
`result.json` and, with `--video`, `flight.gif`. `summary.csv` has one line per mission.

## Files

| File | What |
|---|---|
| `sim/dynamics.py` | Mini 3 motion model: sticks → target speeds through the stick curve of `replay.py`, first-order response. Constants in `sim/mini3_params.json`. |
| `sim/fit_dynamics.py` | Fits those constants to recorded episodes: `python3 -m sim.fit_dynamics <episodes> --save`. The current file is fitted from the **two sample episodes only**; run it on all 50. |
| `sim/render.py` | The camera view: mats, pad (50 cm, H), net, the hall behind it. 16:9, 74° wide, gimbal 50° down, about 25 ms per frame. |
| `sim/env.py` | One flight: DJI auto-landing below 0.6 m when descending, net contact ends the flight, landing on or off the pad, timeout. |
| `sim/mission.py` | The closed loop with honest timing: the brain's real answer time also passes in the simulation, the running skill keeps flying meanwhile, and a silent brain means hover, then land after 15 s. |
| `sim/run.py` | Many missions, summary, success rate. |
| `brain/skills.py` | The skills: hover, rotate, climb, move, vla, servo, land. They only use what the real drone has (picture, height, speeds, heading), so the same code can later run in the bridge. |
| `brain/pilot.py` | `ClaudePilot` (structured output: one `Decision` per call, system prompt cached, `effort=low` for speed) and `ScriptedPilot` (the rule-based baseline). |

## First results (fake VLA, not the real model)

20 missions each, same seeds, `--policy seek` (a hand-made controller standing in for the VLA):

| Arm | Landed on the pad | Other outcomes |
|---|---|---|
| no brain, fake VLA alone | 11 / 20 | 9 hit the net (no search: it flies off when the pad is not in view) |
| scripted brain + fake VLA + servo, servo target 0.75 | 20 / 20 | – |
| scripted brain + servo only, servo target 0.75 | 20 / 20 | – |
| scripted brain + servo only, servo target 0.55 (current default) | 1 / 20 | 19 landed about 35 cm short of the pad |

The last line is a finding about `brain/servo.py`: with the camera 50° down, holding the pad at
`TARGET_Y = 0.55` while descending leaves it about half the height ahead of the drone, and
DJI's auto-landing at 0.6 m then comes down short. In the simulator `0.75` works. This depends
on the simulated camera matching the real one (field of view, gimbal angle), so check it in
shadow mode (`steps.csv`, `pad_y` when the pilot is right above the pad) before changing the
default.

## The lab in 3D (Blender)

`sim/blender/build_lab.py` builds the lab around the cage as a real 3D scene in Blender, from
simple shapes placed from the lab video: the cage (black posts, sagging rope net, black worn
puzzle mats, the pad), the desks, cabinet, bench, fume extractor, robots, walls, ducts and
lights. It renders outside views and drone-camera views, with the drone's auto-exposure (the
floor is black, so the camera brightens it a lot) and warm white balance.

    pip install bpy pillow numpy        # Blender as a Python module, no GPU needed
    python3 sim/blender/build_lab.py --save lab.blend --previews previews/

Positions are estimates; change them in `build_surroundings`. About 20 s per frame on 4 CPU
cores, so it is not yet used inside `sim/run.py` (next step: a fast renderer for flights).

## Limits of this first version

- The picture is drawn from simple shapes. A VLA trained on real frames will probably fly worse
  here than on the drone; this version mainly tests the brain, the skills and the safety rules.
  The realistic copy of the cage (Gaussian Splatting from a video of the cage) replaces
  `sim/render.py` later; everything else stays.
- The motion model is fitted from two episodes, with coarse telemetry (0.1 m/s, 0.1 m steps).
- No wind, no ground effect, no motion blur, constant light. The model's own delay (150–200 ms
  for OpenVLA) is not simulated yet; the brain's delay is.
- The Claude pilot has not been run yet (no API key here). The request shape is tested with a
  fake client in `tests/test_sim.py`.
