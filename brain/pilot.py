"""
pilot.py  -  the high-level pilot ("brain"): looks at the scene every 1-2 s and picks the next
skill (brain/skills.py). It never moves the sticks itself.

Two pilots with the same decide() interface:

  ClaudePilot    Claude looks at the newest frame (and the one before), the telemetry and what
                 happened so far, and answers with one Decision (structured output).
                 Needs `pip install anthropic pydantic` and ANTHROPIC_API_KEY in your terminal.
  ScriptedPilot  a fixed rule set that does the same job without any API: the baseline arm
                 for the thesis and the stand-in for tests.

decide(obs, context) -> Decision
  obs     = {"jpeg", "telemetry", "t"} (the same block the phone sends)
  context = {"mission": str, "history": [last decisions with how they ended],
             "current": description of the running skill or None, "previous_jpeg": bytes | None}

skill "continue" means: keep the running skill; "done" means: the mission is over (landed).
"""
import base64
import io
import json
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from brain.skills import pad_info

MODEL = "claude-opus-5-5"

SkillName = Literal["continue", "hover", "rotate", "climb", "move", "vla", "servo", "land", "abort", "done"]


class Decision(BaseModel):
    observation: str = Field(description="What you see that matters, in one sentence")
    pad_visible: bool
    skill: SkillName
    degrees: Optional[float] = Field(None, description="rotate: + clockwise, - counter-clockwise")
    meters: Optional[float] = Field(None, description="climb: + up, - down")
    forward_m: Optional[float] = Field(None, description="move: + forward, - back")
    right_m: Optional[float] = Field(None, description="move: + right, - left")
    instruction: Optional[str] = Field(None, description="vla: the sentence for the VLA model")
    seconds: Optional[float] = Field(None, description="hover / vla / servo: how long")
    reason: str = Field(description="Why this skill now, in one short sentence")


SYSTEM = """You are the high-level pilot of a small indoor drone (DJI Mini 3) in a closed flight cage:
2 m x 2 m floor, 3 m high, black net on all sides and on top, grey-beige foam mats on the floor
(darker strip along the net). A square red landing pad (50 cm, white H) lies somewhere on the floor.
There is no GPS: you only have the drone's camera and its telemetry (height above take-off, speeds,
heading). The camera is fixed 50 degrees down and sees the floor ahead of the drone; the spot
directly below the drone is NOT in view. So when the drone is right above the pad, the pad has
just slid out of the BOTTOM of the picture.

You do not fly the sticks. Every 1-2 seconds you choose one skill; the skill then flies at 10 Hz
until it finishes or you replace it:

  rotate(degrees)            turn on the spot; + = clockwise (to the right). Use it to search.
  climb(meters)              + up, - down; height stays between 0.4 and 2.2 m. Higher = wider view.
  move(forward_m, right_m)   short straight move, at most 0.8 m per axis, measured from speeds.
  vla(instruction, seconds)  a learned model (VLA) flies from the picture. It was trained on
                             pilot flights with the sentence "Take off, locate the red landing pad,
                             and land on it." and is good at flying toward a VISIBLE pad and
                             descending; it is weak at turning and sideways moves.
  servo(seconds)             a classic controller that centres a VISIBLE pad and descends onto it.
  land                       DJI auto-landing straight down from where the drone is now.
  hover(seconds)             wait.
  continue                   keep the skill that is running now.
  abort                      land now because something is unsafe.
  done                       only after the drone has landed.

Rules:
- Stay away from the net. The cage is small (about 1 m from the centre to each side): never
  move more than 0.5 m toward a side you cannot see is free.
- Search for the pad by rotating (about 60-90 degrees at a time) and, if needed, climbing a little.
- When the pad is visible in the upper or middle part of the picture: approach with vla or move.
- When the pad is large and in the lower part of the picture and roughly centred left-right:
  use servo, or land when it has just left the bottom edge after being centred.
- If a skill is making things worse (the pad drifts away, the drone nears the net), replace it.
- Prefer "continue" while the running skill is clearly working.
Answer with exactly one decision."""


def _image_block(jpeg, width=480):
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def context_text(obs, context):
    tel = obs["telemetry"]
    facts = {"time_s": obs.get("t"), "height_m": tel.get("height_m"), "heading_deg": tel.get("heading"),
             "speed_north_mps": tel.get("vx"), "speed_east_mps": tel.get("vy"), "speed_down_mps": tel.get("vz"),
             "red_pixels_detector": pad_info(obs["jpeg"]),
             "running_skill": context.get("current"), "mission": context.get("mission")}
    hist = context.get("history", [])[-8:]
    return ("Telemetry and state:\n" + json.dumps(facts) +
            "\n\nYour last decisions and how they ended (oldest first):\n" +
            ("\n".join(json.dumps(h) for h in hist) if hist else "(none yet)") +
            "\n\nThe last image is the newest camera frame" +
            (", the one before it is from about 1-2 s earlier." if context.get("previous_jpeg") else ".") +
            " red_pixels_detector: x, y in -1..1 from the image centre (y = 1 is the bottom edge).")


class ClaudePilot:
    def __init__(self, model=MODEL, effort="low", client=None):
        self.model = model
        self.effort = effort
        self._client = client
        self.calls = 0
        self.last_ms = 0.0

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def decide(self, obs, context):
        content = []
        if context.get("previous_jpeg"):
            content.append(_image_block(context["previous_jpeg"]))
        content.append(_image_block(obs["jpeg"]))
        content.append({"type": "text", "text": context_text(obs, context)})
        t0 = time.perf_counter()
        r = self.client.beta.messages.parse(
            model=self.model, max_tokens=4000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # another model answers if this one declines
            messages=[{"role": "user", "content": content}],
            output_format=Decision)
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        if r.stop_reason == "refusal" or r.parsed_output is None:
            return Decision(observation="no answer", pad_visible=False, skill="hover", seconds=1.0,
                            reason=f"no decision ({r.stop_reason}): hover")
        return r.parsed_output


class ScriptedPilot:
    """Rule-based brain with the same skills: search by turning, approach, centre, land."""

    def __init__(self, approach="vla"):
        self.approach = approach          # "vla" or "servo" or "move"
        self.turns = 0
        self.last_ms = 0.0

    def decide(self, obs, context):
        t0 = time.perf_counter()
        pad = pad_info(obs["jpeg"])
        h = float(obs["telemetry"].get("height_m", 1.0))
        current = context.get("current")
        hist = context.get("history", [])
        last = hist[-1] if hist else {}
        if not pad["visible"]:
            # just lost it at the bottom after centring -> it is under us: land
            if last.get("skill") in ("servo", "vla") and last.get("pad_y_before", -1) > 0.45:
                d = Decision(observation="pad left the bottom of the picture", pad_visible=False,
                             skill="land", reason="pad was low and centred, it is under the drone")
            elif current and current.startswith("rotate"):
                d = Decision(observation="no pad yet", pad_visible=False, skill="continue", reason="still turning")
            elif self.turns >= 4 and h < 1.8:
                self.turns = 0
                d = Decision(observation="no pad after a full turn", pad_visible=False, skill="climb",
                             meters=0.4, reason="climb for a wider view")
            else:
                self.turns += 1
                d = Decision(observation="no pad in view", pad_visible=False, skill="rotate", degrees=90,
                             reason="search by turning")
        else:
            self.turns = 0
            x, y = pad["x"], pad["y"]
            if current and current.startswith("rotate") and abs(x) > 0.5:
                d = Decision(observation=f"pad at x={x}", pad_visible=True, skill="rotate",
                             degrees=round(x * 35), reason="turn to face the pad")
            elif abs(x) > 0.6:
                d = Decision(observation=f"pad far to the side (x={x})", pad_visible=True, skill="rotate",
                             degrees=round(x * 35), reason="face the pad")
            elif y > 0.35 or self.approach == "servo":
                if current and current.startswith("servo"):
                    d = Decision(observation=f"pad low at ({x}, {y})", pad_visible=True, skill="continue",
                                 reason="servo is centring")
                else:
                    d = Decision(observation=f"pad low at ({x}, {y})", pad_visible=True, skill="servo",
                                 seconds=10, reason="pad close: centre and descend")
            elif self.approach == "move":
                d = Decision(observation=f"pad ahead at ({x}, {y})", pad_visible=True, skill="move",
                             forward_m=0.3, right_m=round(0.3 * x, 2), reason="step toward the pad")
            elif current and current.startswith("vla"):
                d = Decision(observation=f"pad ahead at ({x}, {y})", pad_visible=True, skill="continue",
                             reason="VLA is approaching")
            else:
                d = Decision(observation=f"pad ahead at ({x}, {y})", pad_visible=True, skill="vla",
                             seconds=4, reason="VLA approaches a visible pad")
        self.last_ms = (time.perf_counter() - t0) * 1000
        return d
