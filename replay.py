from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

csv_file = "/home/shk/drone_database/extracted_csv/telemetry12.csv"
df = pd.read_csv(csv_file)

current_index = 0

TIME_COL = "time_ms"
PITCH_COL = "pitch_cmd"
ROLL_COL = "roll_cmd"
YAW_COL = "yaw_cmd"
THROTTLE_COL = "throttle_cmd"
FLIGHT_MODE_COL = "flight_mode"
GIMBAL_PITCH_COL = "gimbal_pitch"

OUTPUT_SCALE = 1.0

FINAL_SCALE = OUTPUT_SCALE / 100.0


def pct_to_unit(value, max_unit=None):
    return float(value) * FINAL_SCALE

PITCH_SIGN = -1

# How strong the replay is. The original script sent the stick % divided by 100, which the
# drone treats as about 1 degree / 1 m/s at most, so the flight is barely visible.
# The drone does not react in a straight line to the stick: small stick values give very
# little tilt or turning, full stick gives the maximum. This curve was fitted to your own
# recorded flights (stick % against measured tilt and turn rate):
#   value = max * (0.12 * x + 0.88 * x^3), x = stick % / 100
# TILT_MAX_DEG = tilt at full stick, YAW_MAX_DPS = turning at full stick (deg/s),
# THROTTLE_GAIN = m/s of climb per stick %.
# STRENGTH scales everything at once: 1.0 = like the original flight, 0.5 = half as strong.
STRENGTH = 1.0
TILT_MAX_DEG = 37.0
YAW_MAX_DPS = 60.0
THROTTLE_GAIN = 0.003 * STRENGTH

def stick_curve(pct):
    x = float(pct) / 100.0
    return 0.12 * x + 0.88 * x ** 3

def gimbal_pitch(row):
    # Camera tilt of this row in degrees (0 = straight ahead, negative = looking down).
    # Rows without the column (old files) give 0.
    try:
        return float(row[GIMBAL_PITCH_COL])
    except Exception:
        return 0.0

def tilt_deg(pct):
    return stick_curve(pct) * TILT_MAX_DEG * STRENGTH

def turn_dps(pct):
    return stick_curve(pct) * YAW_MAX_DPS * STRENGTH


HARD_SKIP_PHASES = {"Motors_Started", "Confirm_Landing"}
TAKEOFF_PHASES = {"Assisted_Takeoff", "AutoTakeoff"}
ARMING_GESTURE_THRESHOLD = 90.0


def compute_skip_flags(df):
    skip = [False] * len(df)
    prev_phase = None
    in_leading_edge = False

    for i, row in df.iterrows():
        phase = str(row[FLIGHT_MODE_COL]) if FLIGHT_MODE_COL in df.columns else ""

        if phase in HARD_SKIP_PHASES:
            skip[i] = True
            in_leading_edge = False
        elif phase in TAKEOFF_PHASES:
            if phase != prev_phase:
                in_leading_edge = True
            if in_leading_edge:
                raw_vals = [
                    abs(float(row[PITCH_COL])), abs(float(row[ROLL_COL])),
                    abs(float(row[YAW_COL])), abs(float(row[THROTTLE_COL])),
                ]
                if all(v >= ARMING_GESTURE_THRESHOLD for v in raw_vals):
                    skip[i] = True
                else:
                    in_leading_edge = False
        else:
            in_leading_edge = False

        prev_phase = phase

    return skip


SKIP_FLAGS = compute_skip_flags(df)


def hover_response(status, duration_ms=0):
    return jsonify({
        "pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0,
        "duration_ms": duration_ms, "status": status
    })


@app.route("/command", methods=["GET"])
def command():
    global current_index
    cmd = request.args.get("cmd", "stop").lower().strip()

    if cmd == "replay":
        if current_index >= len(df):
            print(f"[REPLAY] row {current_index}/{len(df)} -- FINISHED")
            return hover_response("finished")

        row = df.iloc[current_index]
        i = current_index

        if current_index + 1 < len(df):
            duration_ms = int(df.iloc[current_index + 1][TIME_COL] - row[TIME_COL])
            if duration_ms <= 0:
                duration_ms = 100
        else:
            duration_ms = 100

        current_index += 1
        flight_mode = str(row[FLIGHT_MODE_COL]) if FLIGHT_MODE_COL in df.columns else ""

        if SKIP_FLAGS[i]:
            reason = "non-flight phase" if flight_mode in HARD_SKIP_PHASES else "arming-gesture"
            print(f"[REPLAY] row {current_index}/{len(df)}  phase={flight_mode}  -- SKIPPED ({reason})")
            return jsonify({
                "pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0,
                "duration_ms": duration_ms, "row": current_index, "status": "ok",
                "note": f"skipped ({reason}): {flight_mode}"
            })

        pitch = tilt_deg(row[PITCH_COL]) * PITCH_SIGN
        roll = tilt_deg(row[ROLL_COL])
        yaw = turn_dps(row[YAW_COL])
        throttle = float(row[THROTTLE_COL]) * THROTTLE_GAIN

        print(f"[REPLAY] row {current_index}/{len(df)}  phase={flight_mode}  "
              f"pitch={pitch:+.2f}  roll={roll:+.2f}  yaw={yaw:+.2f}  throttle={throttle:+.2f}")

        return jsonify({
            "pitch": pitch, "roll": roll, "yaw": yaw, "throttle": throttle,
            "duration_ms": duration_ms, "row": current_index, "status": "ok"
        })

    if cmd == "plan":
        # The whole episode in one answer: [pitch, roll, yaw, throttle, duration_ms] per row,
        # with exactly the same values the "replay" command returns row by row. The app plays
        # it from its own clock, so no network delay slows the replay down.
        plan = []
        for i in range(len(df)):
            row = df.iloc[i]
            if i + 1 < len(df):
                duration_ms = int(df.iloc[i + 1][TIME_COL] - row[TIME_COL])
                if duration_ms <= 0:
                    duration_ms = 100
            else:
                duration_ms = 100
            if SKIP_FLAGS[i]:
                plan.append([0.0, 0.0, 0.0, 0.0, duration_ms, gimbal_pitch(row)])
                continue
            plan.append([tilt_deg(row[PITCH_COL]) * PITCH_SIGN,
                         tilt_deg(row[ROLL_COL]),
                         turn_dps(row[YAW_COL]),
                         float(row[THROTTLE_COL]) * THROTTLE_GAIN,
                         duration_ms,
                         gimbal_pitch(row)])
        print(f"[PLAN] sent {len(plan)} rows, {sum(r[4] for r in plan) / 1000:.1f} s")
        return jsonify({"status": "ok", "rows": plan})

    if cmd == "reset":
        current_index = 0
        print(f"[REPLAY] RESET -- row 0/{len(df)}")
        return jsonify({"status": "reset"})

    return hover_response("stop")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "replay server running",
        "usage": "/command?cmd=replay",
        "rows": len(df),
        "current_row": current_index
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
