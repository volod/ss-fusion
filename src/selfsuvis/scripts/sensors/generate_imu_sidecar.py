"""Generate synthetic IMU + barometer sidecar JSONLs for pipeline testing.

Usage: python generate_imu_sidecar.py [BASENAME] [OUTPUT_DIR]
"""
import json
import pathlib
import random
import sys

_BASE   = sys.argv[1] if len(sys.argv) > 1 else "sample_mission_042"
_OUTDIR = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent
rng  = random.Random(99)
base = _OUTDIR

imu_out = base / f"{_BASE}.imu.jsonl"
with open(imu_out, "w") as f:
    for i in range(2000):
        t = i / 200.0
        f.write(json.dumps({
            "t":  round(t, 5),
            "ax": rng.gauss(0.05, 0.15),
            "ay": rng.gauss(0.02, 0.12),
            "az": rng.gauss(-9.81, 0.08),
            "gx": rng.gauss(0.0,  0.01),
            "gy": rng.gauss(0.0,  0.01),
            "gz": rng.gauss(0.0,  0.005),
        }) + "\n")
print(f"IMU: {imu_out}  (2000 samples @ 200 Hz)")

baro_out = base / f"{_BASE}.baro.jsonl"
alt_m = 120.0
with open(baro_out, "w") as f:
    for i in range(50):
        t = i / 5.0
        alt_m += rng.gauss(0.1, 0.05)
        pressure = 1013.25 * (1 - alt_m / 44330) ** 5.255
        f.write(json.dumps({
            "t":            round(t, 3),
            "pressure_hpa": round(pressure + rng.gauss(0, 0.05), 4),
            "temp_c":       round(15.0 - 0.0065 * alt_m + rng.gauss(0, 0.05), 2),
        }) + "\n")
print(f"Baro: {baro_out}  (50 samples @ 5 Hz)")

wind_out = base / f"{_BASE}.wind.jsonl"
with open(wind_out, "w") as f:
    for i in range(10):
        f.write(json.dumps({
            "t":         float(i),
            "speed_ms":  round(abs(rng.gauss(3.0, 1.5)), 2),
            "dir_deg":   round(rng.gauss(180, 15) % 360, 1),
            "gust_ms":   round(abs(rng.gauss(5.0, 1.5)), 2),
        }) + "\n")
print(f"Wind: {wind_out}  (10 samples @ 1 Hz)")
