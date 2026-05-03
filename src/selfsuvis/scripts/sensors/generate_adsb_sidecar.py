"""Generate a synthetic ADS-B sidecar JSONL for pipeline testing.

Each line is a JSON object with fields:
  t         — timestamp (seconds from mission start)
  icao      — 24-bit ICAO aircraft address (hex string)
  callsign  — flight callsign
  lat, lon  — WGS-84 position
  alt_m     — barometric altitude (metres)
  speed_kts — ground speed (knots)
  heading   — true heading (degrees)

Usage: python generate_adsb_sidecar.py [BASENAME] [OUTPUT_DIR]
"""
import json
import math
import pathlib
import random
import sys

_BASE   = sys.argv[1] if len(sys.argv) > 1 else "sample_mission_042"
_OUTDIR = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent
out = _OUTDIR / f"{_BASE}.adsb.jsonl"
rng = random.Random(42)

BASE_LAT, BASE_LON = 51.5, -0.1   # London

with open(out, "w") as f:
    for t in range(0, 300, 5):
        for i in range(rng.randint(0, 3)):
            angle  = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0.01, 0.05)
            f.write(json.dumps({
                "t":         t,
                "icao":      f"{rng.randint(0, 0xFFFFFF):06X}",
                "callsign":  f"SIM{rng.randint(100,999)}",
                "lat":       BASE_LAT + radius * math.sin(angle),
                "lon":       BASE_LON + radius * math.cos(angle),
                "alt_m":     rng.uniform(100, 3000),
                "speed_kts": rng.uniform(100, 450),
                "heading":   rng.uniform(0, 360),
            }) + "\n")

print(f"Written: {out}  ({out.stat().st_size} bytes)")
