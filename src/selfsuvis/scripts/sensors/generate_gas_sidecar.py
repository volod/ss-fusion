"""Generate synthetic gas / radiation sidecar JSONL for pipeline testing.

Usage: python generate_gas_sidecar.py [BASENAME] [OUTPUT_DIR]
"""
import json, random, pathlib, math, sys

_BASE   = sys.argv[1] if len(sys.argv) > 1 else "sample_mission_042"
_OUTDIR = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent
rng = random.Random(42)
out = _OUTDIR / f"{_BASE}.gas.jsonl"

with open(out, "w") as f:
    for t in range(300):
        plume = 400 * math.exp(-((t - 150) ** 2) / (2 * 30 ** 2))
        f.write(json.dumps({
            "t":               t,
            "co2_ppm":         round(410 + plume + rng.gauss(0, 5), 1),
            "voc_ppb":         round(max(0, 20 + rng.gauss(0, 8)), 1),
            "no2_ppb":         round(max(0, 8  + rng.gauss(0, 3)), 1),
            "pm25_ug_m3":      round(max(0, 12 + rng.gauss(0, 4)), 1),
            "pm10_ug_m3":      round(max(0, 20 + rng.gauss(0, 6)), 1),
            "dose_rate_usv_h": round(max(0, 0.08 + rng.gauss(0, 0.01)), 3),
        }) + "\n")

print(f"Written: {out}  ({out.stat().st_size} bytes)")
