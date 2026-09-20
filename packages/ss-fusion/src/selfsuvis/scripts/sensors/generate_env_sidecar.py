"""Generate synthetic atmospheric / environmental sidecar JSONL.

Usage: python generate_env_sidecar.py [BASENAME] [OUTPUT_DIR]
"""

import json
import math
import pathlib
import random
import sys

_BASE = sys.argv[1] if len(sys.argv) > 1 else "sample_mission_042"
_OUTDIR = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent
rng = random.Random(7)
out = _OUTDIR / f"{_BASE}.env.jsonl"

with open(out, "w") as f:
    for t in range(300):
        temp = 18.5 + rng.gauss(0, 0.2) - 0.0065 * 120
        rh = 65.0 + rng.gauss(0, 2.0)
        press = 1013.25 * (1 - 120 / 44330) ** 5.255 + rng.gauss(0, 0.1)
        wind_sp = max(0, rng.gauss(3.5, 1.0))
        wind_dr = (180 + rng.gauss(0, 10)) % 360
        solar = max(0, 650 + rng.gauss(0, 30) * math.cos(math.pi * t / 300))
        f.write(
            json.dumps(
                {
                    "t": t,
                    "temp_c": round(temp, 2),
                    "humidity_pct": round(min(100, max(0, rh)), 1),
                    "pressure_hpa": round(press, 3),
                    "wind_speed_ms": round(wind_sp, 2),
                    "wind_dir_deg": round(wind_dr, 1),
                    "solar_w_m2": round(solar, 1),
                }
            )
            + "\n"
        )

print(f"Written: {out}  ({out.stat().st_size} bytes)")
