"""Generate a synthetic acoustic event log JSONL for pipeline testing.

Usage: python generate_acoustic_sidecar.py [BASENAME] [OUTPUT_DIR]
"""
import json
import pathlib
import random
import sys

_BASE   = sys.argv[1] if len(sys.argv) > 1 else "sample_mission_042"
_OUTDIR = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent
rng = random.Random(13)
CLASSES = ["drone_motor", "wind", "bird_call", "vehicle_engine", "human_voice", "silence"]
out = _OUTDIR / f"{_BASE}.acoustic_events.jsonl"

with open(out, "w") as f:
    t = 0.0
    while t < 300.0:
        label = rng.choice(CLASSES)
        dur   = rng.uniform(0.5, 5.0)
        conf  = rng.uniform(0.6, 0.99)
        f.write(json.dumps({
            "t_start":    round(t, 3),
            "t_end":      round(t + dur, 3),
            "label":      label,
            "confidence": round(conf, 3),
            "doa_deg":    round(rng.uniform(0, 360), 1),
        }) + "\n")
        t += dur + rng.uniform(0.1, 2.0)

print(f"Written: {out}  ({out.stat().st_size} bytes)")
