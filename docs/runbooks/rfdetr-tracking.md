# RF-DETR Tracking Runbook

> Covers: RF-DETR model tiers, Gemma-directed tracking flow,
> SAM Path A vs Path B, IoU matching, and the AMG freeze fix.

---

## 1. Architecture overview

```
step_gemma_directed_tracking (step P3)
  ├─ Sub-step 1: Gemma structured scene analysis (12 sampled frames)
  │    → scene_type, dominant_objects with rough_bbox, tracking_priority
  │
  ├─ Sub-step 2: SAM directed segmentation (12 sampled frames)
  │    Path A (preferred): Gemma rough_bbox → SAMPredictor.predict_boxes()
  │    Path B (fallback):  SAM AMG + CLIP filtering  ← only when Path A yields nothing
  │                                                    capped at _MAX_PATH_B_FRAMES=3
  │
  └─ Sub-step 3: RF-DETR tracking (up to 90 frames)
       RFDETRBase or RFDETRLarge
       → greedy IoU matching (threshold 0.45) across consecutive frames
       → persistent track IDs (reset per video)
       → frame_facts_json["gemma_tracking"]
       → gemma_tracking_results.json, gemma_tracking_summary.md
```

Requires `GEMMA_API_URL` to be set. Silently skipped otherwise.
Disable with `RFDETR_ENABLED=false` or `--no-rfdetr`.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `RFDETR_ENABLED` | `true` | Enable Gemma-directed RF-DETR tracking |
| `RFDETR_MODEL` | `base` | `base` (RFDETRBase, faster) or `large` (RFDETRLarge, better accuracy) |
| `RFDETR_CONFIDENCE` | `0.35` | Minimum detection confidence for RF-DETR |
| `GEMMA_API_URL` | `""` | Required — Gemma sidecar for scene understanding |

---

## 3. RF-DETR model tiers

| Tier | Class | Params | VRAM | COCO mAP | Notes |
|---|---|---|---|---|---|
| `base` | `RFDETRBase` | ~29 M | ~0.3 GB | 53.4 | **Default** — fast, good for most scenes |
| `large` | `RFDETRLarge` | ~128 M | ~1.0 GB | 62.1 | Better for small objects, complex scenes |

Install: `pip install rfdetr`

---

## 4. Quick start

```bash
# Enable Gemma + RF-DETR tracking (Gemma sidecar required)
ssv --mode local --gemma-api-url http://localhost:11434/v1

# Larger RF-DETR for better accuracy
ssv --mode local --gemma-api-url http://localhost:11434/v1 --rfdetr-model large

# Disable RF-DETR (Gemma scene analysis still runs, but no tracking)
ssv --mode local --gemma-api-url http://localhost:11434/v1 --no-rfdetr
```

---

## 5. SAM Path A vs Path B

Path A (Gemma bbox prompts) is the primary path. Path B (SAM AMG) is a **pure
fallback** that only runs when Path A yields no masks at all.

**Why this matters:** Abstract Gemma categories like `traffic_flow`, `intersection`,
`roadway_infrastructure` always receive fallback bounding boxes covering ~80% of the
image. If Path B ran as a supplement (old behavior), `SAM2AutomaticMaskGenerator.generate()`
would execute for all 12 sampled frames — causing 30+ minute freezes.

| Condition | Path taken |
|---|---|
| Gemma gives precise bbox (area < 72% of frame) | Path A: `SAMPredictor.predict_boxes()` (~0.3s/frame) |
| Gemma gives fallback bbox AND Path A found ≥ 1 mask | Path A result kept; Path B skipped |
| Gemma gives fallback bbox AND Path A found nothing | Path B: AMG → CLIP filter (capped at 3 frames) |

---

## 6. IoU tracking explained

RF-DETR detections are matched across frames by Intersection over Union (IoU):
- IoU ≥ 0.45: detection continues an existing track (same track ID)
- IoU < 0.45 for all existing tracks: new track started
- Track IDs are per-video integers starting from 1, reset between videos

**Tuning IoU threshold:** Lower (e.g., 0.3) allows tracking through larger motion
between frames. Higher (e.g., 0.6) is stricter — fewer false continuations but more
track breaks. Default 0.45 is calibrated for ~2 FPS keyframe extraction.

---

## 7. Health check

```bash
# Verify RF-DETR loads
python -c "
from pipeline.vision.rfdetr import RFDETRTracker
t = RFDETRTracker()
print('RF-DETR loaded:', t._model is not None or 'lazy-load')
"

# Check Gemma endpoint (required for step P3)
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e4b","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
# Expected: 200
```

---

## 8. Output artifacts

After a run with Gemma + RF-DETR enabled:

```bash
ls output/<video>/
# gemma_tracking_results.json     — per-frame detections with track IDs + SAM metadata
# gemma_tracking_summary.md       — Gemma scene intel, tracking summary
# gemma_tracking/frame_*_tracked.jpg  — annotated frames
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Step P3 skipped: gemma_api_url not configured` | No `GEMMA_API_URL` | Set `--gemma-api-url` or `GEMMA_API_URL` |
| SAM step freezes for 30+ minutes | Old code: Path B running on all frames | Ensure fix is applied (`path_b_needed = not path_a_found`) |
| `SAM frame N/12: 0 masks` | Gemma gave all-fallback bboxes, Path B capped | Normal when objects are abstract; tracking still proceeds via RF-DETR alone |
| Tracks break every few frames | Fast camera motion, IoU below 0.45 | Lower `--rfdetr-confidence` or switch to `--rfdetr-model large` |
| `rfdetr` not found | Package not installed | `pip install rfdetr` |
| Wrong tracking categories | Gemma returned broad labels | Check `gemma_tracking_summary.md` for parsed categories; tune Gemma prompt via `GEMMA_API_MODEL` |
