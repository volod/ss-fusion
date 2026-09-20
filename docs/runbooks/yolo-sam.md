# YOLO11 + SAM2/3 Runbook

> Covers: YOLO11 model tiers, SAM backend selection, cache location,
> Semantic Scene Graph (SSG), VRAM requirements, and failure modes.

---

## 1. Architecture overview

```
VideoIndexer / step_yolo_sam (step P2)
  ├─ YOLO11 detection
  │    YOLOTracker.detect_frame()         ← ultralytics, weights in .data/.cache/ultralytics/
  │    → bounding boxes + class labels + confidence
  │    → YOLO_SSG_ENABLED: builds 3D semantic scene graph (frame_facts_json["ssg"])
  │
  └─ SAM2/3 segmentation (when SAM_ENABLED=true)
       SAMPredictor.predict_boxes()       ← auto-backend: sam3 → sam2 → sam1
       → pixel-accurate masks per YOLO detection
       → frame_facts_json["sam_masks"]
```

Both YOLO and SAM are **enabled by default**. Disable with `--no-yolo` or `--no-sam`
in the local CLI.

---

## 2. Environment variables

**YOLO**

| Variable | Default | Description |
|---|---|---|
| `YOLO_ENABLED` | `true` | Enable YOLO detection pass |
| `YOLO_MODEL` | `yolo11l` | Model tier: `yolo11n` / `yolo11s` / `yolo11m` / `yolo11l` / `yolo11x` |
| `YOLO_CONFIDENCE` | `0.25` | Minimum detection confidence |
| `YOLO_SSG_ENABLED` | `true` | Build Semantic Scene Graph from detections |
| `YOLO_SSG_MIN_OBSERVATIONS` | `1` | Min frames an object must appear in to enter SSG |
| `YOLO_SSG_CLUSTER_RADIUS_METERS` | `12.0` | Spatial cluster radius for GPS-registered SSG |

**SAM**

| Variable | Default | Description |
|---|---|---|
| `SAM_ENABLED` | `true` | Enable SAM mask refinement |
| `SAM_MODEL` | `auto` | Backend: `auto` / `sam2` / `sam3` / `sam1` |
| `SAM_CHECKPOINT` | `""` | Path to SAM1 `.pth` file (only needed for SAM1 backend) |
| `SAM_MODEL_TYPE` | `vit_h` | SAM1 model type (`vit_h`, `vit_l`, `vit_b`) |

---

## 3. YOLO model tiers

| Model | File | Params | VRAM | COCO mAP50-95 | Speed |
|---|---|---|---|---|---|
| `yolo11n` | yolo11n.pt | 2.6 M | ~0.1 GB | 39.5 | Fastest |
| `yolo11s` | yolo11s.pt | 9.4 M | ~0.2 GB | 47.0 | Very fast |
| `yolo11m` | yolo11m.pt | 20.1 M | ~0.4 GB | 51.5 | Fast |
| `yolo11l` | yolo11l.pt | 25.3 M | ~0.5 GB | **53.4** | **Default** |
| `yolo11x` | yolo11x.pt | 56.9 M | ~1.1 GB | 54.7 | Slower |

Weights are cached in `.data/.cache/ultralytics/` on first run. Pre-download with:
```bash
python -m selfsuvis.scripts.prepare_models --yolo
python -m selfsuvis.scripts.prepare_models --yolo --yolo-model yolo11x  # specific tier
```

---

## 4. SAM backends

| Backend | Package | HF Model | VRAM | Notes |
|---|---|---|---|---|
| `sam3` | `pip install sam3` | `facebook/sam3` | ~0.5 GB | **Preferred** — latest, auto-downloaded |
| `sam2` | `pip install sam2` | `facebook/sam2-hiera-large` | ~0.5 GB | Fallback; auto-downloaded |
| `sam1` | `pip install segment-anything` | manual `.pth` download | 1–6 GB | Last resort; requires checkpoint |

Auto-detection order: sam3 → sam2 → sam1 → disabled.

Pre-download SAM2/SAM3:
```bash
python -m selfsuvis.scripts.prepare_models --sam
```

---

## 5. Quick start

```bash
# Default (YOLO11l + SAM2/3 auto)
ssv --mode local

# Disable SAM (faster, no masks)
ssv --mode local --no-sam

# Disable both YOLO and SAM
ssv --mode local --no-yolo --no-sam

# Higher-quality YOLO
YOLO_MODEL=yolo11x ssv --mode local

# Faster YOLO
YOLO_MODEL=yolo11n ssv --mode local
```

---

## 6. Health check

```bash
# YOLO
python -c "
from pipeline.vision.yolo import YOLOTracker
from PIL import Image
import numpy as np
t = YOLOTracker()
img = Image.fromarray(np.zeros((640,640,3), dtype='uint8'))
r = t.detect_frame(img)
print('YOLO detections:', len(r))
"

# SAM
python -c "
import os; os.environ['SAM_ENABLED']='true'
from pipeline.vision.sam import SAMPredictor
p = SAMPredictor()
print('SAM available:', p.is_available())
"
```

---

## 7. Semantic Scene Graph (SSG)

When `YOLO_SSG_ENABLED=true`, YOLO detections are clustered into a scene graph that
persists across frames:
- GPS-registered missions: clusters by geographic proximity (`YOLO_SSG_CLUSTER_RADIUS_METERS`)
- Local runs without GPS: clusters by PCA-space proximity (`YOLO_SSG_CLUSTER_RADIUS_PCA`)

Output: `frame_facts_json["ssg"]` per frame, and `semantic_graph_summary` in the
run report. Disable if frame-level detection is sufficient: `YOLO_SSG_ENABLED=false`.

---

## 8. VRAM budget (16 GB GPU)

| Component | VRAM |
|---|---|
| YOLO11l | ~0.5 GB |
| SAM2-hiera-large | ~0.5 GB |
| CLIP ViT-B/16 (always loaded) | ~0.3 GB |
| DINOv3 ViT-B/14 | ~0.4 GB |
| **Total P2 step** | **~1.7 GB** |

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `yolo11l.pt` appears in project root | YOLO downloaded to cwd instead of cache | Check `pipeline/vision/yolo.py` `_get_model()` uses `.data/.cache/ultralytics/` path |
| SAM not producing masks | `SAM_ENABLED=false` or no backend installed | Set `SAM_ENABLED=true`; `pip install sam2` |
| Low detection count on aerial footage | Objects too small for YOLO11l | Try `yolo11x` or lower `YOLO_CONFIDENCE=0.15` |
| `CUDA out of memory` during SAM | SAM + YOLO + CLIP all loaded | Switch to `sam2-hiera-tiny` or disable SAM |
| SSG has too many spurious clusters | Low-confidence detections creating noise | Raise `YOLO_CONFIDENCE=0.4` and `YOLO_SSG_MIN_OBSERVATIONS=2` |
