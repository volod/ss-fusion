# HF Object Detection Runbook (RT-DETR / Grounding DINO)

> Covers: HuggingFace detection pass (step P), model selection,
> open-vocabulary labels, and the difference from YOLO step P2.

---

## 1. Architecture overview

```
VideoIndexer
  └─ Detection pass (batch, step P)
       └─ DetectionModel.detect_batch()    ← loaded in worker VRAM
            → normalised bounding boxes
            → frame_facts_json["detections"]
```

This is the **HuggingFace detection pass** (step P, `DETECTION_ENABLED`), separate
from the YOLO step P2. The two serve different roles:

| Pass | Model | Purpose |
|---|---|---|
| Step P (this runbook) | RT-DETR / Grounding DINO | Open-vocabulary detection, earlier in pipeline, feeds Qwen context |
| Step P2 | YOLO11 + SAM2/3 | High-speed detection + mask refinement for tracking and 3D SSG |

Detection is **disabled by default** (`DETECTION_ENABLED=false`).

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `DETECTION_ENABLED` | `false` | Enable HF detection pass |
| `DETECTION_MODEL` | `auto` | Model ID or `auto` for GPU-aware selection |
| `DETECTION_CONFIDENCE` | `0.5` | Minimum confidence threshold |
| `DETECTION_LABELS` | `""` | Comma-separated labels for open-vocabulary models (empty = COCO classes) |
| `DETECTION_BATCH_SIZE` | `8` | Frames per batch |

---

## 3. Model selection

| Model ID | Params | VRAM | Type | Notes |
|---|---|---|---|---|
| `PekingU/rtdetr_r50vd` | 42 M | ~0.1 GB | Closed-vocab | **Fast**; COCO 80 classes; 53.1 mAP |
| `PekingU/rtdetr_r101vd` | 76 M | ~0.2 GB | Closed-vocab | Higher accuracy; same speed class |
| `IDEA-Research/grounding-dino-tiny` | 173 M | ~0.4 GB | Open-vocab | Zero-shot; text-guided labels |
| `IDEA-Research/grounding-dino-base` | 341 M | ~0.7 GB | Open-vocab | **Best open-vocab quality** |
| `omlab/omdet-turbo-swin-large-hf` | 218 M | ~0.5 GB | Open-vocab | Strong speed/accuracy |

**Auto-selection**: picks `rtdetr_r50vd` for < 2 GB, `grounding-dino-tiny` for 2–4 GB,
`grounding-dino-base` for > 4 GB.

---

## 4. Quick start

```bash
# Enable with auto model
DETECTION_ENABLED=true ssv --mode local

# Open-vocabulary with custom labels
DETECTION_ENABLED=true \
  DETECTION_MODEL=IDEA-Research/grounding-dino-base \
  DETECTION_LABELS="vehicle,person,weapon,container,antenna" \
  ssv --mode local

# Download weights
python -m selfsuvis.scripts.prepare_models --detection
```

---

## 5. Health check

```bash
python -c "
import os; os.environ['DETECTION_ENABLED']='true'
from pipeline.vision.detection import DetectionModel
from PIL import Image
import numpy as np
m = DetectionModel()
img = Image.fromarray(np.zeros((640,640,3), dtype='uint8'))
r = m.detect_batch([img])
print('Detection output:', r)
"
```

---

## 6. Open-vocabulary vs closed-vocabulary

**Closed-vocabulary (RT-DETR):** Pre-trained on COCO 80 classes. Fast and reliable
for common objects. Does not accept custom label prompts.

**Open-vocabulary (Grounding DINO, OmDet-Turbo):** Accepts any text labels via
`DETECTION_LABELS`. Use for domain-specific objects not in COCO:
```bash
DETECTION_LABELS="drone,UAV,antenna tower,fuel drum,IED,person with weapon"
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Detection pass skipped` | `DETECTION_ENABLED=false` | Set `DETECTION_ENABLED=true` |
| No detections despite visible objects | `DETECTION_CONFIDENCE` too high | Lower to `0.3` |
| Open-vocab model ignores custom labels | Labels not set or model is closed-vocab | Verify `DETECTION_LABELS` is set and model supports open-vocab |
| `CUDA out of memory` | Grounding DINO + other models | Switch to `rtdetr_r50vd` or run on separate GPU |
| Very slow: >3s per frame | CPU inference | Set `DEVICE=cuda` |
