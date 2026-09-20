# Runbook: Drone Detection Edge Training (Step 30)

Trains a YOLOv8n drone detector on a public dataset plus mission-derived hard
negatives, then exports edge models ready for two hardware targets:

| Target | Model file | Runtime |
|--------|-----------|---------|
| Arm Cortex-A76 (RPi 5, Orin A76 cores) | `drone_yolo8n_a76.onnx` | onnxruntime (CPU) |
| Rockchip RV1106G3 (Luckfox Pico, IPC-AI) | `drone_yolo8n_rv1106_int8.onnx` | onnxruntime int8 / RKNN NPU |

---

## Prerequisites

| Dependency | Notes |
|------------|-------|
| `ultralytics>=8.3` | already in `pyproject.toml` |
| `huggingface_hub` | already in `pyproject.toml` |
| `onnxruntime>=1.18` | already in `pyproject.toml` |
| `rknn-toolkit2` | optional; only needed for RKNN export; install from [github.com/airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) |

Internet access is required for the first run so that HuggingFace can download
the seraphim dataset batch and the YOLOv8n pretrained weights (~6 MB).
Subsequent runs use a shared cache at `.data/local_runs/_drone_detection_cache/`.

---

## CLI flags

| Flag | Default | Effect |
|------|---------|--------|
| `--drone-detection` | on | Enable step 30 |
| `--no-drone-detection` | — | Skip step 30 entirely |

Step 30 is **on by default** when neither flag is passed.
To opt out for a quick run:

```bash
ssv --mode local --videos-dir .data/videos --no-drone-detection ...
```

---

## Code paths

The shipped local pipeline step is [`steps_drone_detection.py`](../../packages/ss-fusion/src/ssv_vdp/steps/drone_detection.py).
That is the code path the runner executes for Step 30, and it currently trains a YOLOv8n model, exports ONNX, and optionally builds an RKNN artifact.

There is also a newer standalone training helper at [`drone_detector.py`](../../packages/ss-fusion/src/selfsuvis/pipeline/training/drone_detector.py) with:

- `DroneDetectorConfig`
- `run_drone_detection_training()`
- `export_drone_detector_onnx()`
- `export_drone_detector_rknn()`

That module implements a custom MobileNetV3-small student detector intended for edge deployment, but it is **not wired into the local runner yet**.
Treat it as an experimental training API until Step 30 is explicitly migrated to call it.

---

## Dataset

**Source:** [`lgrzybowski/seraphim-drone-detection-dataset`](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset)

The dataset provides drone images in YOLO format (zip archives).
The step downloads `train/images/batch_001.zip` and `train/labels/batch_001.zip`
(≈ 400 images), which is enough for a demonstration run.
Extracted files are cached at `.data/local_runs/_drone_detection_cache/` and
reused on re-runs.

### Using more data

To use the full dataset (batch 001–004) for a production-quality model, download
the additional batches manually and place them in the cache directory:

```
.data/local_runs/_drone_detection_cache/
  train_images/   ← all images from batch_001 + extras you add
  train_labels/   ← matching YOLO .txt label files
```

The step picks up whatever images are already in `train_images/` before starting
training, so you can pre-populate it with any additional YOLO-format drone images.

### Adding your own drone images

Place YOLO-format image files in `train_images/` and matching `.txt` label files
(one box per line: `class cx cy w h`, normalised 0–1) in `train_labels/` before
running. The step will include them automatically.

---

## Hard negative injection

The step automatically selects up to 80 frames from the current mission (extracted
at step 1) and adds them to the training set as **hard negatives** — images with
no bounding boxes. These teach the model that sky, roads, buildings, and foliage
are not drones, reducing false positives in the deployment environment.

No configuration is needed; hard negative injection is always active.

---

## Augmentation strategy

| Augmentation | Setting | Purpose |
|---|---|---|
| Mosaic | p=1.0 | Multi-image composition exposes partial drones at frame edges |
| MixUp | p=0.05 | Blended backgrounds prevent over-confidence on single drone appearances |
| Copy-paste | p=0.10 | Pastes drone crops onto hard-negative backgrounds; directly targets FN rate |
| Scale | ±30% | Handles altitude variation (high-altitude = small drone) |
| FlipLR | 50% | Direction invariance |
| Degrees | ±5° | Mild rotation for gimbal-stabilised video |

---

## Outputs

All outputs are written to `.data/local_runs/{video_name}/drone_detection/`.

```
drone_detection/
  dataset/
    data.yaml
    train/images/     mission frames (hard negatives) + seraphim images
    train/labels/
    val/images/
    val/labels/
  runs/train/
    weights/
      best.pt         best YOLOv8n checkpoint
      last.pt
    results.csv       per-epoch metrics
  exports/
    drone_yolo8n_a76.onnx          fp32 ONNX — Cortex-A76
    drone_yolo8n_rv1106_int8.onnx  int8 ONNX — RV1106G3 fallback
    drone_yolo8n_rv1106.rknn       (optional) RKNN NPU model
  test_a76.py         inference test script for Cortex-A76
  test_rv1106.py      inference test script for RV1106G3
  drone_detection_report.md        training summary + edge deployment guide
```

A cross-run model advisor report is also updated at
`.data/local_runs/model_run_advisor.md` and
`.data/local_runs/model_run_advisor.json` with an edge deployment section.

---

## Running inference on edge hardware

### Arm Cortex-A76

```bash
pip install onnxruntime          # or onnxruntime-gpu on CUDA machines
cd .data/local_runs/{video_name}/drone_detection/
python test_a76.py path/to/test_frame.jpg
```

Expected latency: **25-50 ms** @ 640×640 on a single Cortex-A76 thread.

### Rockchip RV1106G3 — int8 ONNX fallback (no NPU)

If `rknn-toolkit2` was not installed at training time, use the int8 ONNX model
with `onnxruntime` on the RV1106G3's Cortex-A7 cores:

```bash
pip install onnxruntime
python test_rv1106.py path/to/test_frame.jpg
```

Expected latency: **80-150 ms** on Cortex-A7 (slower than NPU path).

### Rockchip RV1106G3 — RKNN NPU (requires conversion)

If `rknn-toolkit2` is installed, the `.rknn` model is generated automatically.
Copy it to the device and run with `rknn-toolkit-lite2`:

```bash
# On the RV1106G3 device:
pip install rknn-toolkit-lite2    # available from Airockchip releases
python test_rv1106.py path/to/test_frame.jpg
```

Expected latency: **8-15 ms** on the RV1106G3 NPU.

### Converting ONNX to RKNN offline (x86 workstation)

```bash
pip install rknn-toolkit2         # see github.com/airockchip/rknn-toolkit2
python - <<'EOF'
from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
            target_platform='rv1106')
rknn.load_onnx('drone_yolo8n_rv1106_int8.onnx')
rknn.build(do_quantization=True)
rknn.export_rknn('drone_yolo8n_rv1106.rknn')
rknn.release()
EOF
```

---

## Reading the report

`drone_detection_report.md` contains:

- Training data breakdown (seraphim images + hard negatives)
- Augmentation strategy rationale
- Training metrics table: mAP@50, mAP@50-95, final box loss
- Per-target model sizes and estimated latencies
- Shell commands for RKNN conversion and on-device testing

---

## Tuning

### More training epochs

The default is 5 epochs (demonstration mode). For production quality:

```python
# in steps_drone_detection.py
_TRAIN_EPOCHS = 30  # or 50 for full convergence
```

Monitor `runs/train/results.csv` — training is done when `metrics/mAP50(B)` stops
improving across three consecutive epochs.

### Larger model

YOLOv8n (nano, ~3 MB) is the default. For higher accuracy at the cost of
inference speed:

```python
# in steps_drone_detection.py
model = YOLO("yolov8s.pt")  # small: ~11 MB, ~2× slower
model = YOLO("yolov8m.pt")  # medium: ~26 MB, ~4× slower
```

YOLOv8n fits in the RV1106G3 NPU at int8. Larger models may not.

### Confidence threshold

The test scripts use `CONF_THRESH = 0.25`. To reduce false positives at the cost
of higher false negative rate, increase this:

```python
CONF_THRESH = 0.40  # in test_a76.py / test_rv1106.py
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `HuggingFace download failed` | No internet / rate-limit | Check `~/.cache/huggingface/`; pre-download the zip files and place in cache |
| `No training images available — drone detection step skipped` | Empty `train_images/` + download failed | Pre-populate `_drone_detection_cache/train_images/` manually |
| `YOLOv8n training failed: ultralytics not installed` | Missing dep | `pip install ultralytics` |
| ONNX export fails | ultralytics ONNX export error | Check opset compatibility; try `opset=12` |
| `rknn-toolkit2 not found — RKNN skipped` | Normal on most machines | Install `rknn-toolkit2` from Airockchip releases to enable NPU export |
| mAP@50 < 0.30 after 5 epochs | Too few images | Download batch_002–004 to expand training set |
| High false-positive rate on sky / buildings | Insufficient hard negatives | Add more mission frames with `_MAX_NEGATIVES = 150` in `steps_drone_detection.py` |

---

## Related

- [`ssv_vdp/steps/drone_detection.py`](../../packages/ss-fusion/src/ssv_vdp/steps/drone_detection.py)
- [`ssv_vdp/steps/model_advisor.py`](../../packages/ss-fusion/src/ssv_vdp/steps/model_advisor.py) — edge section of the advisor report
- [seraphim dataset](https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset)
- [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- Step 29: [knowledge distillation](../learning_path/06_adaptation_eval_steps_28_35.md#step-29-knowledge-distillation) — context for edge model compression
- Step 31: [ONNX export and gallery build](../learning_path/06_adaptation_eval_steps_28_35.md#step-31-onnx-export-and-gallery-build) — DINO backbone ONNX export (separate from drone detection export)
