"""Step 30: Drone-detection model training for edge deployment.

Downloads a subset of the seraphim-drone-detection-dataset from HuggingFace,
injects hard-negative samples from the current mission, trains YOLOv8n, and
exports artefacts for two edge targets:

  Arm Cortex-A76  — ONNX fp32 (onnxruntime-arm64)
  Rockchip RV1106G3 — ONNX int8-quantised; optional RKNN if toolkit is present

Outputs (all under video_dir/drone_detection/):
  dataset/          YOLO-format dataset (images + labels + data.yaml)
  runs/train/       ultralytics training output
  exports/
    drone_yolo8n_a76.onnx          fp32 ONNX for Cortex-A76
    drone_yolo8n_rv1106_int8.onnx  int8-quantised ONNX for RV1106G3
    drone_yolo8n_rv1106.rknn       (optional) RKNN model
  test_a76.py        inference test script for Cortex-A76
  test_rv1106.py     inference test script / RV1106G3 RKNN
  drone_detection_report.md        training summary + edge-deployment notes
"""

import os
import shutil
import textwrap
import time
import zipfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from .common import write_markdown_artifact

_log = get_logger("pipeline.local.drone_detection")

# Maximum images we download from the HF dataset for the demo training run.
_MAX_TRAIN_IMAGES = 400
_MAX_VAL_IMAGES = 100
# Negative-sample frames injected from the current mission.
_MAX_NEGATIVES = 80
# Quick training: 3 epochs + 1 val epoch is enough to show the workflow.
_TRAIN_EPOCHS = 5
_IMG_SIZE = 640
_HF_REPO = "lgrzybowski/seraphim-drone-detection-dataset"
_YOLOV8N_MODEL = "yolov8n.pt"
_ULTRALYTICS_AUX_MODELS = ("yolov8n.pt", "yolo26n.pt")


# -- Dataset helpers -----------------------------------------------------------


def _download_batch_zip(
    repo_id: str,
    zip_path: str,
    cache_dir: Path,
) -> Path | None:
    """Download one zip from a HuggingFace dataset repo. Returns local path."""
    local = cache_dir / zip_path.replace("/", "_")
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=zip_path,
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        return Path(downloaded)
    except Exception as exc:
        _log.warning("HuggingFace download failed (%s): %s", zip_path, exc)
        return None


def _extract_zip(zip_path: Path, dest: Path, max_files: int) -> list[Path]:
    """Extract up to max_files entries from a zip, return extracted paths."""
    extracted: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")][:max_files]
            for member in members:
                target = dest / Path(member).name
                if not target.exists():
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                extracted.append(target)
    except Exception as exc:
        _log.warning("Failed to extract %s: %s", zip_path, exc)
    return extracted


def _build_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    neg_frames: list[Path],
    dataset_dir: Path,
) -> Path | None:
    """Assemble a YOLO-format dataset directory; return path to data.yaml."""
    for split in ("train/images", "train/labels", "val/images", "val/labels"):
        (dataset_dir / split).mkdir(parents=True, exist_ok=True)

    img_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not img_paths:
        _log.warning("No images found in %s", images_dir)
        return None

    # 80/20 train/val split
    split_idx = max(1, int(len(img_paths) * 0.8))
    train_imgs, val_imgs = img_paths[:split_idx], img_paths[split_idx:]

    def _copy_pair(src_img: Path, split: str) -> None:
        dst_img = dataset_dir / split / "images" / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        stem = src_img.stem
        src_lbl = labels_dir / (stem + ".txt")
        dst_lbl = dataset_dir / split / "labels" / (stem + ".txt")
        if not dst_lbl.exists():
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("")  # hard negative: no annotations

    for img in train_imgs:
        _copy_pair(img, "train")
    for img in val_imgs:
        _copy_pair(img, "val")

    # Add mission frames as hard negatives (empty label files).
    for neg in neg_frames[:_MAX_NEGATIVES]:
        dst = dataset_dir / "train" / "images" / neg.name
        if not dst.exists():
            shutil.copy2(neg, dst)
        lbl = dataset_dir / "train" / "labels" / (neg.stem + ".txt")
        if not lbl.exists():
            lbl.write_text("")

    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: drone\n",
        encoding="utf-8",
    )
    return yaml_path


# -- Training ------------------------------------------------------------------


def _configure_ultralytics_cache() -> None:
    """Point ultralytics weights_dir at the selfsuvis cache so downloads never land in CWD."""
    try:
        from ultralytics.utils import SETTINGS

        cache_dir = str(_ultralytics_cache_dir())
        if SETTINGS.get("weights_dir") != cache_dir:
            SETTINGS.update({"weights_dir": cache_dir})
    except Exception:
        pass


def _train_yolov8n(yaml_path: Path, run_dir: Path, device: str) -> dict[str, Any]:
    """Train YOLOv8n on the prepared dataset. Returns training metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"error": "ultralytics not installed"}

    _configure_ultralytics_cache()
    _relocate_repo_root_ultralytics_artifacts()
    model = YOLO(str(_ultralytics_cached_model_path(_YOLOV8N_MODEL)))
    torch_device = "0" if device == "cuda" else "cpu"
    results = model.train(
        data=str(yaml_path),
        epochs=_TRAIN_EPOCHS,
        imgsz=_IMG_SIZE,
        batch=8 if device == "cuda" else 4,
        device=torch_device,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        verbose=False,
        # Augmentations that reduce false positives on sky/urban backgrounds
        augment=True,
        degrees=5.0,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.1,
        save=True,
        plots=False,
    )
    _relocate_repo_root_ultralytics_artifacts()
    best_pt = run_dir / "train" / "weights" / "best.pt"
    metrics: dict[str, Any] = {"best_pt": str(best_pt) if best_pt.exists() else ""}
    try:
        # ultralytics Results object exposes box metrics
        if hasattr(results, "results_dict"):
            rd = results.results_dict
            metrics["map50"] = float(rd.get("metrics/mAP50(B)", 0.0))
            metrics["map50_95"] = float(rd.get("metrics/mAP50-95(B)", 0.0))
            metrics["box_loss"] = float(rd.get("train/box_loss", float("nan")))
    except Exception:
        pass
    return metrics


def _relocate_repo_root_ultralytics_artifacts() -> None:
    """Move stray Ultralytics weight downloads from cwd into the cache dir.

    Ultralytics sometimes drops helper weights like ``yolo26n.pt`` into the
    process cwd during AMP checks. They are cache artifacts, not project files.
    """
    cache_dir = _ultralytics_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd()
    for model_name in _ULTRALYTICS_AUX_MODELS:
        src = repo_root / model_name
        dst = cache_dir / model_name
        if not src.exists() or src.resolve() == dst.resolve():
            continue
        if dst.exists():
            src.unlink(missing_ok=True)
            continue
        shutil.move(str(src), str(dst))


# -- Export helpers ------------------------------------------------------------


def _export_onnx_fp32(best_pt: Path, export_dir: Path) -> Path | None:
    """Export best.pt → ONNX fp32. Returns ONNX path."""
    out = export_dir / "drone_yolo8n_a76.onnx"
    if out.exists():
        return out
    try:
        from ultralytics import YOLO

        m = YOLO(str(best_pt))
        exported = m.export(
            format="onnx",
            imgsz=_IMG_SIZE,
            simplify=find_spec("onnxslim") is not None,
            opset=17,
        )
        src = Path(str(exported))
        if src.exists():
            shutil.move(str(src), str(out))
        return out if out.exists() else None
    except Exception as exc:
        _log.warning("ONNX fp32 export failed: %s", exc)
        return None


def _ultralytics_cached_model_path(model_name: str) -> Path:
    """Return the canonical ultralytics cache path for *model_name*."""
    return _ultralytics_cache_dir() / model_name


def _ultralytics_cache_dir() -> Path:
    data_dir = Path(getattr(settings, "DATA_DIR", "./.data"))
    return Path(os.getenv("CACHE_DIR", str(data_dir / ".cache"))) / "ultralytics"


def _quantize_onnx_int8(onnx_fp32: Path, export_dir: Path) -> Path | None:
    """Quantize fp32 ONNX to int8 dynamic quantisation. Returns int8 path."""
    out = export_dir / "drone_yolo8n_rv1106_int8.onnx"
    if out.exists():
        return out
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(onnx_fp32),
            model_output=str(out),
            weight_type=QuantType.QInt8,
        )
        return out if out.exists() else None
    except Exception as exc:
        _log.warning("ONNX int8 quantisation failed: %s", exc)
        return None


def _try_rknn_export(onnx_path: Path, export_dir: Path) -> Path | None:
    """Try to convert ONNX → RKNN for RV1106G3. Skips gracefully if toolkit absent."""
    out = export_dir / "drone_yolo8n_rv1106.rknn"
    if out.exists():
        return out
    try:
        from rknn.api import RKNN  # type: ignore[import]
    except ImportError:
        return None
    try:
        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rv1106")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            return None
        ret = rknn.build(do_quantization=True)
        if ret != 0:
            return None
        ret = rknn.export_rknn(str(out))
        rknn.release()
        return out if (ret == 0 and out.exists()) else None
    except Exception as exc:
        _log.warning("RKNN export failed: %s", exc)
        return None


# -- Test-script generation ----------------------------------------------------


def _write_test_a76(dest: Path, onnx_path: Path) -> None:
    script = textwrap.dedent(f"""
        \"\"\"Inference test for Arm Cortex-A76 using ONNX Runtime.
        Install:  pip install onnxruntime  (or onnxruntime-gpu on CUDA machines)
        Usage:    python test_a76.py <image_path>
        \"\"\"
        import sys
        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        MODEL = "{onnx_path.name}"
        IMG_SIZE = {_IMG_SIZE}
        CONF_THRESH = 0.25
        IOU_THRESH  = 0.45

        def preprocess(img_path, size=IMG_SIZE):
            img = Image.open(img_path).convert("RGB").resize((size, size))
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.transpose(2, 0, 1)[None]  # NCHW

        def xywh2xyxy(x):
            out = np.zeros_like(x)
            out[..., 0] = x[..., 0] - x[..., 2] / 2
            out[..., 1] = x[..., 1] - x[..., 3] / 2
            out[..., 2] = x[..., 0] + x[..., 2] / 2
            out[..., 3] = x[..., 1] + x[..., 3] / 2
            return out

        sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name

        img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
        inp = preprocess(img_path)
        outputs = sess.run(None, {{inp_name: inp}})
        # YOLOv8 output shape: (1, 5+nc, 8400)
        pred = outputs[0][0].T  # (8400, 5+nc)
        scores = pred[:, 4]
        keep = scores > CONF_THRESH
        pred = pred[keep]
        if len(pred) == 0:
            print("No drones detected.")
        else:
            boxes = xywh2xyxy(pred[:, :4])
            for i, (box, sc) in enumerate(zip(boxes, pred[:, 4])):
                print(f"Drone {{i}}: conf={{sc:.3f}}  box={{box.tolist()}}")
    """).lstrip()
    dest.write_text(script, encoding="utf-8")


def _write_test_rv1106(dest: Path, rknn_available: bool, onnx_int8_path: Path) -> None:
    if rknn_available:
        script = textwrap.dedent(f"""
            \"\"\"Inference test for Rockchip RV1106G3 using RKNN Runtime.
            Requires rknn-toolkit-lite2 on the RV1106 device.
            Usage:   python test_rv1106.py <image_path>
            \"\"\"
            import sys
            import numpy as np
            from rknnlite.api import RKNNLite
            from PIL import Image

            MODEL   = "drone_yolo8n_rv1106.rknn"
            SIZE    = {_IMG_SIZE}
            CONF_TH = 0.25

            rknn_lite = RKNNLite()
            rknn_lite.load_rknn(MODEL)
            rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

            img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
            img = np.array(Image.open(img_path).convert("RGB").resize((SIZE, SIZE)), dtype=np.uint8)
            outputs = rknn_lite.inference(inputs=[img])
            pred = outputs[0][0].T
            keep = pred[:, 4] > CONF_TH
            print(f"Drones detected: {{keep.sum()}}")
            rknn_lite.release()
        """).lstrip()
    else:
        script = textwrap.dedent(f"""
            \"\"\"Inference test for Rockchip RV1106G3 using int8 ONNX model.

            rknn-toolkit2 was not found at training time, so the RKNN model was not
            generated. Use the int8 ONNX model with onnxruntime on the RV1106G3's
            A7 cores, or convert offline:

              pip install rknn-toolkit2  # register at https://github.com/airockchip/rknn-toolkit2
              python -c "
              from rknn.api import RKNN
              rknn = RKNN(); rknn.config(target_platform='rv1106')
              rknn.load_onnx('{onnx_int8_path.name}')
              rknn.build(do_quantization=True)
              rknn.export_rknn('drone_yolo8n_rv1106.rknn')
              "

            INT8 ONNX fallback (runs on Cortex-A7 via onnxruntime):
            \"\"\"
            import sys
            import numpy as np
            import onnxruntime as ort
            from PIL import Image

            MODEL = "{onnx_int8_path.name}"
            SIZE  = {_IMG_SIZE}

            sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
            img = Image.open(img_path).convert("RGB").resize((SIZE, SIZE))
            arr = np.array(img, dtype=np.float32)[None] / 255.0
            arr = arr.transpose(0, 3, 1, 2)
            out = sess.run(None, {{inp_name: arr}})
            pred = out[0][0].T
            n_det = int((pred[:, 4] > 0.25).sum())
            print(f"Drones detected (int8 ONNX fallback): {{n_det}}")
        """).lstrip()
    dest.write_text(script, encoding="utf-8")


# -- Report --------------------------------------------------------------------


def _write_report(
    report_path: Path,
    metrics: dict[str, Any],
    n_train: int,
    n_neg: int,
    onnx_fp32: Path | None,
    onnx_int8: Path | None,
    rknn_path: Path | None,
    elapsed: float,
) -> None:
    lines = [
        "# Drone Detection — Edge Model Training Report",
        "",
        f"Dataset: `{_HF_REPO}` (batch_001 subset + mission negatives)",
        f"Model: YOLOv8n  |  imgsz={_IMG_SIZE}  |  epochs={_TRAIN_EPOCHS}",
        "",
        "## Training Data",
        "",
        "| Split | Images |",
        "|-------|--------|",
        f"| Train (seraphim) | {n_train} |",
        f"| Hard negatives (mission frames) | {n_neg} |",
        "",
        "### Augmentation strategy",
        "",
        "- **Mosaic** (p=1.0): combines 4 images — exposes the model to partial drones at edges",
        "- **MixUp** (p=0.05): blends two images — reduces over-fitting to canonical drone poses",
        "- **Copy-paste** (p=0.1): pastes drone crops onto negatives — directly targets false negatives",
        "- **Scale** ±30%: handles altitude variation",
        "- **FlipLR** 50%: direction invariance",
        "- **Degrees** ±5°: mild rotation for gimbal-stabilised video",
        "- **Mission hard negatives**: sky crops, roads, foliage from the current run  →  reduces urban/sky FPs",
        "",
        "## Training Metrics",
        "",
    ]
    if "error" in metrics:
        lines += [f"Training failed: `{metrics['error']}`", ""]
    else:
        map50 = metrics.get("map50", float("nan"))
        map50_95 = metrics.get("map50_95", float("nan"))
        box_loss = metrics.get("box_loss", float("nan"))
        import math as _math

        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| mAP@50 | {map50:.4f} |" if not _math.isnan(map50) else "| mAP@50 | n/a |",
            f"| mAP@50-95 | {map50_95:.4f} |"
            if not _math.isnan(map50_95)
            else "| mAP@50-95 | n/a |",
            f"| Final box loss | {box_loss:.4f} |"
            if not _math.isnan(box_loss)
            else "| Final box loss | n/a |",
            f"| Train time | {elapsed:.1f}s |",
            "",
        ]

    lines += [
        "## Edge Targets",
        "",
        "### Arm Cortex-A76 (e.g. Raspberry Pi 5, NVIDIA Orin A76 cores)",
        "",
        f"- **Model**: `{onnx_fp32.name if onnx_fp32 else 'export failed'}`",
        "- **Runtime**: `onnxruntime` (CPU provider)",
        "- **Precision**: fp32",
        "- **Estimated latency**: ~25-50 ms @ 640×640 on Cortex-A76 (1 thread)",
        "- **Test script**: `test_a76.py`",
        "",
        "```bash",
        "pip install onnxruntime",
        "python test_a76.py frame_0000.jpg",
        "```",
        "",
        "### Rockchip RV1106G3 (e.g. Luckfox Pico, IPC-AI modules)",
        "",
        f"- **int8 ONNX model**: `{onnx_int8.name if onnx_int8 else 'quantisation failed'}`",
        f"- **RKNN model**: `{'generated [ok]' if rknn_path and rknn_path.exists() else 'not generated — install rknn-toolkit2'}`",
        "- **NPU**: Rockchip 0.5 TOPS NPU via rknn-toolkit-lite2 on device",
        "- **Estimated latency**: ~8-15 ms on RV1106G3 NPU @ int8",
        "- **Test script**: `test_rv1106.py`",
        "",
        "#### Converting to RKNN offline",
        "",
        "```bash",
        "# On x86 workstation (not on device):",
        "pip install rknn-toolkit2  # see https://github.com/airockchip/rknn-toolkit2",
        'python -c "',
        "from rknn.api import RKNN",
        "rknn = RKNN()",
        "rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rv1106')",
        f"rknn.load_onnx('{onnx_int8.name if onnx_int8 else 'drone_yolo8n_rv1106_int8.onnx'}')",
        "rknn.build(do_quantization=True)",
        "rknn.export_rknn('drone_yolo8n_rv1106.rknn')",
        "rknn.release()",
        '"',
        "",
        "# On device (RV1106G3), install rknn-toolkit-lite2 and run test_rv1106.py",
        "```",
        "",
        "## False-Positive Reduction",
        "",
        "| Source | Technique | Effect |",
        "|--------|-----------|--------|",
        "| Mission frames (sky, roads, foliage) | Hard negative injection | Reduces scene-specific FPs |",
        "| MixUp augmentation | Blended backgrounds | Prevents over-confidence on single-appearance drones |",
        "| Copy-paste augmentation | Synthetic occluded drones | Reduces FNs in cluttered scenes |",
        "| Scale augmentation ±30% | Multi-scale exposure | Handles altitude variation, reduces size-bias FPs |",
        "",
        "## Next Steps",
        "",
        "1. **More data**: download `batch_002`-`batch_004` zips for full dataset coverage",
        "2. **Fine-tune negatives**: curate sky-only crops and high-FP frames from deployment logs",
        "3. **Quantisation-aware training (QAT)**: use `model.export(format='onnx', int8=True)` with a calibration set",
        "4. **On-device benchmark**: run `rknn_benchmark` on the RV1106G3 to measure real NPU latency",
        "5. **Integration**: call `EdgeClassifier` from `selfsuvis.pipeline.inference.edge` with the exported ONNX",
    ]
    write_markdown_artifact(report_path, lines)


# -- Public step function ------------------------------------------------------


def step_drone_detection_training(
    frame_list: list[tuple[str, float]],
    video_name: str,
    video_dir: Path,
    output_dir: Path,
    device: str,
    args: Any,
) -> dict[str, Any]:
    """Train YOLOv8n drone detector; export ONNX fp32 + int8; optional RKNN."""
    result: dict[str, Any] = {
        "skipped": False,
        "model_fp32": "",
        "model_int8": "",
        "model_rknn": "",
        "n_train_images": 0,
        "n_negatives": 0,
        "map50": float("nan"),
    }
    t0 = time.monotonic()

    drone_dir = video_dir / "drone_detection"
    dataset_dir = drone_dir / "dataset"
    run_dir = drone_dir / "runs"
    export_dir = drone_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Shared HF cache across all videos in this run
    hf_cache = output_dir / "_drone_detection_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    _log.info("Downloading seraphim-drone-detection batch_001 …")
    img_zip = _download_batch_zip(_HF_REPO, "train/images/batch_001.zip", hf_cache)
    lbl_zip = _download_batch_zip(_HF_REPO, "train/labels/batch_001.zip", hf_cache)

    images_dir = hf_cache / "train_images"
    labels_dir = hf_cache / "train_labels"
    if img_zip:
        _log.info("Extracting training images (max %d) …", _MAX_TRAIN_IMAGES)
        _extract_zip(img_zip, images_dir, _MAX_TRAIN_IMAGES)
    if lbl_zip:
        _extract_zip(lbl_zip, labels_dir, _MAX_TRAIN_IMAGES)

    n_imgs = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
    if n_imgs == 0:
        _log.warning("No training images available — drone detection step skipped")
        result["skipped"] = True
        result["error"] = "HuggingFace dataset download failed"
        return result

    # Select negative-sample frames from the current mission
    neg_frames = [Path(fp) for fp, _ in frame_list[:_MAX_NEGATIVES]]
    n_neg = len(neg_frames)

    _log.info("Building YOLO dataset (%d seraphim + %d mission negatives) …", n_imgs, n_neg)
    yaml_path = _build_yolo_dataset(images_dir, labels_dir, neg_frames, dataset_dir)
    if yaml_path is None:
        result["skipped"] = True
        result["error"] = "Dataset assembly failed"
        return result

    n_train = len(list((dataset_dir / "train" / "images").glob("*")))
    result["n_train_images"] = n_train
    result["n_negatives"] = n_neg

    _log.info(
        "Training YOLOv8n (%d epochs, imgsz=%d, device=%s) …", _TRAIN_EPOCHS, _IMG_SIZE, device
    )
    metrics = _train_yolov8n(yaml_path, run_dir, device)
    if "error" in metrics:
        _log.warning("YOLOv8n training failed: %s", metrics["error"])
        result["error"] = metrics["error"]
        # Still write a partial report
        _write_report(
            drone_dir / "drone_detection_report.md",
            metrics,
            n_train,
            n_neg,
            None,
            None,
            None,
            time.monotonic() - t0,
        )
        return result

    result["map50"] = metrics.get("map50", float("nan"))
    result["map50_95"] = metrics.get("map50_95", float("nan"))
    result["box_loss"] = metrics.get("box_loss", float("nan"))

    best_pt = Path(metrics.get("best_pt", ""))
    onnx_fp32: Path | None = None
    onnx_int8: Path | None = None
    rknn_path: Path | None = None

    if best_pt.exists():
        _log.info("Exporting ONNX fp32 for Cortex-A76 …")
        onnx_fp32 = _export_onnx_fp32(best_pt, export_dir)
        if onnx_fp32:
            result["model_fp32"] = str(onnx_fp32)
            _log.info("  [ok] %s (%.1f MB)", onnx_fp32.name, onnx_fp32.stat().st_size / 1e6)

            _log.info("Quantising ONNX to int8 for RV1106G3 …")
            onnx_int8 = _quantize_onnx_int8(onnx_fp32, export_dir)
            if onnx_int8:
                result["model_int8"] = str(onnx_int8)
                _log.info("  [ok] %s (%.1f MB)", onnx_int8.name, onnx_int8.stat().st_size / 1e6)

            _log.info("Attempting RKNN export (skips if rknn-toolkit2 absent) …")
            src_for_rknn = onnx_int8 or onnx_fp32
            rknn_path = _try_rknn_export(src_for_rknn, export_dir)
            if rknn_path:
                result["model_rknn"] = str(rknn_path)
                _log.info("  [ok] RKNN model: %s", rknn_path)
            else:
                _log.info("  [info] RKNN skipped (install rknn-toolkit2 for RV1106G3 NPU)")

    _log.info("Writing test scripts …")
    _write_test_a76(drone_dir / "test_a76.py", onnx_fp32 or export_dir / "drone_yolo8n_a76.onnx")
    _write_test_rv1106(
        drone_dir / "test_rv1106.py",
        rknn_available=rknn_path is not None,
        onnx_int8_path=onnx_int8 or export_dir / "drone_yolo8n_rv1106_int8.onnx",
    )

    report_path = drone_dir / "drone_detection_report.md"
    _write_report(
        report_path, metrics, n_train, n_neg, onnx_fp32, onnx_int8, rknn_path, time.monotonic() - t0
    )
    _log.info("  [ok] Written %s", report_path)

    result["elapsed_sec"] = time.monotonic() - t0
    return result
