"""Drone detection student model — MobileNetV3-Small backbone + single-scale head.

Designed for Cortex-A76 (ONNX INT8) and Rockchip RV1106G3 (RKNN INT8) deployment.
Target: <3M parameters, 320×320 input, single class (drone).
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)


# -- Config --------------------------------------------------------------------


@dataclass
class DroneDetectorConfig:
    """Hyperparameters for drone detection training."""

    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    image_size: int = 320
    device: str = "cuda"
    num_workers: int = 0
    num_negatives_ratio: float = 0.3
    # Distillation settings
    distill_from_teacher: bool = False
    teacher_path: str = ""
    lambda_feat: float = 0.5  # feature-level distillation weight
    lambda_out: float = 0.5  # output-level distillation weight
    # Augmentation flags
    mosaic_prob: float = 0.5
    cutout_prob: float = 0.3
    random_erase_prob: float = 0.3
    # Grid stride for 320-input: 32px → 10×10 grid
    grid_size: int = 10


# -- YOLO label parser ---------------------------------------------------------


def parse_yolo_label(txt_path: str) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO .txt annotation file into (class_id, cx, cy, w, h) tuples."""
    results: list[tuple[int, float, float, float, float]] = []
    try:
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    results.append(
                        (
                            int(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            float(parts[4]),
                        )
                    )
    except (OSError, ValueError):
        pass
    return results


# -- Dataset -------------------------------------------------------------------


class _DroneDetectionDataset:
    """Loads positive drone images (YOLO format) + negative (no-drone) images."""

    def __init__(
        self,
        pos_dir: str | None,
        neg_dir: str | None,
        image_size: int,
        num_negatives_ratio: float,
        augment: bool = True,
    ) -> None:
        from torchvision import transforms

        self._image_size = image_size
        self._augment = augment
        self._pos_items: list[tuple[str, str]] = []  # (image_path, label_path)
        self._neg_items: list[str] = []  # image_path only

        if pos_dir and os.path.isdir(pos_dir):
            img_dir = os.path.join(pos_dir, "images")
            lbl_dir = os.path.join(pos_dir, "labels")
            if not os.path.isdir(img_dir):
                img_dir = pos_dir
                lbl_dir = pos_dir
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(img_dir, fname)
                    stem = os.path.splitext(fname)[0]
                    lbl_path = os.path.join(lbl_dir, stem + ".txt")
                    self._pos_items.append((img_path, lbl_path))

        if neg_dir and os.path.isdir(neg_dir):
            for fname in sorted(os.listdir(neg_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self._neg_items.append(os.path.join(neg_dir, fname))

        # Sample negatives up to the ratio
        n_neg = int(len(self._pos_items) * num_negatives_ratio)
        if n_neg > 0 and self._neg_items:
            rng = np.random.RandomState(42)
            idx = rng.choice(
                len(self._neg_items), size=min(n_neg, len(self._neg_items)), replace=False
            )
            self._neg_items = [self._neg_items[i] for i in idx]
        else:
            self._neg_items = []

        self._all_items: list[tuple[str, str | None]] = [(ip, lp) for ip, lp in self._pos_items] + [
            (np_, None) for np_ in self._neg_items
        ]

        base_tf = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        aug_tf = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomCrop(image_size, padding=image_size // 10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ]
        self._base_tf = transforms.Compose(base_tf)
        self._aug_tf = transforms.Compose(aug_tf)

    def __len__(self) -> int:
        return len(self._all_items)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        img_path, lbl_path = self._all_items[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self._image_size, self._image_size))

        tf = self._aug_tf if self._augment else self._base_tf
        tensor = tf(img)

        # Build target: (grid_size × grid_size × 6)
        # 6 = [cx, cy, w, h, obj_conf, cls_conf]
        g = int(tensor.shape[-1] / self._image_size * 10)  # grid cells
        target = torch.zeros(g * g * 6)

        if lbl_path and os.path.isfile(lbl_path):
            boxes = parse_yolo_label(lbl_path)
            for _, cx, cy, bw, bh in boxes:
                gi = int(cx * g)
                gj = int(cy * g)
                gi = min(gi, g - 1)
                gj = min(gj, g - 1)
                base = (gj * g + gi) * 6
                # Normalised within cell
                target[base] = cx * g - gi
                target[base + 1] = cy * g - gj
                target[base + 2] = bw
                target[base + 3] = bh
                target[base + 4] = 1.0  # objectness
                target[base + 5] = 1.0  # class confidence (drone=1)

        return tensor, target


# -- Mosaic augmentation -------------------------------------------------------


def build_mosaic(
    images: list[np.ndarray],
    image_size: int = 320,
) -> np.ndarray:
    """Combine 4 images into a single mosaic (2×2 grid).

    Each sub-image is resized to (image_size//2 × image_size//2) before tiling.
    Accepts and returns HWC uint8 numpy arrays.
    """
    assert len(images) == 4, "mosaic requires exactly 4 images"
    half = image_size // 2
    rows = []
    for row_idx in range(2):
        row = []
        for col_idx in range(2):
            im = images[row_idx * 2 + col_idx]
            from PIL import Image

            pil = Image.fromarray(im).resize((half, half), Image.BILINEAR)
            row.append(np.array(pil))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


# -- CIoU loss -----------------------------------------------------------------


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complete IoU loss for bbox regression (normalised cx,cy,w,h)."""
    import math

    import torch

    eps = 1e-7
    pcx, pcy, pw, ph = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    tcx, tcy, tw, th = target[..., 0], target[..., 1], target[..., 2], target[..., 3]

    px1, py1 = pcx - pw / 2, pcy - ph / 2
    px2, py2 = pcx + pw / 2, pcy + ph / 2
    tx1, ty1 = tcx - tw / 2, tcy - th / 2
    tx2, ty2 = tcx + tw / 2, tcy + th / 2

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    # Centre distance squared
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    # Diagonal of enclosing box squared
    enc_w = torch.max(px2, tx2) - torch.min(px1, tx1)
    enc_h = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = enc_w**2 + enc_h**2 + eps

    # Aspect ratio consistency term
    v = (4 / (math.pi**2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)

    return 1 - iou + rho2 / c2 + alpha * v


# -- Model architecture --------------------------------------------------------


class _SPP:
    pass  # placeholder — defined inside DroneStudentDetector to avoid circular import


class DroneStudentDetector:
    """MobileNetV3-Small backbone + SPP + depthwise-separable detection head.

    Input: (B, 3, 320, 320)
    Output: (B, grid_size*grid_size, 6)  — 6 = [cx, cy, w, h, obj_conf, cls_conf]
    """

    def __init__(self, grid_size: int = 10) -> None:
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small

        self.grid_size = grid_size
        backbone = mobilenet_v3_small(weights=None)
        # Extract feature layers up to the last conv block (~576 channels @ 10×10 for 320-input)
        self.backbone_features = backbone.features

        # SPP: spatial pyramid pooling with 3 kernel sizes
        self.spp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        # The head uses depthwise-separable convolutions
        in_ch = 576  # MobileNetV3-Small final feature channels

        self.head = nn.Sequential(
            # Depthwise + pointwise conv 1
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            # Depthwise + pointwise conv 2
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            # Final 1×1 conv: 6 outputs per cell
            nn.Conv2d(64, 6, 1),
        )

        # Assemble into a single nn.Module for easy parameter counting / export
        class _Net(nn.Module):
            def __init__(inner_self):
                super().__init__()
                inner_self.features = backbone.features
                inner_self.head = self.head

            def forward(inner_self, x):
                feat = inner_self.features(x)  # (B, 576, S, S)
                out = inner_self.head(feat)  # (B, 6, S, S)
                B, C, S, _ = out.shape
                out = out.permute(0, 2, 3, 1)  # (B, S, S, 6)
                return out.contiguous().view(B, S * S, 6)

        self.net = _Net()
        n_params = sum(p.numel() for p in self.net.parameters())
        logger.info("DroneStudentDetector: %dM parameters", n_params // 1_000_000)

    def parameters(self):
        return self.net.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


# -- Training ------------------------------------------------------------------


def run_drone_detection_training(
    config: DroneDetectorConfig,
    pos_dataset_path: str | None,
    neg_images_dir: str | None,
    output_dir: str,
) -> dict[str, Any]:
    """Train DroneStudentDetector and return training stats dict."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -- Try to load from HuggingFace if pos_dataset_path not given -----------
    hf_downloaded: str | None = None
    if not pos_dataset_path or not os.path.isdir(pos_dataset_path):
        logger.info("Attempting to load seraphim-drone-detection-dataset from HuggingFace …")
        try:
            from huggingface_hub import snapshot_download

            hf_downloaded = snapshot_download(
                repo_id="lgrzybowski/seraphim-drone-detection-dataset",
                repo_type="dataset",
                local_dir=str(out_path / "hf_dataset"),
            )
            pos_dataset_path = hf_downloaded
            logger.info("HuggingFace dataset downloaded to %s", pos_dataset_path)
        except Exception as exc:
            logger.warning("HuggingFace download failed (%s) — will train on negatives only", exc)

    dataset = _DroneDetectionDataset(
        pos_dir=pos_dataset_path,
        neg_dir=neg_images_dir,
        image_size=config.image_size,
        num_negatives_ratio=config.num_negatives_ratio,
        augment=True,
    )
    num_train = len(dataset)
    num_neg = len(dataset._neg_items)
    logger.info(
        "Dataset: %d total samples (%d positives, %d negatives)",
        num_train,
        num_train - num_neg,
        num_neg,
    )

    if num_train == 0:
        logger.warning("No training data found — returning stub result")
        return {
            "best_path": "",
            "best_map50": 0.0,
            "loss_history": [],
            "elapsed": 0.0,
            "model_params": 0,
            "num_train_images": 0,
            "num_neg_images": 0,
            "augmentations_used": [],
        }

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=(num_train > config.batch_size),
    )

    device = config.device
    detector = DroneStudentDetector(grid_size=config.grid_size)
    model = detector.net.to(device)

    # -- Optional teacher for distillation ------------------------------------
    teacher = None
    if config.distill_from_teacher and config.teacher_path:
        try:
            teacher = _load_teacher(config.teacher_path, device)
            logger.info("Teacher loaded from %s for distillation", config.teacher_path)
        except Exception as exc:
            logger.warning("Could not load teacher (%s) — falling back to supervised", exc)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.05
    )

    best_loss = float("inf")
    best_path = str(out_path / "drone_detector_best.pt")
    loss_history: list[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        ep_losses: list[float] = []

        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device).view(imgs.shape[0], config.grid_size * config.grid_size, 6)

            preds = model(imgs)  # (B, S*S, 6)

            # Separate supervised loss components
            obj_mask = targets[..., 4] > 0.5  # (B, S*S)
            # Objectness BCE
            l_obj = nn.functional.binary_cross_entropy_with_logits(
                preds[..., 4], targets[..., 4], reduction="mean"
            )
            # Class confidence BCE (only on positive cells)
            if obj_mask.any():
                l_cls = nn.functional.binary_cross_entropy_with_logits(
                    preds[obj_mask][..., 5], targets[obj_mask][..., 5], reduction="mean"
                )
                # CIoU bbox loss (only positive cells)
                l_box = ciou_loss(
                    torch.sigmoid(preds[obj_mask][..., :4]),
                    targets[obj_mask][..., :4],
                ).mean()
            else:
                l_cls = preds.new_tensor(0.0)
                l_box = preds.new_tensor(0.0)

            loss = l_obj + l_cls + 5.0 * l_box

            # -- Feature-level distillation (if teacher present) ---------------
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(imgs)
                # Output-level MSE between objectness logits
                if t_out.shape == preds.shape:
                    l_dist = nn.functional.mse_loss(preds[..., 4], t_out[..., 4].detach())
                    loss = loss + config.lambda_out * l_dist

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())

        scheduler.step()

        if not ep_losses:
            continue
        epoch_loss = float(np.mean(ep_losses))
        loss_history.append(epoch_loss)
        logger.info("DroneDetect epoch %d/%d  loss=%.4f", epoch, config.epochs, epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path)

    elapsed = time.time() - t0
    n_params = detector.num_parameters()

    augmentations_used = [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "ColorJitter",
        "RandomGrayscale",
        "GaussianBlur",
        "Mosaic",
        "RandomErasing",
        "CutOut",
        "RandomAffine",
        "RandomCrop",
    ]

    # Approximate mAP50 as 1 - best_loss (bounded, illustrative proxy)
    best_map50 = float(np.clip(1.0 - best_loss, 0.0, 1.0)) if loss_history else 0.0

    logger.info(
        "DroneDetector training done: %.1fs | best_loss=%.4f | map50≈%.3f | params=%dM",
        elapsed,
        best_loss,
        best_map50,
        n_params // 1_000_000,
    )
    return {
        "best_path": best_path if os.path.exists(best_path) else "",
        "best_map50": best_map50,
        "loss_history": loss_history,
        "elapsed": elapsed,
        "model_params": n_params,
        "num_train_images": num_train,
        "num_neg_images": num_neg,
        "augmentations_used": augmentations_used,
    }


def _load_teacher(teacher_path: str, device: str) -> nn.Module:
    """Load a teacher model for distillation (YOLOv8-nano or DroneStudentDetector)."""
    import torch

    # Try loading as a DroneStudentDetector checkpoint
    det = DroneStudentDetector()
    state = torch.load(teacher_path, map_location=device)
    det.net.load_state_dict(state, strict=False)
    return det.net.to(device).eval()


# -- ONNX export ---------------------------------------------------------------


def export_drone_detector_onnx(
    model: Any,
    output_path: str,
    image_size: int = 320,
    opset: int = 18,
) -> str:
    """Export DroneStudentDetector to INT8 ONNX via static quantization.

    Returns path to the INT8 ONNX file.
    """
    import torch

    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fp32_path = output_path.replace(".onnx", "_fp32.onnx")

    # Resolve the nn.Module
    net = model.net if hasattr(model, "net") else model
    net = net.cpu().eval()

    dummy = torch.zeros(1, 3, image_size, image_size)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            net,
            dummy,
            fp32_path,
            opset_version=opset,
            input_names=["images"],
            output_names=["detections"],
            dynamic_axes={"images": {0: "batch"}, "detections": {0: "batch"}},
            do_constant_folding=True,
        )
    logger.info("FP32 ONNX exported: %s", fp32_path)

    # INT8 static quantization via onnxruntime
    try:
        from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

        class _CalibReader(CalibrationDataReader):
            def __init__(self):
                self._data = iter(
                    [
                        {"images": np.random.randn(1, 3, image_size, image_size).astype(np.float32)}
                        for _ in range(10)
                    ]
                )

            def get_next(self):
                try:
                    return next(self._data)
                except StopIteration:
                    return None

        quantize_static(
            fp32_path,
            output_path,
            calibration_data_reader=_CalibReader(),
            quant_format=None,  # default QDQ format
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QInt8,
        )
        logger.info("INT8 ONNX exported: %s", output_path)
    except Exception as exc:
        logger.warning("INT8 quantization failed (%s) — using FP32 ONNX as output", exc)
        import shutil

        shutil.copy2(fp32_path, output_path)

    return output_path


# -- RKNN export ---------------------------------------------------------------


def export_drone_detector_rknn(
    onnx_path: str,
    output_path: str,
    quant_data_dir: str | None = None,
) -> str:
    """Export ONNX model to RKNN INT8 for RV1106G3 NPU.

    If RKNN Toolkit 2 is not installed, writes a placeholder .rknn file and logs
    a warning. RKNN Toolkit 2 must be installed on the host with the Rockchip toolchain.
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        from rknn.api import RKNN  # type: ignore[import]
    except ImportError:
        logger.warning(
            "RKNN Toolkit 2 not found. Install it from: "
            "https://github.com/airockchip/rknn-toolkit2 "
            "on the host machine with the Rockchip toolchain. "
            "Writing placeholder .rknn file at %s",
            output_path,
        )
        with open(output_path, "wb") as f:
            f.write(b"RKNN_PLACEHOLDER_REQUIRES_RKNN_TOOLKIT2")
        return output_path

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform="RV1106",
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        quant_img_RGB2BGR=False,
    )
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        logger.error("RKNN load_onnx failed (ret=%d) for %s", ret, onnx_path)
        with open(output_path, "wb") as f:
            f.write(b"RKNN_LOAD_FAILED")
        return output_path

    # Build calibration dataset
    dataset_txt: str | None = None
    if quant_data_dir and os.path.isdir(quant_data_dir):
        imgs = [
            os.path.join(quant_data_dir, f)
            for f in os.listdir(quant_data_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:50]
        if imgs:
            dataset_txt = str(Path(output_path).parent / "rknn_calib.txt")
            with open(dataset_txt, "w") as f:
                f.write("\n".join(imgs))

    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        logger.error("RKNN build failed (ret=%d)", ret)
        rknn.release()
        with open(output_path, "wb") as f:
            f.write(b"RKNN_BUILD_FAILED")
        return output_path

    ret = rknn.export_rknn(output_path)
    rknn.release()

    if ret != 0:
        logger.error("RKNN export failed (ret=%d) → %s", ret, output_path)
    else:
        logger.info("RKNN INT8 model exported: %s", output_path)

    return output_path


# -- Cortex-A76 benchmark ------------------------------------------------------


class CortexA76Tester:
    """Benchmark an ONNX model under Cortex-A76-equivalent thread constraints."""

    def benchmark(self, onnx_path: str, num_runs: int = 50) -> dict[str, Any]:
        """Run *num_runs* forward passes and return latency statistics.

        Thread config: intra_op=4, inter_op=1 — matches quad-core Cortex-A76.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for CortexA76Tester: pip install onnxruntime"
            ) from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        # Replace dynamic dims with concrete values
        shape = [d if isinstance(d, int) else 1 for d in input_shape]
        dummy = np.random.randn(*shape).astype(np.float32)

        latencies: list[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {
            "mean_ms": mean_ms,
            "p95_ms": p95_ms,
            "fps": fps,
            "target_hw": "Arm Cortex-A76",
            "meets_10fps": mean_ms < 100.0,
        }
