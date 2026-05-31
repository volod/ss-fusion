#!/usr/bin/env python
"""Export fine-tuned DINOv3 backbone to ONNX for edge deployment.

Usage:
    # Export with validation
    python scripts/export_onnx.py \\
        --checkpoint .data/checkpoints/dino_ssl_best.pt \\
        --output .data/models/dino_edge.onnx \\
        --validate

    # Export and quantize to INT8
    python scripts/export_onnx.py \\
        --checkpoint .data/checkpoints/dino_ssl_best.pt \\
        --output .data/models/dino_edge.onnx \\
        --quantize \\
        --calibration-dir .data/frames \\
        --calibration-samples 500

    # Use the ONNX model on robot:
    from selfsuvis.pipeline.training.edge_inference import EdgeClassifier
    clf = EdgeClassifier(".data/models/dino_edge_int8.onnx", ".data/gallery/mission_objects.npz")
    results = clf.classify(frame_pil)   # [(label, score), ...]
"""

import argparse
import glob
import os

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)


def _load_backbone(model_name: str, checkpoint: str | None, device: str):
    """Load DINOv3/DINOv2 backbone and optionally restore fine-tuned weights."""
    import torch

    from selfsuvis.models.dino_model import _resolve_dino_hub, hub_load_dino

    _, repo_or_dir, actual_name = _resolve_dino_hub(model_name)
    logger.info("Loading backbone: %s (resolved: %s)", model_name, actual_name)
    backbone = hub_load_dino(model_name, pretrained=True)
    backbone = backbone.to(device)

    if checkpoint:
        logger.info("Loading fine-tuned weights from %s", checkpoint)
        state = torch.load(checkpoint, map_location=device)
        backbone.load_state_dict(state)
        logger.info("Fine-tuned weights loaded.")

    backbone.eval()
    return backbone


def _export_onnx(backbone, output_path: str, image_size: int, opset: int) -> None:
    """Trace and export the backbone to ONNX."""
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size)

    logger.info(
        "Exporting ONNX to %s (opset=%d, image_size=%d) ...", output_path, opset, image_size
    )
    torch.onnx.export(
        backbone,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    logger.info("ONNX export complete: %s", output_path)


def _validate_parity(backbone, onnx_path: str, image_size: int, device: str) -> None:
    """Run a forward pass through both PyTorch and ONNX and assert outputs are close."""
    import numpy as np
    import torch

    try:
        import onnxruntime as ort
    except ImportError as exc:
        logger.warning("onnxruntime not installed — skipping parity validation: %s", exc)
        return

    logger.info("Validating PyTorch ↔ ONNX parity ...")
    dummy_np = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    dummy_pt = torch.tensor(dummy_np).to(device)

    with torch.no_grad():
        pt_out = backbone(dummy_pt).cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: dummy_np})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    logger.info("Max absolute diff PyTorch vs ONNX: %.2e", max_diff)
    if max_diff >= 1e-3:
        raise RuntimeError(
            f"Parity check FAILED: max abs diff {max_diff:.2e} >= 1e-3. "
            "The ONNX model may not reproduce the PyTorch forward pass faithfully."
        )
    logger.info("Parity check PASSED (max diff %.2e < 1e-3).", max_diff)


def _collect_calibration_images(calibration_dir: str, n_samples: int) -> list:
    """Collect up to n_samples image paths from calibration_dir."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths: list = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(calibration_dir, "**", ext), recursive=True))
    paths = sorted(paths)[:n_samples]
    if not paths:
        raise ValueError(f"No images found under calibration dir: {calibration_dir!r}")
    logger.info("Calibration: found %d images (requested %d)", len(paths), n_samples)
    return paths


def _quantize_static(
    onnx_path: str, output_path: str, calibration_paths: list, image_size: int
) -> None:
    """Quantize ONNX model to INT8 using static quantization."""
    try:
        from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
    except ImportError as exc:
        logger.warning(
            "onnxruntime.quantization is not available in this build — skipping INT8 quantization. "
            "Install onnxruntime-tools or a newer onnxruntime build. Error: %s",
            exc,
        )
        return

    from PIL import Image

    from selfsuvis.pipeline.training.edge_inference import _preprocess_image

    class _CalibReader(CalibrationDataReader):
        def __init__(self, paths: list, input_name: str, size: int):
            self._iter = iter(paths)
            self._input_name = input_name
            self._size = size

        def get_next(self):
            try:
                p = next(self._iter)
            except StopIteration:
                return None
            img = Image.open(p).convert("RGB")
            arr = _preprocess_image(img, self._size)
            return {self._input_name: arr}

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    del session

    reader = _CalibReader(calibration_paths, input_name=input_name, size=image_size)

    logger.info("Running static INT8 quantization → %s ...", output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    quantize_static(
        onnx_path,
        output_path,
        calibration_data_reader=reader,
        quant_type=QuantType.QInt8,
    )
    logger.info("INT8 quantization complete: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fine-tuned DINOv3 backbone to ONNX for edge deployment"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to dino_ssl_best.pt (fine-tuned backbone weights). "
        "If omitted, uses pretrained hub weights.",
    )
    parser.add_argument(
        "--model-name",
        default="dinov3_vitb14",
        help="Hub model name (default: dinov3_vitb14)",
    )
    parser.add_argument(
        "--output",
        default=".data/models/dino_edge.onnx",
        help="Output ONNX path (default: .data/models/dino_edge.onnx)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run PyTorch ↔ ONNX forward-pass parity check after export",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also produce an INT8 quantized ONNX model",
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help="Frames directory for quantization calibration (required with --quantize)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=500,
        help="Number of calibration frames to use (default: 500)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for PyTorch export: cpu | cuda (default: cpu)",
    )

    args = parser.parse_args()

    if args.quantize and not args.calibration_dir:
        parser.error("--calibration-dir is required when --quantize is set.")

    print("\nDINOv3 → ONNX export")
    print(f"  checkpoint      : {args.checkpoint or '(pretrained hub weights)'}")
    print(f"  model_name      : {args.model_name}")
    print(f"  output          : {args.output}")
    print(f"  image_size      : {args.image_size}")
    print(f"  opset           : {args.opset}")
    print(f"  validate        : {args.validate}")
    print(f"  quantize        : {args.quantize}")
    if args.quantize:
        print(f"  calibration_dir : {args.calibration_dir}")
        print(f"  calibration_n   : {args.calibration_samples}")
    print(f"  device          : {args.device}")
    print()

    # 1. Load backbone
    backbone = _load_backbone(args.model_name, args.checkpoint, args.device)

    # 2. Export ONNX
    _export_onnx(backbone, args.output, args.image_size, args.opset)

    # 3. Validate parity
    if args.validate:
        _validate_parity(backbone, args.output, args.image_size, args.device)

    # 4. Quantize
    if args.quantize:
        stem = os.path.splitext(args.output)[0]
        int8_path = f"{stem}_int8.onnx"
        calib_paths = _collect_calibration_images(args.calibration_dir, args.calibration_samples)
        _quantize_static(args.output, int8_path, calib_paths, args.image_size)
        print(f"\nINT8 model: {int8_path}")

    print(f"\nDone. ONNX model: {args.output}")
    print(
        f"To classify on robot:\n"
        f"  from pipeline.training.edge_inference import EdgeClassifier\n"
        f"  clf = EdgeClassifier('{args.output}', '.data/gallery/mission_objects.npz')\n"
        f"  labels = clf.classify(frame_pil)"
    )


if __name__ == "__main__":
    main()
