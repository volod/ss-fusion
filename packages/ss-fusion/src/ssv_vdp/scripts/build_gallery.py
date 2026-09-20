#!/usr/bin/env python
"""Build a gallery NPZ of mission-object embeddings for edge classification.

The gallery maps label names (e.g. "vehicle", "barrier", "personnel") to L2-normalised
embedding vectors.  EdgeClassifier loads this NPZ to perform cosine nearest-neighbour
classification without retraining.

Usage:

    # Using an ONNX model (preferred for edge pipelines):
    python scripts/build_gallery.py \\
        --onnx .data/models/dino_edge.onnx \\
        --labels "vehicle:.data/frames/vid1/frame_0010.jpg,vehicle:.data/frames/vid1/frame_0020.jpg" \\
        --labels "barrier:.data/frames/vid2/frame_0005.jpg" \\
        --output .data/gallery/mission_objects.npz

    # Using a labels-file (JSON):
    python scripts/build_gallery.py \\
        --onnx .data/models/dino_edge.onnx \\
        --labels-file .data/gallery/labels.json \\
        --output .data/gallery/mission_objects.npz

    # Using a PyTorch checkpoint (when ONNX not yet exported):
    python scripts/build_gallery.py \\
        --checkpoint .data/checkpoints/dino_ssl_best.pt \\
        --model-name dinov3_vitb14 \\
        --labels-file .data/gallery/labels.json \\
        --output .data/gallery/mission_objects.npz

labels-file JSON format:
    {
        "vehicle":   ["path/to/frame1.jpg", "path/to/frame2.jpg"],
        "barrier":   ["path/to/frame3.jpg"],
        "personnel": ["path/to/frame4.jpg", "path/to/frame5.jpg"]
    }

labels-file YAML format (requires PyYAML):
    vehicle:
      - path/to/frame1.jpg
      - path/to/frame2.jpg
    barrier:
      - path/to/frame3.jpg
"""

import argparse
import json
import os
from collections import defaultdict

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)


def _parse_labels_arg(labels_args: list[str]) -> dict[str, list[str]]:
    """Parse --labels entries of the form "label:path" or "label:path1,label:path2,...".

    Each --labels value is a comma-separated list of label:path pairs.
    Multiple --labels flags are allowed.
    """
    result: dict[str, list[str]] = defaultdict(list)
    for arg in labels_args:
        for token in arg.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                raise ValueError(f"Invalid --labels token {token!r}. Expected format: label:path")
            label, path = token.split(":", 1)
            result[label.strip()].append(path.strip())
    return dict(result)


def _load_labels_file(path: str) -> dict[str, list[str]]:
    """Load a JSON or YAML labels file."""
    _, ext = os.path.splitext(path)
    with open(path) as fh:
        content = fh.read()

    if ext.lower() in {".json"}:
        data = json.loads(content)
    elif ext.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML labels files. Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(content)
    else:
        # Try JSON first, then YAML
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            try:
                import yaml

                data = yaml.safe_load(content)
            except Exception:
                raise ValueError(f"Could not parse labels file {path!r} as JSON or YAML.")

    if not isinstance(data, dict):
        raise ValueError(
            f"Labels file must be a dict mapping label → list of paths, got {type(data).__name__}"
        )
    return {k: list(v) for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a gallery NPZ of mission-object embeddings for EdgeClassifier"
    )

    # Embedding source (exactly one required)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--onnx",
        default=None,
        help="Path to ONNX model (onnxruntime-based; no PyTorch required at inference time)",
    )
    source_group.add_argument(
        "--checkpoint",
        default=None,
        help="Path to dino_ssl_best.pt — use PyTorch backbone directly (when ONNX not yet built)",
    )

    parser.add_argument(
        "--model-name",
        default="dinov3_vitb14",
        help="Hub model name used when --checkpoint is provided (default: dinov3_vitb14)",
    )

    # Label specification (exactly one required)
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument(
        "--labels",
        action="append",
        default=None,
        metavar="label:path[,label:path,...]",
        help="Comma-separated label:path pairs. May be repeated.",
    )
    label_group.add_argument(
        "--labels-file",
        default=None,
        help="JSON or YAML file mapping label → [path, ...] (see module docstring for format)",
    )

    parser.add_argument(
        "--output",
        default=".data/gallery/mission_objects.npz",
        help="Output gallery NPZ path (default: .data/gallery/mission_objects.npz)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size for preprocessing (default: 224)",
    )

    args = parser.parse_args()

    # Parse labels_map
    if args.labels is not None:
        labels_map = _parse_labels_arg(args.labels)
    else:
        labels_map = _load_labels_file(args.labels_file)

    if not labels_map:
        parser.error("No labels found. Provide at least one label:path pair.")

    # Print summary
    total_frames = sum(len(v) for v in labels_map.values())
    print("\nBuild gallery NPZ")
    print(
        f"  source         : {'ONNX: ' + args.onnx if args.onnx else 'PyTorch checkpoint: ' + args.checkpoint}"
    )
    if args.checkpoint and not args.onnx:
        print(f"  model_name     : {args.model_name}")
    print(f"  output         : {args.output}")
    print(f"  image_size     : {args.image_size}")
    print(f"  labels ({len(labels_map):>3})   :")
    for label, paths in sorted(labels_map.items()):
        print(f"    {label:20s}  {len(paths)} frame(s)")
    print(f"  total frames   : {total_frames}")
    print()

    from selfsuvis.pipeline.training.edge_inference import build_gallery

    if args.onnx:
        build_gallery(
            labels_map=labels_map,
            output_path=args.output,
            onnx_path=args.onnx,
            image_size=args.image_size,
        )
    else:
        # Load PyTorch backbone
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        from selfsuvis.models.dino_model import hub_load_dino

        logger.info("Loading backbone %s on %s ...", args.model_name, device)
        backbone = hub_load_dino(args.model_name, pretrained=True)
        backbone = backbone.to(device)

        if args.checkpoint:
            logger.info("Loading checkpoint %s ...", args.checkpoint)
            state = torch.load(args.checkpoint, map_location=device)
            backbone.load_state_dict(state)

        backbone.eval()

        build_gallery(
            labels_map=labels_map,
            output_path=args.output,
            backbone=backbone,
            image_size=args.image_size,
        )

    print(f"\nDone. Gallery saved to: {args.output}")
    print(
        f"To use on robot:\n"
        f"  from pipeline.training.edge_inference import EdgeClassifier\n"
        f"  clf = EdgeClassifier('dino_edge.onnx', '{args.output}')\n"
        f"  labels = clf.classify(frame_pil)"
    )


if __name__ == "__main__":
    main()
