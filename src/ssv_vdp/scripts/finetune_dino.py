#!/usr/bin/env python
"""CLI for self-supervised DINOv3 domain adaptation on mission frames.

Usage:
    python scripts/finetune_dino.py [OPTIONS]

Minimal example (CPU, for smoke-testing):
    python scripts/finetune_dino.py \\
        --frames-dir .data/frames \\
        --output-dir .data/checkpoints \\
        --epochs 2 --batch-size 8 --device cpu

GPU example (recommended):
    python scripts/finetune_dino.py \\
        --frames-dir .data/frames \\
        --output-dir .data/checkpoints \\
        --approach temporal \\
        --epochs 10 --batch-size 64 \\
        --device cuda

After training, point the worker/API at the fine-tuned weights:
    export DINO_CHECKPOINT=.data/checkpoints/dino_ssl_best.pt
    make up

The fine-tuned backbone will be loaded by DINOEmbedder automatically when
DINO_CHECKPOINT is set and the file exists.
"""

import argparse

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-supervised DINOv3 domain adaptation on mission frames"
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Root directory containing extracted mission frames (default: DATA_DIR/frames)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write checkpoints (default: SSL_CHECKPOINT_DIR / DATA_DIR/checkpoints)",
    )
    parser.add_argument(
        "--approach",
        choices=["temporal", "augment"],
        default=None,
        help="Positive-pair strategy: 'temporal' (consecutive frames from same video dir) "
        "or 'augment' (two augmented views of the same frame). Default: SSL_FINETUNE_APPROACH",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="DINOv3 hub model name, e.g. dinov3_vitb14 (default: derived from MODEL_NAME)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: SSL_FINETUNE_EPOCHS)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size (default: SSL_FINETUNE_BATCH_SIZE)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (default: SSL_FINETUNE_LR)"
    )
    parser.add_argument(
        "--freeze-blocks",
        type=int,
        default=None,
        help="Number of transformer blocks to freeze from the start "
        "(default: SSL_FINETUNE_FREEZE_BLOCKS)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="NT-Xent softmax temperature (default: SSL_FINETUNE_TEMPERATURE)",
    )
    parser.add_argument(
        "--device", default=None, help="Compute device: cpu | cuda | cuda:N (default: auto)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker processes (default: 4)"
    )
    parser.add_argument(
        "--max-gap", type=int, default=3, help="Maximum frame gap for temporal pairs (default: 3)"
    )
    parser.add_argument(
        "--save-every", type=int, default=1, help="Save checkpoint every N epochs (default: 1)"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Build config from settings, then overlay CLI overrides
    from selfsuvis.pipeline.training.ssl import config_from_settings, run_finetune

    cfg = config_from_settings()

    if args.frames_dir is not None:
        cfg.frames_dir = args.frames_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.approach is not None:
        cfg.approach = args.approach
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.freeze_blocks is not None:
        cfg.freeze_blocks = args.freeze_blocks
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.device is not None:
        cfg.device = args.device
    cfg.num_workers = args.num_workers
    cfg.max_gap = args.max_gap
    cfg.save_every = args.save_every
    cfg.seed = args.seed

    # Resolve "auto" device
    import torch

    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nSelf-supervised DINOv3 fine-tuning")
    print(f"  frames_dir    : {cfg.frames_dir}")
    print(f"  output_dir    : {cfg.output_dir}")
    print(f"  approach      : {cfg.approach}")
    print(f"  model         : {cfg.model_name}")
    print(f"  epochs        : {cfg.epochs}")
    print(f"  batch_size    : {cfg.batch_size}")
    print(f"  lr            : {cfg.lr}")
    print(f"  freeze_blocks : {cfg.freeze_blocks}")
    print(f"  temperature   : {cfg.temperature}")
    print(f"  device        : {cfg.device}")
    print()

    best = run_finetune(cfg)
    print(f"\nDone. Best checkpoint: {best}")
    print(f"To use: export DINO_CHECKPOINT={best}")


if __name__ == "__main__":
    main()
