#!/usr/bin/env python3
"""Supervised contrastive fine-tuning of DINOv3 on CVAT-annotated frames.

Uses the Supervised Contrastive Loss (SupCon, Khosla et al. NeurIPS 2020) to
specialise the DINOv3 backbone on annotated mission footage.

Workflow:
  1. Export annotations from CVAT as "CVAT 1.1" format.
  2. Run this script:

     python scripts/supervised_finetune_dino.py \\
         --frames-dir .data/cvat_frames \\
         --cvat-xml   .data/cvat_annotations.xml \\
         --output-dir .data/checkpoints/supervised

  3. Point DINO_CHECKPOINT to the resulting dino_sup_best.pt checkpoint.

Warm-starting from an SSL checkpoint (recommended):
     python scripts/supervised_finetune_dino.py \\
         --frames-dir .data/frames \\
         --cvat-xml   /path/to/cvat_annotations.xml \\
         --output-dir .data/checkpoints/supervised \\
         --ssl-checkpoint .data/checkpoints/dino_ssl_best.pt
"""

import argparse
from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.training.supervised import SupervisedFinetuneConfig, run_supervised_finetune

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames-dir", required=True, help="Directory containing JPEG/PNG frames")
    p.add_argument("--cvat-xml", required=True, help="Path to CVAT XML 1.1 annotation file")
    p.add_argument("--output-dir", required=True, help="Directory to write checkpoint files")

    # Model
    p.add_argument("--model", default="dinov3_vitb14", help="DINOv3 hub model name")
    p.add_argument(
        "--embed-dim", type=int, default=768, help="Backbone embedding dim (768 ViT-B, 1024 ViT-L)"
    )
    p.add_argument("--proj-out-dim", type=int, default=128, help="Projection head output dim")
    p.add_argument(
        "--freeze-blocks", type=int, default=8, help="Number of transformer blocks to freeze"
    )
    p.add_argument(
        "--ssl-checkpoint", default="", help="Optional SSL backbone checkpoint to warm-start from"
    )

    # Training
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.04, help="AdamW weight decay")
    p.add_argument("--temperature", type=float, default=0.07, help="SupCon temperature τ")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    p.add_argument(
        "--save-every", type=int, default=1, help="Save per-epoch checkpoint every N epochs"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Device
    p.add_argument(
        "--device", default="auto", help="Device: 'auto' (prefer CUDA), 'cpu', or 'cuda'"
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = SupervisedFinetuneConfig(
        frames_dir=args.frames_dir,
        cvat_xml_path=args.cvat_xml,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        freeze_blocks=args.freeze_blocks,
        embed_dim=args.embed_dim,
        proj_out_dim=args.proj_out_dim,
        num_workers=args.num_workers,
        save_every=args.save_every,
        device=device,
        seed=args.seed,
        ssl_checkpoint=args.ssl_checkpoint or None,
    )

    best_ckpt = run_supervised_finetune(cfg)
    print(f"\nDone. Best checkpoint: {best_ckpt}")
    print(f"Set DINO_CHECKPOINT={best_ckpt} to use with the worker/API.")


if __name__ == "__main__":
    main()
