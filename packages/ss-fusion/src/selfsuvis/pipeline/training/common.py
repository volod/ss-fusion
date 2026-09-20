"""Shared filesystem helpers for training stages."""

from pathlib import Path

import torch

PathLike = str | Path


def ensure_output_dir(path: PathLike) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def checkpoint_path(output_dir: PathLike, filename: str) -> str:
    return str(ensure_output_dir(output_dir) / filename)


def epoch_checkpoint_path(output_dir: PathLike, prefix: str, epoch: int) -> str:
    return checkpoint_path(output_dir, f"{prefix}_{epoch:03d}.pt")


def save_backbone_checkpoint(backbone, path: PathLike) -> str:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), str(ckpt_path))
    return str(ckpt_path)
