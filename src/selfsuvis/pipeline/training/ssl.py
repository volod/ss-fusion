"""Self-supervised domain adaptation for DINOv3.

Fine-tunes the last N transformer blocks of a pretrained DINOv3 (or DINOv2)
backbone on mission keyframes using a contrastive NT-Xent (SimCLR) loss.
No annotations required — uses only the frames already collected in DATA_DIR/frames/.

Two approaches are supported (SSL_FINETUNE_APPROACH env var):

  "temporal"  — positive pairs are consecutive frames from the same video directory
                (frame[i], frame[i+k], k ∈ 1..max_gap). Exploits temporal continuity.
  "augment"   — positive pairs are two independent random augmentations of the same frame.
                Works even when frames are not organised by video / timestamp.

Training produces:
  {SSL_CHECKPOINT_DIR}/dino_ssl_{epoch:03d}.pt  — per-epoch checkpoints (backbone weights only)
  {SSL_CHECKPOINT_DIR}/dino_ssl_best.pt          — best (lowest loss) checkpoint

Loading the fine-tuned model:
  Set DINO_CHECKPOINT=/path/to/dino_ssl_best.pt before starting the worker/API.
  DINOEmbedder will load the weights automatically.

Usage (standalone):
    python scripts/finetune_dino.py --frames-dir data/frames --output-dir data/checkpoints
"""

import glob
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from selfsuvis.pipeline.core.logging import get_logger

from .common import (
    checkpoint_path,
    ensure_output_dir,
    epoch_checkpoint_path,
    save_backbone_checkpoint,
)

logger = get_logger(__name__)


@dataclass
class TemporalVisualPair:
    anchor_frame_path: str
    positive_frame_path: str
    time_delta_sec: float = 0.0
    track_id: int | None = None
    pose_overlap_score: float | None = None
    sample_weight: float = 1.0
    modality_payload: dict[str, Any] = field(default_factory=dict)
    pair_source: str = "temporal_visual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_type": "temporal_visual",
            "pair_source": self.pair_source,
            "anchor_frame_path": self.anchor_frame_path,
            "positive_frame_path": self.positive_frame_path,
            "time_delta_sec": float(self.time_delta_sec),
            "track_id": self.track_id,
            "pose_overlap_score": self.pose_overlap_score,
            "sample_weight": float(self.sample_weight),
            "modality_payload": dict(self.modality_payload),
        }


@dataclass
class CrossModalPair:
    anchor_frame_path: str
    positive_frame_path: str
    time_delta_sec: float = 0.0
    track_id: int | None = None
    pose_overlap_score: float | None = None
    sample_weight: float = 1.0
    modality_payload: dict[str, Any] = field(default_factory=dict)
    pair_source: str = "cross_modal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_type": "cross_modal",
            "pair_source": self.pair_source,
            "anchor_frame_path": self.anchor_frame_path,
            "positive_frame_path": self.positive_frame_path,
            "time_delta_sec": float(self.time_delta_sec),
            "track_id": self.track_id,
            "pose_overlap_score": self.pose_overlap_score,
            "sample_weight": float(self.sample_weight),
            "modality_payload": dict(self.modality_payload),
        }


@dataclass
class GeometryPair:
    anchor_frame_path: str
    positive_frame_path: str
    time_delta_sec: float = 0.0
    track_id: int | None = None
    pose_overlap_score: float | None = None
    sample_weight: float = 1.0
    modality_payload: dict[str, Any] = field(default_factory=dict)
    pair_source: str = "geometry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_type": "geometry",
            "pair_source": self.pair_source,
            "anchor_frame_path": self.anchor_frame_path,
            "positive_frame_path": self.positive_frame_path,
            "time_delta_sec": float(self.time_delta_sec),
            "track_id": self.track_id,
            "pose_overlap_score": self.pose_overlap_score,
            "sample_weight": float(self.sample_weight),
            "modality_payload": dict(self.modality_payload),
        }


MultimodalPair = TemporalVisualPair | CrossModalPair | GeometryPair


def collate_multimodal_pairs(
    pairs: list[MultimodalPair],
) -> dict[str, Any]:
    """Collate pair metadata into a JSON-friendly batch payload.

    Missing optional modality targets are represented as ``None`` so callers can
    mask them out without losing batch alignment.
    """
    batch: dict[str, Any] = {
        "records": pairs,
        "pair_types": [],
        "pair_sources": [],
        "anchor_frame_paths": [],
        "positive_frame_paths": [],
        "time_delta_sec": torch.tensor([], dtype=torch.float32),
        "sample_weight": torch.tensor([], dtype=torch.float32),
        "track_id": [],
        "pose_overlap_score": [],
        "depth_similarity_target": [],
        "motion_similarity_target": [],
        "geometry_similarity_target": [],
        "modality_payload": [],
    }
    if not pairs:
        return batch

    time_deltas: list[float] = []
    sample_weights: list[float] = []
    for pair in pairs:
        payload = dict(getattr(pair, "modality_payload", {}) or {})
        pair_type = pair.to_dict().get("pair_type", "unknown")
        batch["pair_types"].append(pair_type)
        batch["pair_sources"].append(getattr(pair, "pair_source", pair_type))
        batch["anchor_frame_paths"].append(pair.anchor_frame_path)
        batch["positive_frame_paths"].append(pair.positive_frame_path)
        batch["track_id"].append(pair.track_id)
        batch["pose_overlap_score"].append(pair.pose_overlap_score)
        batch["depth_similarity_target"].append(payload.get("depth_similarity_target"))
        batch["motion_similarity_target"].append(payload.get("motion_similarity_target"))
        batch["geometry_similarity_target"].append(
            payload.get("geometry_similarity_target", pair.pose_overlap_score)
        )
        batch["modality_payload"].append(payload)
        time_deltas.append(float(pair.time_delta_sec))
        sample_weights.append(float(pair.sample_weight))

    batch["time_delta_sec"] = torch.tensor(time_deltas, dtype=torch.float32)
    batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
    return batch


# ── Augmentation pipeline ─────────────────────────────────────────────────────


def build_augment_transform(image_size: int = 224) -> transforms.Compose:
    """Strong random augmentation for contrastive self-supervised learning.

    Follows SimCLR / MoCo conventions: random crop + flip + colour jitter + blur.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic centre-crop transform (matches DINOEmbedder.preprocess)."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ── Datasets ──────────────────────────────────────────────────────────────────


def _collect_frame_paths(frames_dir: str) -> list[str]:
    """Recursively collect all JPEG/PNG frame files under frames_dir."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, "**", ext), recursive=True))
    return sorted(paths)


class AugmentPairDataset(Dataset):
    """Returns two independently augmented views of the same frame.

    Each __getitem__ call applies the augmentation transform twice to the same
    image, producing a positive pair for contrastive training without requiring
    any temporal or label information.
    """

    def __init__(self, frames_dir: str, transform: transforms.Compose):
        self.paths = _collect_frame_paths(frames_dir)
        if not self.paths:
            raise ValueError(f"No frames found under {frames_dir!r}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)


def _crop_bbox(
    img: "Image.Image",
    bbox_norm: list[float],
    padding: float = 0.15,
    size: int = 224,
) -> "Image.Image":
    """Crop image around a normalised bbox [x1,y1,x2,y2] with padding, resize to size.

    A 15 % padding margin on each side prevents the model from seeing only the
    tightest crop; it provides enough visual context for the object while still
    being much tighter than a full-frame crop.  Degenerate bboxes (area < 1 %)
    fall back to the full image to avoid empty tensors.
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox_norm
    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    x1 = max(0.0, x1 - pw)
    y1 = max(0.0, y1 - ph)
    x2 = min(1.0, x2 + pw)
    y2 = min(1.0, y2 + ph)
    cx1, cy1, cx2, cy2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return img.resize((size, size), Image.BICUBIC)
    return img.crop((cx1, cy1, cx2, cy2)).resize((size, size), Image.BICUBIC)


class TemporalPairDataset(Dataset):
    """Positive pairs are consecutive frames from the same video directory.

    Directory structure assumed:
        frames_dir/
            {video_id}/
                frame_0001.jpg
                frame_0002.jpg
                ...

    Pairs: (frame[i], frame[i+k]) for k sampled uniformly from 1..max_gap.
    Directories with fewer than 2 frames are skipped.
    """

    def __init__(
        self,
        frames_dir: str,
        transform: transforms.Compose,
        max_gap: int = 3,
    ):
        self.transform = transform
        self.max_gap = max(1, max_gap)
        self.pairs: list[tuple[str, str]] = []
        self._build_pairs(frames_dir)
        if not self.pairs:
            raise ValueError(
                f"No temporal pairs found under {frames_dir!r}. "
                "Ensure frames are organised in per-video subdirectories."
            )

    def _build_pairs(self, frames_dir: str) -> None:
        exts = {".jpg", ".jpeg", ".png"}
        frames_root = Path(frames_dir)
        direct_frames = (
            sorted(p for p in frames_root.iterdir() if p.is_file() and p.suffix.lower() in exts)
            if frames_root.exists()
            else []
        )
        if len(direct_frames) >= 2:
            self._append_pairs_for_sequence(direct_frames)
            return

        for video_dir in sorted(frames_root.iterdir()):
            if not video_dir.is_dir():
                continue
            frames = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in exts)
            if len(frames) < 2:
                continue
            self._append_pairs_for_sequence(frames)

    def _append_pairs_for_sequence(self, frames: list[Path]) -> None:
        # Iterate over all frames except the last (which has no successor)
        for i in range(len(frames) - 1):
            max_possible = len(frames) - 1 - i
            gap = random.randint(1, min(self.max_gap, max_possible))
            self.pairs.append((str(frames[i]), str(frames[i + gap])))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p1, p2 = self.pairs[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        return self.transform(img1), self.transform(img2)


class TrackPairDataset(Dataset):
    """Positive pairs built from RF-DETR track IDs: two bbox-crops of the same object.

    Each positive pair is (crop of object at time t, crop of same object at time t+k)
    where k ∈ [min_gap, max_gap].  Cropping around the tracked bbox provides a much
    tighter positive-pair signal than full-frame temporal pairs — the model must learn
    appearance-invariant features for the specific object instance, not just spatial
    proximity between consecutive frames.

    Args:
        track_map:  {track_id: [(frame_path, bbox_norm, t_sec), ...]} sorted by t_sec.
                    Produced by steps_ssl._extract_track_map().
        transform:  Augmentation transform applied to each crop.
        min_gap:    Minimum number of appearances apart within the track (default 2).
        max_gap:    Maximum number of appearances apart within the track (default 5).
        image_size: Output crop size fed to the backbone (default 224).
        crop_pad:   Fractional padding around bbox on each side (default 0.15).
    """

    def __init__(
        self,
        track_map: "dict[int, list[tuple[str, list[float], float]]]",
        transform: transforms.Compose,
        min_gap: int = 2,
        max_gap: int = 5,
        image_size: int = 224,
        crop_pad: float = 0.15,
    ):
        self.transform = transform
        self.image_size = image_size
        self.crop_pad = crop_pad
        self.pairs: list[tuple[str, list[float], str, list[float]]] = []
        for appearances in track_map.values():
            n = len(appearances)
            if n < 2:
                continue
            for i in range(n - min_gap):
                hi = min(n - 1, i + max_gap)
                lo = i + min_gap
                if lo > hi:
                    continue
                j = random.randint(lo, hi)
                fp_a, bbox_a, _ = appearances[i]
                fp_b, bbox_b, _ = appearances[j]
                self.pairs.append((fp_a, bbox_a, fp_b, bbox_b))
        if not self.pairs:
            raise ValueError(
                "TrackPairDataset: no valid pairs — need tracks with ≥2 appearances "
                f"at gap {min_gap}–{max_gap}."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fp_a, bbox_a, fp_b, bbox_b = self.pairs[idx]
        img_a = Image.open(fp_a).convert("RGB")
        img_b = Image.open(fp_b).convert("RGB")
        crop_a = _crop_bbox(img_a, bbox_a, self.crop_pad, self.image_size)
        crop_b = _crop_bbox(img_b, bbox_b, self.crop_pad, self.image_size)
        return self.transform(crop_a), self.transform(crop_b)


class TrackTripletDataset(Dataset):
    """Triplets (A, B, C) from the same track for cycle-consistency training.

    A = appearances[i], B = appearances[i+k], C = appearances[i+2k] where
    k ∈ [min_gap, max_gap].  All three crops are of the same tracked object at
    increasing times, enabling the CycleConsistencyLoss to enforce:
      - forward:  embed(A) ≈ embed(B)
      - backward: embed(B) ≈ embed(C)
      - cycle:    embed(A) ≈ embed(C)  (long-horizon consistency, down-weighted)

    The cycle term is the key addition over standard temporal pairs — it explicitly
    teaches the model that the same object identity must be preserved across the
    widest temporal gap in the triplet, preventing drift in long tracks.

    Args:
        track_map:  {track_id: [(frame_path, bbox_norm, t_sec), ...]} sorted by t_sec.
        transform:  Augmentation transform applied to each crop.
        min_gap:    Minimum step k between consecutive triplet members (default 2).
        max_gap:    Maximum step k (default 5).
        image_size: Output crop size (default 224).
        crop_pad:   Fractional padding around bbox (default 0.15).
    """

    def __init__(
        self,
        track_map: "dict[int, list[tuple[str, list[float], float]]]",
        transform: transforms.Compose,
        min_gap: int = 2,
        max_gap: int = 5,
        image_size: int = 224,
        crop_pad: float = 0.15,
    ):
        self.transform = transform
        self.image_size = image_size
        self.crop_pad = crop_pad
        self.triplets: list[tuple[str, list[float], str, list[float], str, list[float]]] = []
        for appearances in track_map.values():
            n = len(appearances)
            if n < 2 * min_gap + 1:
                continue
            for i in range(n - 2 * min_gap):
                max_k = min(max_gap, (n - 1 - i) // 2)
                if max_k < min_gap:
                    continue
                k = random.randint(min_gap, max_k)
                j = i + k
                end_idx = i + 2 * k
                if end_idx >= n:
                    continue
                fp_a, bbox_a, _ = appearances[i]
                fp_b, bbox_b, _ = appearances[j]
                fp_c, bbox_c, _ = appearances[end_idx]
                self.triplets.append((fp_a, bbox_a, fp_b, bbox_b, fp_c, bbox_c))
        if not self.triplets:
            raise ValueError(
                "TrackTripletDataset: no valid triplets — need tracks with ≥"
                f"{2 * min_gap + 1} appearances at gap {min_gap}–{max_gap}."
            )

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fp_a, bbox_a, fp_b, bbox_b, fp_c, bbox_c = self.triplets[idx]
        img_a = Image.open(fp_a).convert("RGB")
        img_b = Image.open(fp_b).convert("RGB")
        img_c = Image.open(fp_c).convert("RGB")
        crop_a = _crop_bbox(img_a, bbox_a, self.crop_pad, self.image_size)
        crop_b = _crop_bbox(img_b, bbox_b, self.crop_pad, self.image_size)
        crop_c = _crop_bbox(img_c, bbox_c, self.crop_pad, self.image_size)
        return self.transform(crop_a), self.transform(crop_b), self.transform(crop_c)


class MultimodalPairDataset(Dataset):
    """Return transformed RGB pairs plus typed multimodal metadata.

    The metadata record stays out-of-band from the image tensors so training can
    add auxiliary losses only when a pair provides the required side-channel.
    """

    def __init__(
        self,
        pairs: list[MultimodalPair],
        transform: transforms.Compose,
    ):
        if not pairs:
            raise ValueError("MultimodalPairDataset: no pairs supplied")
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, MultimodalPair]:
        pair = self.pairs[idx]
        img1 = Image.open(pair.anchor_frame_path).convert("RGB")
        img2 = Image.open(pair.positive_frame_path).convert("RGB")
        return self.transform(img1), self.transform(img2), pair


def multimodal_batch_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, MultimodalPair]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Stack image tensors and collate typed pair metadata."""
    v1 = torch.stack([item[0] for item in batch], dim=0)
    v2 = torch.stack([item[1] for item in batch], dim=0)
    meta = collate_multimodal_pairs([item[2] for item in batch])
    return v1, v2, meta


# ── Loss ──────────────────────────────────────────────────────────────────────


class NTXentLoss(nn.Module):
    """NT-Xent (Normalised Temperature-scaled Cross Entropy) loss.

    InfoNCE loss for contrastive learning (SimCLR formulation).
    Given a batch of (z1, z2) positive pairs, treats all other samples in the
    batch as negatives.

    Args:
        temperature: Softmax temperature τ (default 0.07 following SimCLR).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: (B, D) L2-normalised embeddings for view 1.
            z2: (B, D) L2-normalised embeddings for view 2.

        Returns:
            Scalar loss.
        """
        B = z1.size(0)
        # Concatenate: [z1; z2] shape (2B, D)
        z = torch.cat([z1, z2], dim=0)
        # Similarity matrix (2B, 2B)
        sim = torch.mm(z, z.t()) / self.temperature
        # Mask out self-similarity on diagonal
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))
        # Positive indices: for i in [0,B), positive is i+B; for i in [B,2B), positive is i-B
        labels = torch.cat(
            [
                torch.arange(B, 2 * B, device=z.device),
                torch.arange(0, B, device=z.device),
            ]
        )
        return F.cross_entropy(sim, labels)


class CycleConsistencyLoss(nn.Module):
    """Cycle-consistency contrastive loss for temporal track triplets.

    For a triplet (A, B, C) representing the same tracked object at times t, t+k,
    t+2k, the loss has three components:

        forward:   NTXent(z_A, z_B)             — adjacent-pair consistency
        backward:  NTXent(z_B, z_C)             — adjacent-pair consistency
        cycle:     λ · NTXent(z_A, z_C)         — long-horizon consistency

    The cycle term is the key addition over standard temporal SSL.  It enforces
    that the object's representation at time t and time t+2k are mutually
    consistent through the intermediate frame at t+k, preventing embedding drift
    along long tracks.  λ < 1 because the wider gap is genuinely harder: the
    object may have moved, rotated, or been partially occluded.

    Args:
        base_loss:     NTXentLoss instance (controls temperature).
        lambda_cycle:  Weight for the long-horizon term (default 0.3).
    """

    def __init__(self, base_loss: NTXentLoss, lambda_cycle: float = 0.3):
        super().__init__()
        self.base = base_loss
        self.lambda_cycle = lambda_cycle

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        """Compute cycle-consistency loss.

        Args:
            z_a: (B, D) L2-normalised embeddings for crop A (time t).
            z_b: (B, D) L2-normalised embeddings for crop B (time t+k).
            z_c: (B, D) L2-normalised embeddings for crop C (time t+2k).

        Returns:
            Scalar loss.
        """
        loss_ab = self.base(z_a, z_b)
        loss_bc = self.base(z_b, z_c)
        loss_ac = self.base(z_a, z_c)
        return loss_ab + loss_bc + self.lambda_cycle * loss_ac


class MultimodalConsistencyLoss(nn.Module):
    """Base contrastive loss with optional auxiliary consistency terms.

    Auxiliary terms supervise pairwise embedding cosine similarity against
    modality-specific targets in ``[0, 1]``. Missing targets are ignored.
    """

    def __init__(
        self,
        base_loss: NTXentLoss,
        depth_weight: float = 0.0,
        motion_weight: float = 0.0,
        geometry_weight: float = 0.0,
    ):
        super().__init__()
        self.base = base_loss
        self.depth_weight = float(depth_weight)
        self.motion_weight = float(motion_weight)
        self.geometry_weight = float(geometry_weight)

    @staticmethod
    def _masked_target_loss(
        pair_cos: torch.Tensor,
        targets: list[float | None],
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        target_values = [
            float(v) if v is not None and math.isfinite(float(v)) else float("nan") for v in targets
        ]
        target_tensor = torch.tensor(target_values, dtype=pair_cos.dtype, device=pair_cos.device)
        mask = torch.isfinite(target_tensor)
        if not bool(mask.any()):
            return pair_cos.new_tensor(0.0)
        diff = (pair_cos[mask] - target_tensor[mask]) ** 2
        weights = sample_weight[mask].to(pair_cos.device)
        denom = torch.clamp(weights.sum(), min=1e-6)
        return torch.sum(diff * weights) / denom

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_meta: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        base_loss = self.base(z1, z2)
        total = base_loss
        components: dict[str, float] = {
            "contrastive_loss": float(base_loss.detach().item()),
            "depth_consistency_loss": 0.0,
            "motion_consistency_loss": 0.0,
            "geometry_consistency_loss": 0.0,
        }
        if not batch_meta:
            return total, components

        pair_cos = torch.clamp(torch.sum(z1 * z2, dim=-1), -1.0, 1.0)
        sample_weight = batch_meta.get("sample_weight")
        if not isinstance(sample_weight, torch.Tensor) or sample_weight.numel() != pair_cos.numel():
            sample_weight = torch.ones_like(pair_cos)
        else:
            sample_weight = sample_weight.to(pair_cos.device, dtype=pair_cos.dtype)

        if self.depth_weight > 0.0:
            depth_loss = self._masked_target_loss(
                pair_cos,
                batch_meta.get("depth_similarity_target", []),
                sample_weight,
            )
            total = total + self.depth_weight * depth_loss
            components["depth_consistency_loss"] = float(depth_loss.detach().item())

        if self.motion_weight > 0.0:
            motion_loss = self._masked_target_loss(
                pair_cos,
                batch_meta.get("motion_similarity_target", []),
                sample_weight,
            )
            total = total + self.motion_weight * motion_loss
            components["motion_consistency_loss"] = float(motion_loss.detach().item())

        if self.geometry_weight > 0.0:
            geometry_loss = self._masked_target_loss(
                pair_cos,
                batch_meta.get("geometry_similarity_target", []),
                sample_weight,
            )
            total = total + self.geometry_weight * geometry_loss
            components["geometry_consistency_loss"] = float(geometry_loss.detach().item())

        return total, components


# ── Model wrapper ─────────────────────────────────────────────────────────────


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR style).

    Maps backbone CLS token → normalised lower-dimensional representation.
    Only used during training; discarded at inference time.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DINOFineTuner:
    """Wraps a pretrained DINOv3/DINOv2 ViT backbone for contrastive fine-tuning.

    Strategy: freeze the first `freeze_blocks` transformer blocks (protect
    low-level feature representations); fine-tune the remaining blocks + the
    projection head. This avoids catastrophic forgetting of generic features
    while adapting higher-level representations to the mission domain.

    ViT-B/14 has 12 transformer blocks. Default freeze_blocks=10 leaves the
    last 2 blocks + projection head trainable (~14 M parameters out of 86 M).

    Args:
        model_name:     DINOv3 hub model name (e.g. "dinov3_vitb14").
        freeze_blocks:  Number of transformer blocks to freeze from the start.
        device:         torch device string.
        embed_dim:      Backbone output dimension (768 for ViT-B, 1024 for ViT-L).
        proj_out_dim:   Projection head output dimension.
        temperature:    NT-Xent temperature.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb14",
        freeze_blocks: int = 10,
        device: str = "cpu",
        embed_dim: int = 768,
        proj_out_dim: int = 128,
    ):
        self.device = device
        self.model_name = model_name

        # Load backbone
        from selfsuvis.models.dino_model import hub_load_dino

        self.backbone = hub_load_dino(model_name, pretrained=True)
        self.backbone = self.backbone.to(device)

        # Freeze first N blocks
        self._freeze_blocks(freeze_blocks)

        # Projection head (trained from scratch, on top of backbone)
        self.head = ProjectionHead(in_dim=embed_dim, out_dim=proj_out_dim).to(device)

        logger.info(
            "DINOFineTuner: model=%s freeze_blocks=%d trainable_params=%d",
            model_name,
            freeze_blocks,
            self._count_trainable(),
        )

    def _freeze_blocks(self, n: int) -> None:
        """Freeze the patch embedding + first n transformer blocks."""
        # Freeze patch embed, pos embed, cls token
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        # Unfreeze blocks[n:] and the final norm
        blocks = list(self.backbone.blocks)
        for block in blocks[n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze final LayerNorm
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def _count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def train(self) -> None:
        self.backbone.train()
        self.head.train()

    def eval(self) -> None:
        self.backbone.eval()
        self.head.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: backbone CLS token → projection head → normalised vector."""
        feats = self.backbone(x)  # (B, embed_dim)
        return self.head(feats)  # (B, proj_out_dim), L2-normalised

    def save_checkpoint(self, path: str) -> None:
        """Save backbone state dict only (head is discarded at inference time)."""
        saved_path = save_backbone_checkpoint(self.backbone, path)
        logger.info("Checkpoint saved: %s", saved_path)

    @classmethod
    def load_backbone_weights(cls, backbone, checkpoint_path: str, device: str) -> None:
        """Load saved backbone weights into an existing model in-place."""
        state = torch.load(checkpoint_path, map_location=device)
        backbone.load_state_dict(state)
        logger.info("Loaded fine-tuned backbone from %s", checkpoint_path)


# ── Training config ───────────────────────────────────────────────────────────


@dataclass
class FinetuneConfig:
    frames_dir: str
    output_dir: str
    model_name: str = "dinov3_vitb14"
    approach: str = "temporal"  # "multimodal" | "track_cycle" | "track" | "temporal" | "augment"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.04
    temperature: float = 0.07
    freeze_blocks: int = 10
    embed_dim: int = 768
    proj_out_dim: int = 128
    num_workers: int = 4
    save_every: int = 1  # save checkpoint every N epochs
    max_gap: int = 3  # TemporalPairDataset only
    device: str = "cpu"
    seed: int = 42
    depth_consistency_weight: float = 0.15
    motion_consistency_weight: float = 0.10
    geometry_consistency_weight: float = 0.15


# ── Main training loop ────────────────────────────────────────────────────────


def run_finetune(cfg: FinetuneConfig) -> str:
    """Run self-supervised contrastive fine-tuning.

    Args:
        cfg: FinetuneConfig instance.

    Returns:
        Path to the best checkpoint (lowest average epoch loss).
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ensure_output_dir(cfg.output_dir)

    # Dataset
    transform = build_augment_transform()
    if cfg.approach == "temporal":
        dataset: Dataset = TemporalPairDataset(
            cfg.frames_dir, transform=transform, max_gap=cfg.max_gap
        )
    else:
        dataset = AugmentPairDataset(cfg.frames_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
        drop_last=True,
    )
    logger.info(
        "Dataset: %d pairs | approach=%s | epochs=%d | batch=%d | device=%s",
        len(dataset),
        cfg.approach,
        cfg.epochs,
        cfg.batch_size,
        cfg.device,
    )

    # Model + optimiser
    tuner = DINOFineTuner(
        model_name=cfg.model_name,
        freeze_blocks=cfg.freeze_blocks,
        device=cfg.device,
        embed_dim=cfg.embed_dim,
        proj_out_dim=cfg.proj_out_dim,
    )
    optimizer = torch.optim.AdamW(
        tuner.trainable_params(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = NTXentLoss(temperature=cfg.temperature)

    best_loss = float("inf")
    best_path = checkpoint_path(cfg.output_dir, "dino_ssl_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        tuner.train()
        epoch_losses: list[float] = []

        for v1, v2 in loader:
            v1 = v1.to(cfg.device)
            v2 = v2.to(cfg.device)

            z1 = tuner.forward(v1)
            z2 = tuner.forward(v2)
            loss = loss_fn(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        logger.info(
            "Epoch %d/%d  loss=%.4f  lr=%.2e",
            epoch,
            cfg.epochs,
            avg_loss,
            scheduler.get_last_lr()[0],
        )

        if epoch % cfg.save_every == 0:
            ckpt = epoch_checkpoint_path(cfg.output_dir, "dino_ssl", epoch)
            tuner.save_checkpoint(ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            tuner.save_checkpoint(best_path)
            logger.info("New best checkpoint: loss=%.4f → %s", best_loss, best_path)

    logger.info("Fine-tuning complete. Best loss=%.4f  checkpoint=%s", best_loss, best_path)
    return best_path


# ── Config from environment ───────────────────────────────────────────────────

# ── SkipStep sentinel ─────────────────────────────────────────────────────────


class SkipStep(RuntimeError):
    """Raised by GemmaSSLFinetuner when a required pre-condition is not met.

    Callers (demo_runner, worker) should catch this and log the reason without
    treating it as a hard failure — the pipeline continues with the DINOv3
    baseline instead.
    """


# ── GemmaSSLFinetuner ─────────────────────────────────────────────────────────


class GemmaSSLFinetuner:
    """Fine-tunes DINOv3 using Gemma vision encoder embeddings as SSL targets.

    Instead of NT-Xent contrastive pairs, this trainer uses a regression target:
    for each frame, the Gemma vision encoder produces a language-grounded
    embedding, and the DINOv3 student is trained to predict it via cosine loss.
    This grounds DINOv3 in language concepts, improving text-query retrieval.

    **Pre-condition:** CUDA must be available. Raises :exc:`SkipStep` on CPU-only
    machines — Gemma vision encoder requires ≥8 GB VRAM for batched inference.

    Args:
        gemma_embedder: A model with an ``encode_images(List[PIL.Image])``
            method returning ``(N, dim)`` float32 numpy arrays (L2-normalised).
            Typically an instance of ``models.gemma_model.GemmaEmbedder``.
        dino_model_name: DINOv3/DINOv2 hub model name for the student backbone.
        device:          Torch device string (``"cuda"`` is required; ``"auto"``
            will resolve to CUDA or raise SkipStep if unavailable).
        freeze_blocks:   Number of ViT transformer blocks to freeze (default 10).
        embed_dim:       Student backbone output dimension (768 for ViT-B).
        proj_out_dim:    Projection head output dimension — must match the Gemma
            embedding dimension so the cosine loss is well-defined.

    Raises:
        SkipStep: If ``torch.cuda.is_available()`` is False.
    """

    def __init__(
        self,
        gemma_embedder,
        dino_model_name: str = "dinov3_vitb14",
        device: str = "auto",
        freeze_blocks: int = 10,
        embed_dim: int = 768,
        proj_out_dim: int = 1152,  # Gemma-4 vision encoder dim
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            raise SkipStep(
                "GemmaSSL requires CUDA — CPU not supported. "
                "Falling back to DINOv3→EfficientViT-S1 baseline."
            )

        self._gemma = gemma_embedder
        self._device = device
        self._dino_model_name = dino_model_name
        self._freeze_blocks = freeze_blocks
        self._embed_dim = embed_dim
        self._proj_out_dim = proj_out_dim

        self._tuner = DINOFineTuner(
            model_name=dino_model_name,
            freeze_blocks=freeze_blocks,
            device=device,
            embed_dim=embed_dim,
            proj_out_dim=proj_out_dim,
        )

    def train(
        self,
        frame_paths: list[str],
        output_dir: str,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-5,
        weight_decay: float = 0.04,
        seed: int = 42,
    ) -> str:
        """Fine-tune DINOv3 toward Gemma embedding targets.

        For each mini-batch of frames:
          1. Embed with Gemma (frozen) → teacher targets T.
          2. Embed with DINOv3 student → student embeddings S.
          3. Minimise 1 − cosine_similarity(linear(S), T) for each frame.

        The linear projection aligns the student's 768-dim space with the
        teacher's ``proj_out_dim``-dim space.  It is discarded after training.

        Args:
            frame_paths: Absolute paths to training frames.
            output_dir:  Directory for student checkpoints.
            epochs:      Training epochs.
            batch_size:  Mini-batch size (reduce if OOM).
            lr:          AdamW learning rate.
            weight_decay: AdamW weight decay.
            seed:        Random seed for reproducibility.

        Returns:
            Path to the best checkpoint (lowest cosine loss).
        """
        import os as _os

        random.seed(seed)
        torch.manual_seed(seed)
        _os.makedirs(output_dir, exist_ok=True)

        eval_transform = build_eval_transform()
        best_loss = float("inf")
        best_path = _os.path.join(output_dir, "gemma_ssl_best.pt")
        optimizer = torch.optim.AdamW(
            self._tuner.trainable_params(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            epoch_losses: list[float] = []
            # Process frame_paths in batches
            indices = list(range(len(frame_paths)))
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch_paths = [frame_paths[i] for i in batch_indices]

                # Load PIL images
                from PIL import Image as _PIL_Image

                pil_images = []
                tensors = []
                for p in batch_paths:
                    try:
                        img = _PIL_Image.open(p).convert("RGB")
                        pil_images.append(img)
                        tensors.append(eval_transform(img))
                    except Exception:
                        logger.warning("GemmaSSL: skipping unreadable frame %s", p)

                if not tensors:
                    continue

                # Gemma teacher embeddings (frozen, no grad)
                with torch.no_grad():
                    teacher_np = self._gemma.encode_images(pil_images)
                    teacher = torch.from_numpy(teacher_np).to(self._device)  # (B, gemma_dim)
                    teacher = torch.nan_to_num(teacher, nan=0.0)

                # Student forward
                self._tuner.train()
                batch_tensor = torch.stack(tensors).to(self._device)
                student = self._tuner.forward(batch_tensor)  # (B, proj_out_dim) normalised

                # Cosine loss: 1 − cos(student, teacher)
                teacher_norm = torch.nn.functional.normalize(teacher, dim=-1)
                loss = (1.0 - (student * teacher_norm).sum(dim=-1)).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self._tuner.trainable_params()), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
            logger.info("GemmaSSL epoch %d/%d  loss=%.4f", epoch, epochs, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._tuner.save_checkpoint(best_path)
                logger.info("GemmaSSL new best: loss=%.4f → %s", best_loss, best_path)

        logger.info("GemmaSSL fine-tuning complete. best_loss=%.4f ckpt=%s", best_loss, best_path)
        return best_path

    def student_backbone(self) -> torch.nn.Module:
        """Return the fine-tuned student backbone (projection head discarded)."""
        self._tuner.eval()
        return self._tuner.backbone


def config_from_settings() -> FinetuneConfig:
    """Build FinetuneConfig from pipeline.core.config.settings."""
    from selfsuvis.pipeline.core.config import settings

    device = settings.DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "dinov3_vitb14" if settings.MODEL_NAME == "dinov3" else "dinov2_vitb14"
    # ViT-B variants: embed_dim=768; ViT-L: embed_dim=1024. Default to ViT-B.
    embed_dim = 1024 if "vitl" in model_name else 768

    return FinetuneConfig(
        frames_dir=settings.FRAMES_DIR,
        output_dir=settings.SSL_CHECKPOINT_DIR,
        model_name=model_name,
        approach=settings.SSL_FINETUNE_APPROACH,
        epochs=settings.SSL_FINETUNE_EPOCHS,
        batch_size=settings.SSL_FINETUNE_BATCH_SIZE,
        lr=settings.SSL_FINETUNE_LR,
        temperature=settings.SSL_FINETUNE_TEMPERATURE,
        freeze_blocks=settings.SSL_FINETUNE_FREEZE_BLOCKS,
        embed_dim=embed_dim,
        device=device,
    )
