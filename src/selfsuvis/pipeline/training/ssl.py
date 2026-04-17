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
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def build_augment_transform(image_size: int = 224) -> transforms.Compose:
    """Strong random augmentation for contrastive self-supervised learning.

    Follows SimCLR / MoCo conventions: random crop + flip + colour jitter + blur.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic centre-crop transform (matches DINOEmbedder.preprocess)."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Datasets ──────────────────────────────────────────────────────────────────

def _collect_frame_paths(frames_dir: str) -> List[str]:
    """Recursively collect all JPEG/PNG frame files under frames_dir."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths: List[str] = []
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)


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
        self.pairs: List[Tuple[str, str]] = []
        self._build_pairs(frames_dir)
        if not self.pairs:
            raise ValueError(
                f"No temporal pairs found under {frames_dir!r}. "
                "Ensure frames are organised in per-video subdirectories."
            )

    def _build_pairs(self, frames_dir: str) -> None:
        exts = {".jpg", ".jpeg", ".png"}
        for video_dir in sorted(Path(frames_dir).iterdir()):
            if not video_dir.is_dir():
                continue
            frames = sorted(
                p for p in video_dir.iterdir() if p.suffix.lower() in exts
            )
            if len(frames) < 2:
                continue
            # Iterate over all frames except the last (which has no successor)
            for i in range(len(frames) - 1):
                max_possible = len(frames) - 1 - i
                gap = random.randint(1, min(self.max_gap, max_possible))
                self.pairs.append((str(frames[i]), str(frames[i + gap])))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p1, p2 = self.pairs[idx]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        return self.transform(img1), self.transform(img2)


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
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


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
            model_name, freeze_blocks, self._count_trainable(),
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
        feats = self.backbone(x)          # (B, embed_dim)
        return self.head(feats)           # (B, proj_out_dim), L2-normalised

    def save_checkpoint(self, path: str) -> None:
        """Save backbone state dict only (head is discarded at inference time)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.backbone.state_dict(), path)
        logger.info("Checkpoint saved: %s", path)

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
    approach: str = "temporal"        # "temporal" | "augment"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.04
    temperature: float = 0.07
    freeze_blocks: int = 10
    embed_dim: int = 768
    proj_out_dim: int = 128
    num_workers: int = 4
    save_every: int = 1               # save checkpoint every N epochs
    max_gap: int = 3                  # TemporalPairDataset only
    device: str = "cpu"
    seed: int = 42


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
    os.makedirs(cfg.output_dir, exist_ok=True)

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
        len(dataset), cfg.approach, cfg.epochs, cfg.batch_size, cfg.device,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    loss_fn = NTXentLoss(temperature=cfg.temperature)

    best_loss = float("inf")
    best_path = os.path.join(cfg.output_dir, "dino_ssl_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        tuner.train()
        epoch_losses: List[float] = []

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
        logger.info("Epoch %d/%d  loss=%.4f  lr=%.2e",
                    epoch, cfg.epochs, avg_loss, scheduler.get_last_lr()[0])

        if epoch % cfg.save_every == 0:
            ckpt = os.path.join(cfg.output_dir, f"dino_ssl_{epoch:03d}.pt")
            tuner.save_checkpoint(ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            tuner.save_checkpoint(best_path)
            logger.info("New best checkpoint: loss=%.4f → %s", best_loss, best_path)

    logger.info("Fine-tuning complete. Best loss=%.4f  checkpoint=%s",
                best_loss, best_path)
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
        frame_paths: List[str],
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
            epoch_losses: List[float] = []
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
                torch.nn.utils.clip_grad_norm_(
                    list(self._tuner.trainable_params()), max_norm=1.0
                )
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
