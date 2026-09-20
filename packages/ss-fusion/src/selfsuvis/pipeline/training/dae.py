"""Denoising Autoencoder (DAE) for self-supervised frame pretraining.

Implements a convolutional encoder-decoder network trained to reconstruct
clean frames from synthetically corrupted inputs.  The pretext task is
inspired by the denoising autoencoder literature (Vincent et al., 2008) and
the anomaly detection approach where reconstruction error at inference time
serves as an anomaly score without requiring any labeled data.

Two corruption modes are supported:
  "gaussian"  -- additive Gaussian noise (classic DAE)
  "masking"   -- random patch masking + Gaussian noise (hybrid MAE/DAE)
  "both"      -- both modes applied sequentially

Training produces:
  {output_dir}/dae_best.pt     -- best reconstruction-loss checkpoint (full model)
  {output_dir}/dae_encoder.pt  -- encoder weights only (for downstream feature use)

The trained encoder provides reconstruction-motivated representations complementary
to DINOv3 contrastive embeddings.  At inference, reconstruction MSE of a full frame
against its own reconstruction serves as a per-frame anomaly score.

Usage (standalone):
    python scripts/finetune_dae.py --frames-dir .data/frames --output-dir .data/checkpoints
"""

import glob
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from selfsuvis.pipeline.core.logging import get_logger

from .common import ensure_output_dir

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = get_logger(__name__)

_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


# -- Transforms ----------------------------------------------------------------


def build_dae_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic normalizing transform (no random augmentation for DAE)."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD),
        ]
    )


def _add_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + torch.randn_like(x) * std


def _add_patch_mask(x: torch.Tensor, patch_size: int, mask_frac: float) -> torch.Tensor:
    """Mask a fraction of non-overlapping patches by replacing with channel mean.

    Channel-mean fill prevents the decoder from using zero-fill boundaries as
    trivial reconstruction cues that leak patch positions.
    """
    _, H, W = x.shape
    gh = H // patch_size
    gw = W // patch_size
    n_patches = gh * gw
    n_masked = max(1, int(n_patches * mask_frac))
    masked_idx = random.sample(range(n_patches), k=min(n_masked, n_patches))
    out = x.clone()
    for idx in masked_idx:
        r = (idx // gw) * patch_size
        c = (idx % gw) * patch_size
        fill = x[:, r : r + patch_size, c : c + patch_size].mean(dim=(1, 2), keepdim=True)
        out[:, r : r + patch_size, c : c + patch_size] = fill
    return out


def corrupt(
    x: torch.Tensor,
    mode: str = "both",
    noise_std: float = 0.2,
    patch_size: int = 16,
    mask_frac: float = 0.15,
) -> torch.Tensor:
    """Apply corruption to a (C, H, W) tensor.

    Args:
        x:          Clean input tensor (normalised).
        mode:       "gaussian" | "masking" | "both".
        noise_std:  Gaussian noise standard deviation.
        patch_size: Patch side length for masking corruption.
        mask_frac:  Fraction of patches to mask.

    Returns:
        Corrupted tensor of the same shape.
    """
    if mode in ("gaussian", "both"):
        x = _add_gaussian_noise(x, noise_std)
    if mode in ("masking", "both"):
        x = _add_patch_mask(x, patch_size, mask_frac)
    return x


# -- Model ---------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEncoder(nn.Module):
    """4-block strided-conv encoder: 3x224x224 -> latent_ch x 14x14.

    Downsamples by 2x at each block (stride-2 conv), giving a 16x total
    spatial reduction.  The 14x14 output grid matches the DINOv2/v3 patch
    token grid (patch_size=16), keeping representations comparable.

    Channel progression: 3 -> 64 -> 128 -> 256 -> latent_ch.
    """

    def __init__(self, latent_ch: int = 256):
        super().__init__()
        self.blocks = nn.Sequential(
            _ConvBlock(3, 64, stride=2),
            _ConvBlock(64, 128, stride=2),
            _ConvBlock(128, 256, stride=2),
            _ConvBlock(256, latent_ch, stride=2),
        )
        self._latent_ch = latent_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    @property
    def out_channels(self) -> int:
        return self._latent_ch


class ConvDecoder(nn.Module):
    """Symmetric transposed-conv decoder: latent_ch x 14x14 -> 3x224x224."""

    def __init__(self, latent_ch: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenoisingAutoencoder(nn.Module):
    """Convolutional Denoising Autoencoder.

    Encodes a corrupted frame into a 14x14 spatial bottleneck then reconstructs
    the clean original.  Training minimises pixel-space MSE between the
    reconstruction and the clean frame.

    At inference, the MSE between a raw frame and its own reconstruction serves
    as an anomaly score: frames far from the training distribution yield high
    reconstruction error because the learned prior cannot model them well.

    Args:
        latent_ch:  Number of channels in the bottleneck feature map (default 256).
        image_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, latent_ch: int = 256, image_size: int = 224):
        super().__init__()
        self.encoder = ConvEncoder(latent_ch=latent_ch)
        self.decoder = ConvDecoder(latent_ch=latent_ch)
        self.image_size = image_size

    def forward(self, corrupted: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction of a corrupted input.

        Args:
            corrupted: (B, 3, H, W) corrupted input tensor (normalised).

        Returns:
            (B, 3, H, W) reconstructed image tensor.
        """
        return self.decoder(self.encoder(corrupted))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode x to the bottleneck feature map (B, latent_ch, H//16, W//16)."""
        return self.encoder(x)


# -- Dataset -------------------------------------------------------------------


def _collect_frame_paths(frames_dir: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, "**", ext), recursive=True))
    return sorted(paths)


class DenoisingDataset(Dataset):
    """Returns (corrupted_frame, clean_frame) pairs from a frames directory.

    For each item, the clean frame is loaded and corruption is applied once to
    produce the noisy input.  The clean frame is the reconstruction target.

    Args:
        frames_dir:  Root directory containing JPEG/PNG frames (recursive search).
        transform:   Normalizing transform (applied before corruption).
        mode:        Corruption mode: "gaussian" | "masking" | "both".
        noise_std:   Gaussian noise std (default 0.2).
        patch_size:  Masking patch side length in pixels (default 16).
        mask_frac:   Fraction of patches to mask (default 0.15).
    """

    def __init__(
        self,
        frames_dir: str,
        transform: transforms.Compose,
        mode: str = "both",
        noise_std: float = 0.2,
        patch_size: int = 16,
        mask_frac: float = 0.15,
    ):
        self.paths = _collect_frame_paths(frames_dir)
        if not self.paths:
            raise ValueError(f"No frames found under {frames_dir!r}")
        self.transform = transform
        self.mode = mode
        self.noise_std = noise_std
        self.patch_size = patch_size
        self.mask_frac = mask_frac

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("RGB")
        clean = self.transform(img)
        corrupted = corrupt(
            clean.clone(),
            mode=self.mode,
            noise_std=self.noise_std,
            patch_size=self.patch_size,
            mask_frac=self.mask_frac,
        )
        return corrupted, clean


# -- Config --------------------------------------------------------------------


@dataclass
class DAEFinetuneConfig:
    frames_dir: str
    output_dir: str
    image_size: int = 224
    latent_ch: int = 256
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    corruption_mode: str = "both"
    noise_std: float = 0.2
    patch_size: int = 16
    mask_frac: float = 0.15
    num_workers: int = 4
    save_every: int = 1
    device: str = "cpu"
    seed: int = 42


# -- Training loop -------------------------------------------------------------


def run_dae_finetune(cfg: DAEFinetuneConfig) -> str:
    """Train a Denoising Autoencoder on unlabeled outdoor frames.

    Args:
        cfg: DAEFinetuneConfig instance.

    Returns:
        Path to the best checkpoint (lowest average reconstruction MSE).
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ensure_output_dir(cfg.output_dir)

    transform = build_dae_transform(cfg.image_size)
    dataset: Dataset = DenoisingDataset(
        cfg.frames_dir,
        transform=transform,
        mode=cfg.corruption_mode,
        noise_std=cfg.noise_std,
        patch_size=cfg.patch_size,
        mask_frac=cfg.mask_frac,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
        drop_last=True,
    )
    logger.info(
        "DAE training: %d frames | corruption=%s | epochs=%d | batch=%d | device=%s",
        len(dataset),
        cfg.corruption_mode,
        cfg.epochs,
        cfg.batch_size,
        cfg.device,
    )

    model = DenoisingAutoencoder(latent_ch=cfg.latent_ch, image_size=cfg.image_size).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_loss = float("inf")
    best_path = os.path.join(cfg.output_dir, "dae_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for corrupted, clean in loader:
            corrupted = corrupted.to(cfg.device)
            clean = clean.to(cfg.device)
            recon = model(corrupted)
            loss = F.mse_loss(recon, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        avg = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        logger.info(
            "DAE epoch %d/%d  mse=%.5f  lr=%.2e",
            epoch,
            cfg.epochs,
            avg,
            scheduler.get_last_lr()[0],
        )
        if epoch % cfg.save_every == 0:
            _ep_path = os.path.join(cfg.output_dir, f"dae_epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), _ep_path)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), best_path)
            logger.info("DAE new best: mse=%.5f -> %s", best_loss, best_path)

    encoder_path = os.path.join(cfg.output_dir, "dae_encoder.pt")
    torch.save(model.encoder.state_dict(), encoder_path)
    logger.info("DAE training complete. best_mse=%.5f  ckpt=%s", best_loss, best_path)
    return best_path


# -- Anomaly scoring -----------------------------------------------------------


class DAEAnomalyScorer:
    """Wraps a trained DenoisingAutoencoder for per-frame reconstruction scoring.

    Reconstruction MSE is the anomaly signal: frames the model struggles to
    reconstruct (novel scenes, strong lighting changes, degraded video) have
    high MSE.  No corruption is applied at inference -- the model reconstructs
    the raw frame and its MSE against the original is the score.

    Args:
        checkpoint_path:  Path to a saved DenoisingAutoencoder state dict.
        device:           Torch device string.
        image_size:       Input spatial resolution (must match training config).
        latent_ch:        Bottleneck channel count (must match training config).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        image_size: int = 224,
        latent_ch: int = 256,
    ):
        self.device = device
        self._transform = build_dae_transform(image_size)
        self._model = DenoisingAutoencoder(latent_ch=latent_ch, image_size=image_size)
        state = torch.load(checkpoint_path, map_location=device)
        self._model.load_state_dict(state)
        self._model.to(device).eval()
        logger.info("DAEAnomalyScorer loaded from %s", checkpoint_path)

    @torch.no_grad()
    def score_image(self, img: "PILImage.Image") -> float:
        """Return reconstruction MSE for a single PIL image."""
        x = self._transform(img).unsqueeze(0).to(self.device)
        return float(F.mse_loss(self._model(x), x).item())

    @torch.no_grad()
    def score_batch(self, images: "list[PILImage.Image]") -> list[float]:
        """Score a list of PIL images, returning per-image MSE scores."""
        if not images:
            return []
        tensors = torch.stack([self._transform(img) for img in images]).to(self.device)
        recon = self._model(tensors)
        per_frame = F.mse_loss(recon, tensors, reduction="none").mean(dim=(1, 2, 3))
        return per_frame.tolist()
