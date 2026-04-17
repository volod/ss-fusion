"""EfficientViT-S1 image embedder via timm.

Provides a drop-in replacement for :class:`models.dino_model.DINOEmbedder` for
knowledge distillation and edge-deployment scenarios.  EfficientViT-S1 produces
384-dimensional L2-normalised image embeddings at roughly 2× the throughput of
DINOv3 ViT-S/14 at the same embedding dimension.

Reference: `timm.create_model("efficientvit_b1", pretrained=True)`

Usage::

    from selfsuvis.models.efficientvit_model import EfficientViTEmbedder
    model = EfficientViTEmbedder(device="cuda")
    embs  = model.encode_images([pil_img1, pil_img2])  # (N, 384) float32 numpy

For distillation (Stage 1→2 — DINOv3 teacher → EfficientViT student)::

    from selfsuvis.pipeline.training.distill import DistillConfig, run_distillation_efficientvit
    cfg  = DistillConfig(student_model="efficientvit_b1", lambda_rkd_a=0.0, ...)
    stats = run_distillation_efficientvit(teacher_backbone, frame_paths, ckpt_dir, cfg)
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_EFFICIENTVIT_INPUT_SIZE = 224
_EFFICIENTVIT_MEAN = (0.485, 0.456, 0.406)
_EFFICIENTVIT_STD  = (0.229, 0.224, 0.225)


class EfficientViTEmbedder:
    """EfficientViT-B1 image embedder (384-dim, L2-normalised).

    Loads from timm with ``pretrained=True`` by default.  The classification
    head is discarded; global average pooling over the final feature map is
    used as the embedding.

    Args:
        device:    ``"cuda"``, ``"cpu"``, or ``"auto"`` (prefers CUDA).
        use_fp16:  Enable FP16 inference on CUDA (reduces VRAM, slightly faster).
        pretrained: Whether to load timm pretrained weights.
    """

    def __init__(
        self,
        device: str = "auto",
        use_fp16: bool = True,
        pretrained: bool = True,
    ) -> None:
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._use_fp16 = use_fp16 and device == "cuda"

        try:
            import timm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "timm is required for EfficientViTEmbedder: pip install timm"
            ) from exc

        try:
            self._model = timm.create_model(
                "efficientvit_b1", pretrained=pretrained, num_classes=0
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "CUDA out of memory" in str(exc):
                free_gb = 0.0
                try:
                    free_gb = torch.cuda.mem_get_info()[0] / 1e9
                except Exception:
                    pass
                raise RuntimeError(
                    f"EfficientViT Stage 1→2 requires ≥8GB VRAM; "
                    f"detected {free_gb:.1f}GB free. "
                    "Try: device='cpu', or free VRAM before loading."
                ) from exc
            raise

        if self._use_fp16:
            self._model = self._model.half()
        self._model = self._model.to(device).eval()

        # Resolve output dimension via a dummy forward pass
        import torch
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, _EFFICIENTVIT_INPUT_SIZE, _EFFICIENTVIT_INPUT_SIZE,
                                 device=device, dtype=torch.float16 if self._use_fp16 else torch.float32)
            self._dim: int = int(self._model(_dummy).shape[-1])

        logger.info("EfficientViTEmbedder ready on %s (dim=%d, fp16=%s)", device, self._dim, self._use_fp16)

    def image_dim(self) -> int:
        """Return the embedding dimensionality (384 for EfficientViT-B1)."""
        return self._dim

    def _preprocess(self, images: List[Image.Image]):
        """Preprocess PIL images → (B, 3, H, W) float tensor."""
        import torch
        import torchvision.transforms.functional as TF

        tensors = []
        for img in images:
            img = img.convert("RGB").resize(
                (_EFFICIENTVIT_INPUT_SIZE, _EFFICIENTVIT_INPUT_SIZE), Image.BICUBIC
            )
            t = TF.to_tensor(img)
            t = TF.normalize(t, mean=list(_EFFICIENTVIT_MEAN), std=list(_EFFICIENTVIT_STD))
            tensors.append(t)
        batch = torch.stack(tensors).to(self._device)
        if self._use_fp16:
            batch = batch.half()
        return batch

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Embed a list of PIL images.

        Returns:
            float32 numpy array of shape (N, dim), L2-normalised.
        """
        if not images:
            return np.zeros((0, self._dim), dtype=np.float32)

        import torch

        batch = self._preprocess(images)
        with torch.no_grad():
            try:
                embs = self._model(batch)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    free_gb = 0.0
                    try:
                        free_gb = torch.cuda.mem_get_info()[0] / 1e9
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"EfficientViT Stage 1→2 requires ≥8GB VRAM; "
                        f"detected {free_gb:.1f}GB free."
                    ) from exc
                raise

        embs = embs.float().cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (embs / norms).astype(np.float32)

    def as_torch_backbone(self):
        """Return the underlying timm model for use as a distillation backbone."""
        return self._model
