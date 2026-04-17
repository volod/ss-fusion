from typing import List, Optional

import numpy as np
import torch
import open_clip
from PIL import Image

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.gpu_utils import is_cuda_oom, resolve_device
from selfsuvis.pipeline.core.logging import get_logger


class OpenCLIPEmbedder:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.device = self._resolve_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.OPENCLIP_MODEL,
            pretrained=settings.OPENCLIP_PRETRAINED,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(settings.OPENCLIP_MODEL)
        self.logger.info(
            "OpenCLIP loaded: %s (%s) on %s",
            settings.OPENCLIP_MODEL,
            settings.OPENCLIP_PRETRAINED,
            self.device,
        )

    def _resolve_device(self) -> str:
        return resolve_device()

    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            try:
                actual_device = next(self.model.parameters()).device
            except StopIteration:
                actual_device = torch.device(self.device)
            try:
                tensors = torch.stack([self.preprocess(img) for img in batch]).to(actual_device)
                with torch.no_grad():
                    if settings.USE_FP16 and str(actual_device).startswith("cuda"):
                        with torch.cuda.amp.autocast():
                            feats = self.model.encode_image(tensors)
                    else:
                        feats = self.model.encode_image(tensors)
            except Exception as exc:
                if not is_cuda_oom(exc) or not str(actual_device).startswith("cuda"):
                    raise
                self.logger.warning(
                    "OpenCLIP CUDA OOM during image encoding; moving backbone to CPU for remaining batches."
                )
                self.model.cpu()
                actual_device = torch.device("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tensors = torch.stack([self.preprocess(img) for img in batch]).to(actual_device)
                with torch.no_grad():
                    feats = self.model.encode_image(tensors)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.detach().cpu().numpy())
        if not embeddings:
            return np.zeros((0, self.model.visual.output_dim), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                actual_device = next(self.model.parameters()).device
            except StopIteration:
                actual_device = torch.device(self.device)
            try:
                tokens = self.tokenizer(batch).to(actual_device)
                with torch.no_grad():
                    if settings.USE_FP16 and str(actual_device).startswith("cuda"):
                        with torch.cuda.amp.autocast():
                            feats = self.model.encode_text(tokens)
                    else:
                        feats = self.model.encode_text(tokens)
            except Exception as exc:
                if not is_cuda_oom(exc) or not str(actual_device).startswith("cuda"):
                    raise
                self.logger.warning(
                    "OpenCLIP CUDA OOM during text encoding; moving backbone to CPU."
                )
                self.model.cpu()
                actual_device = torch.device("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                tokens = self.tokenizer(batch).to(actual_device)
                with torch.no_grad():
                    feats = self.model.encode_text(tokens)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.append(feats.detach().cpu().numpy())
        if not embeddings:
            return np.zeros((0, self.model.text.output_dim), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    def image_dim(self) -> int:
        return self.model.visual.output_dim

    def text_dim(self) -> int:
        return self.model.text.output_dim


