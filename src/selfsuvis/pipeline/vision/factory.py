import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.pipeline.core import get_logger, settings

from .labels import load_labels


@dataclass
class MaskSegment:
    segment_id: str
    label: str
    bbox: tuple[int, int, int, int]
    mean_color: tuple[int, int, int]
    area: int


class OpenCLIPTagger:
    def __init__(self, labels: list[str] | None = None, labels_file: str | None = None):
        self.logger = get_logger(__name__)
        if labels is not None:
            self.labels = labels
        else:
            self.labels = load_labels(labels_file or settings.LABELS_FILE)
        self.embedder = OpenCLIPEmbedder()
        self._text_embeddings = None

    def _ensure_text_embeddings(self) -> None:
        if self._text_embeddings is not None:
            return
        prompts = [f"a photo of {label}" for label in self.labels]
        self._text_embeddings = self.embedder.encode_texts(prompts)

    def describe_image(self, image: Image.Image, top_k: int = 3) -> dict[str, Any]:
        self._ensure_text_embeddings()
        img_emb = self.embedder.encode_images([image], batch_size=1)[0]
        sims = np.dot(self._text_embeddings, img_emb)
        top_idx = np.argsort(-sims)[:top_k]
        results = [{"label": self.labels[i], "score": float(sims[i])} for i in top_idx]
        return {"labels": results}

    def label_segments(self, crops: list[Image.Image], top_k: int = 1) -> list[dict[str, Any]]:
        self._ensure_text_embeddings()
        if not crops:
            return []
        img_embs = self.embedder.encode_images(crops, batch_size=8)
        sims = img_embs @ self._text_embeddings.T
        labels = []
        for row in sims:
            top_idx = np.argsort(-row)[:top_k]
            labels.append(
                [{"label": self.labels[i], "score": float(row[i])} for i in top_idx]
            )
        return labels


class SAMSegmenter:
    def __init__(self, model_type: str | None = None, checkpoint: str | None = None):
        self.logger = get_logger(__name__)
        self.model_type = model_type or settings.SAM_MODEL_TYPE
        self.checkpoint = checkpoint or settings.SAM_CHECKPOINT
        self._mask_generator = None

    def _init(self) -> None:
        if self._mask_generator is not None:
            return
        if not self.checkpoint or not os.path.exists(self.checkpoint):
            raise RuntimeError("SAM checkpoint missing. Set SAM_CHECKPOINT env or --sam-checkpoint")
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        device = "cuda" if settings.DEVICE != "cpu" and _cuda_available() else "cpu"
        sam.to(device=device)
        self._mask_generator = SamAutomaticMaskGenerator(sam)
        self.logger.info("SAM loaded type=%s device=%s", self.model_type, device)

    def segment(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        self._init()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._mask_generator.generate(rgb)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def mask_to_segments(
    frame_bgr: np.ndarray,
    masks: list[dict[str, Any]],
    tagger: OpenCLIPTagger,
    max_masks: int = 20,
) -> list[MaskSegment]:
    segments: list[MaskSegment] = []
    crops: list[Image.Image] = []
    raw_masks: list[np.ndarray] = []

    for i, mask in enumerate(sorted(masks, key=lambda m: m.get("area", 0), reverse=True)):
        if i >= max_masks:
            break
        seg_mask = mask["segmentation"].astype(np.uint8)
        x, y, w, h = mask["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        crop = frame_bgr[y : y + h, x : x + w].copy()
        if crop.size == 0:
            continue
        mask_crop = seg_mask[y : y + h, x : x + w]
        crop[mask_crop == 0] = 0
        crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        raw_masks.append(mask_crop)

    labels = tagger.label_segments(crops, top_k=1)
    for idx, mask in enumerate(sorted(masks, key=lambda m: m.get("area", 0), reverse=True)[: len(crops)]):
        x, y, w, h = mask["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        seg_mask = raw_masks[idx]
        crop = frame_bgr[y : y + h, x : x + w].copy()
        if crop.size == 0:
            continue
        masked = crop[seg_mask > 0]
        if masked.size == 0:
            mean_color = (0, 0, 0)
        else:
            mean_color = tuple(int(v) for v in masked.mean(axis=0).tolist())
        label = labels[idx][0]["label"] if labels else "segment"
        segments.append(
            MaskSegment(
                segment_id=f"mask_{idx}",
                label=label,
                bbox=(x, y, w, h),
                mean_color=mean_color,
                area=int(mask.get("area", 0)),
            )
        )
    return segments
