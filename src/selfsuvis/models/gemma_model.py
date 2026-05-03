"""Gemma open-weight local embedder.

Uses mean pooling of the last transformer hidden layer to produce
L2-normalised embedding vectors from text and images.

Supports any Gemma variant available on HuggingFace:
    google/gemma-4-it-2b   — Gemma 4 2B multimodal (images + audio), default
    google/gemma-4-it-4b   — Gemma 4 4B multimodal
    google/gemma-3-2b-it   — Gemma 3 2B text-only (well-tested, no vision)

Implements the same public interface as :class:`OpenCLIPEmbedder`:
    encode_images(), encode_texts(), image_dim(), text_dim()

Requirements:
    pip install transformers accelerate
    (torch is installed via the project environment)
"""

import hashlib
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger(__name__)

# Neutral prompt prepended to images so the model sees a well-formed input.
_IMAGE_PROMPT = "Describe this image:"

# Text prompt template for embedding; {text} is substituted.
_TEXT_PROMPT = "{text}"

# Maximum text tokens passed to the model (truncation threshold).
_MAX_TEXT_TOKENS = 512


class GemmaEmbedder:
    """Open-weight local embedder backed by a Gemma model.

    Loads the model once in ``__init__`` using ``transformers``.  Embeddings
    are produced by running a forward pass with ``output_hidden_states=True``
    and mean-pooling the last hidden layer over non-padding token positions,
    then L2-normalising the result.

    Image embeddings use the model's multimodal processor when available
    (Gemma 4 multimodal variants); text-only models fall back to a text
    description prompt.

    Parameters
    ----------
    model_id:
        HuggingFace model repository ID.
    device:
        ``"cpu"``, ``"cuda"``, or ``"auto"`` (picks CUDA if available).
    use_bf16:
        Use bfloat16 precision on GPU (recommended for Gemma; ignored on CPU).
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-it-2b",
        device: str = "cpu",
        use_bf16: bool = True,
        hf_token: str = "",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.bfloat16 if (use_bf16 and device != "cpu") else torch.float32

        # Resolve token: explicit arg > env var HUGGING_FACE_HUB_TOKEN > HF_TOKEN
        import os as _os

        from selfsuvis.pipeline.core.config import mask_secret as _mask_secret  # noqa: PLC0415
        token = hf_token or _os.getenv("HUGGING_FACE_HUB_TOKEN") or _os.getenv("HF_TOKEN") or None
        _log.info(
            "GemmaEmbedder: HF token %s",
            _mask_secret(token) if token else "<not set — may fail for gated models>",
        )

        _log.info("GemmaEmbedder: loading processor from %s …", model_id)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                use_fast=False,
                trust_remote_code=True,
                token=token,
            )
        except OSError:
            # Text-only Gemma models (e.g. gemma-3-1b-it) have no preprocessor_config.json.
            # Fall back to AutoTokenizer — image encoding will use the text-prompt path.
            _log.info(
                "  No preprocessor_config.json for %s — loading AutoTokenizer (text-only model)",
                model_id,
            )
            from transformers import AutoTokenizer  # noqa: PLC0415
            self.processor = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=token,
            )

        _log.info(
            "GemmaEmbedder: loading model from %s (dtype=%s, device=%s) …",
            model_id, dtype, device,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            token=token,
        )
        self.model.eval()

        self._model_id = model_id
        self._device   = device
        # Gemma 4 multimodal uses a nested text_config; plain Gemma has hidden_size directly.
        _cfg = self.model.config
        self._dim = (
            getattr(_cfg, "hidden_size", None)
            or getattr(getattr(_cfg, "text_config", None), "hidden_size", None)
            or self.model.get_input_embeddings().weight.shape[1]
        )
        # True when the processor accepts image inputs (multimodal variants)
        self._is_multimodal = hasattr(self.processor, "image_processor")
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._image_cache_max = 2048

        _log.info(
            "GemmaEmbedder ready  model=%s  dim=%d  multimodal=%s",
            model_id, self._dim, self._is_multimodal,
        )

    # ── Public interface (matches OpenCLIPEmbedder) ───────────────────────────

    def encode_images(
        self, images: list[Image.Image], batch_size: int = 4
    ) -> np.ndarray:
        """Return L2-normalised image embeddings, shape ``(N, dim)``."""
        if not images:
            return np.zeros((0, self._dim), dtype=np.float32)
        results: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            try:
                results.append(self._encode_images_cached(batch, batch_size))
            except Exception as exc:
                _log.warning("GemmaEmbedder: image batch %d failed (%s) — using zeros", start, exc)
                results.append(np.zeros((len(batch), self._dim), dtype=np.float32))
        arr = np.concatenate(results, axis=0)
        return self._l2_norm(arr)

    def encode_texts(
        self, texts: list[str], batch_size: int = 16
    ) -> np.ndarray:
        """Return L2-normalised text embeddings, shape ``(N, dim)``."""
        results: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                results.append(self._encode_text_batch(batch))
            except Exception as exc:
                _log.warning("GemmaEmbedder: text batch %d failed (%s) — using zeros", start, exc)
                results.append(np.zeros((len(batch), self._dim), dtype=np.float32))
        arr = np.concatenate(results, axis=0)
        return self._l2_norm(arr)

    def encode_images_temporal(
        self, images: list[Image.Image], batch_size: int = 4
    ) -> np.ndarray:
        """Return a single L2-normalised embedding for a sequence of frames.

        Computes per-frame embeddings then mean-pools them, producing one
        holistic temporal embedding of shape ``(1, dim)``.
        """
        if not images:
            return np.zeros((1, self._dim), dtype=np.float32)
        per_frame = self.encode_images(images, batch_size=batch_size)
        mean_vec  = per_frame.mean(axis=0, keepdims=True)
        return self._l2_norm(mean_vec)

    def image_dim(self) -> int:
        return self._dim

    def text_dim(self) -> int:
        return self._dim

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _input_device(self) -> torch.device:
        """Device where the model's embedding layer lives.

        With ``device_map`` (accelerate), individual layers may be placed on
        different devices.  Inputs must land on the *embedding* layer's device;
        accelerate's forward hooks move intermediate activations from there
        automatically.  Sending inputs to ``self._device`` (e.g. ``cuda:0``)
        when the embedding weight is still on CPU causes the
        ``index on cuda / weight on cpu`` mismatch that shows up as a warning.
        """
        try:
            return next(self.model.get_input_embeddings().parameters()).device
        except Exception:
            return self._torch.device(self._device)

    def _move_inputs_to_device(self, inputs):
        """Move tensor-like processor outputs to the embedding layer's device."""
        target = self._input_device()
        try:
            return inputs.to(target)
        except Exception:
            return {
                k: v.to(target) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

    def _build_multimodal_inputs(self, image: Image.Image):
        """Build a single-image multimodal prompt with an explicit image token."""
        # Strategy 1: processor chat template with structured image placeholder.
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _IMAGE_PROMPT},
                ],
            }]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return self.processor(
                text=text,
                images=[image.convert("RGB")],
                return_tensors="pt",
            )
        except AttributeError:
            pass
        except Exception as exc:
            _log.debug("GemmaEmbedder: processor chat template failed (%s)", exc)

        # Strategy 2: tokenizer chat template with explicit image placeholder.
        try:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            messages = [{
                "role": "user",
                "content": f"<|image_1|>\n{_IMAGE_PROMPT}",
            }]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return self.processor(
                text=text,
                images=[image.convert("RGB")],
                return_tensors="pt",
            )
        except Exception as exc:
            _log.debug("GemmaEmbedder: tokenizer chat template failed (%s)", exc)

        # Strategy 3: direct processor call for older processors.
        return self.processor(
            images=image.convert("RGB"),
            text=_IMAGE_PROMPT,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    @staticmethod
    def _image_cache_key(image: Image.Image) -> str:
        rgb = image.convert("RGB")
        digest = hashlib.sha1(rgb.tobytes()).hexdigest()
        return f"{rgb.size[0]}x{rgb.size[1]}:{digest}"

    def _cache_get_image(self, key: str) -> np.ndarray | None:
        cached = self._image_cache.get(key)
        if cached is None:
            return None
        self._image_cache.move_to_end(key)
        return cached.copy()

    def _cache_put_image(self, key: str, embedding: np.ndarray) -> None:
        self._image_cache[key] = embedding.astype(np.float32, copy=True)
        self._image_cache.move_to_end(key)
        while len(self._image_cache) > self._image_cache_max:
            self._image_cache.popitem(last=False)

    def _encode_image_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Forward pass for one batch of PIL images → numpy (batch, dim)."""
        if self._is_multimodal:
            # Gemma 4 multimodal: processor treats a list of images as multiple
            # images for ONE conversation, not N separate image-text pairs.
            # Process one image at a time and stack results.
            per_image: list[np.ndarray] = []
            for img in images:
                inputs = self._move_inputs_to_device(self._build_multimodal_inputs(img))
                with self._torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                mask = inputs.get("attention_mask")
                per_image.append(self._pool(last_hidden, mask))
            return np.concatenate(per_image, axis=0)
        else:
            # Text-only model: describe the image via its text encoder
            # Encode a neutral prompt (no actual vision; use for graceful fallback)
            dummy_texts = [_IMAGE_PROMPT] * len(images)
            inputs = self.processor(
                text=dummy_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16,
            )

        target = self._input_device()
        inputs = {
            k: v.to(target) for k, v in inputs.items()
            if isinstance(v, self._torch.Tensor)
        }

        with self._torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        mask = inputs.get("attention_mask")
        return self._pool(last_hidden, mask)

    def _encode_images_cached(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray | None] = [None] * len(images)
        misses: list[Image.Image] = []
        miss_indices: list[int] = []

        for idx, image in enumerate(images):
            key = self._image_cache_key(image)
            cached = self._cache_get_image(key)
            if cached is not None:
                outputs[idx] = cached
            else:
                misses.append(image)
                miss_indices.append(idx)

        if misses:
            computed = self._encode_image_batch(misses)
            for src_idx, emb in enumerate(computed):
                out_idx = miss_indices[src_idx]
                key = self._image_cache_key(images[out_idx])
                self._cache_put_image(key, emb)
                outputs[out_idx] = emb.astype(np.float32, copy=True)

        return np.stack([o for o in outputs if o is not None], axis=0)

    def _encode_text_batch(self, texts: list[str]) -> np.ndarray:
        """Forward pass for one batch of strings → numpy (batch, dim)."""
        prompts = [_TEXT_PROMPT.format(text=t) for t in texts]
        inputs  = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_MAX_TEXT_TOKENS,
        )
        target = self._input_device()
        inputs = {k: v.to(target) for k, v in inputs.items()
                  if isinstance(v, self._torch.Tensor)}

        with self._torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]
        mask = inputs.get("attention_mask")
        return self._pool(last_hidden, mask)

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor | None) -> np.ndarray:
        """Mean pool *hidden* over non-padding positions, return numpy array."""
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (hidden.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.float().mean(dim=1)
        return pooled.cpu().numpy()

    @staticmethod
    def _l2_norm(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)
