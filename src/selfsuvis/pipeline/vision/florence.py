"""Florence-2-large image captioning model wrapper.

Loaded at VideoIndexer.__init__() — if weights are missing the worker crashes on
startup rather than failing mid-mission.

caption_batch() contract:
    - Input:  List[PIL.Image.Image]
    - Output: List[Tuple[str, float]]  — (caption_text, confidence)
    - len(result) == len(input) always (stable ordering)
    - Per-image exception → ("", 0.5) for that index; never crashes the batch
    - confidence: mean of softmax(scores[i])[generated_token_id] across all generated
      tokens, clamped to [0.0, 1.0]. Falls back to 0.5 when scores unavailable.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from selfsuvis.pipeline.core import get_logger, resolve_device, settings

_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"
_MODEL_ID = "microsoft/Florence-2-large"
_MODEL_BASE_NAME = "florence-2-large"

logger = get_logger(__name__)


def _best_attn_impl() -> str:
    """Return the best available attention backend: flash_attention_2 > sdpa."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


class FlorenceModel:
    """Florence-2-large captioner. Load once; call caption_batch() many times."""

    def __init__(self) -> None:
        # Import here so the module is importable even without transformers installed
        # (unit tests that don't touch this class won't need the heavy deps).
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.47 is required for Florence-2 captioning. "
                "Install it with: pip install 'transformers>=4.47'"
            ) from exc

        self.device = self._resolve_device()
        logger.info("Loading Florence-2-large on %s …", self.device)

        # Clear any leftover VRAM fragmentation before loading the model.
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # dtype: FP16 on CUDA, FP32 on CPU
        torch_dtype = torch.float16 if (self.device == "cuda" and settings.USE_FP16) else torch.float32
        source = _resolve_local_model_source(_MODEL_ID)
        source_label = str(source) if isinstance(source, Path) else _MODEL_ID
        load_common_kwargs = {
            "trust_remote_code": True,
            "local_files_only": isinstance(source, Path),
        }

        self._processor = AutoProcessor.from_pretrained(
            source_label, **load_common_kwargs
        )
        attn_impl = _best_attn_impl()
        # Flash Attention 2.0 requires float16 or bfloat16; fall back to sdpa for float32.
        if attn_impl == "flash_attention_2" and torch_dtype == torch.float32:
            attn_impl = "sdpa"
        load_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_impl,
        }
        load_kwargs.update(load_common_kwargs)
        # Flash Attention 2.0 requires weights to land on GPU at load time.
        # Single-GPU: use {"": 0} to skip accelerate's conservative 90/10 memory
        # estimation (which emits an INFO log and unnecessarily caps usable VRAM).
        # Multi-GPU: fall back to "auto" so accelerate distributes layers.
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto" if torch.cuda.device_count() > 1 else {"": 0}
        self._model = AutoModelForCausalLM.from_pretrained(source_label, **load_kwargs)
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()

        logger.info(
            "Florence-2-large ready (device=%s, dtype=%s)",
            self.device,
            torch_dtype,
        )

    def release(self) -> None:
        """Unload model weights from GPU to free VRAM for subsequent models."""
        import gc
        if getattr(self, "_model", None) is not None:
            try:
                # Move to CPU first so accelerate's device_map hooks release GPU pages.
                self._model.cpu()
            except Exception:
                pass
            del self._model
            self._model = None  # type: ignore[assignment]
        if getattr(self, "_processor", None) is not None:
            del self._processor
            self._processor = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ── public API ────────────────────────────────────────────────────────────

    def caption_batch(
        self,
        images: List[Image.Image],
        batch_size: int | None = None,
    ) -> List[Tuple[str, float]]:
        """Caption a list of PIL images.

        Returns a list of (caption, confidence) tuples, one per input image.
        Stable order: result[i] corresponds to images[i].
        Any per-image failure returns ("", 0.5) for that index.
        """
        if not images:
            return []

        effective_batch = batch_size if batch_size is not None else settings.FLORENCE_BATCH_SIZE
        results: List[Tuple[str, float]] = []

        for batch_start in range(0, len(images), effective_batch):
            batch = images[batch_start : batch_start + effective_batch]
            batch_results = self._caption_batch_chunk(batch, effective_batch)
            results.extend(batch_results)

        return results

    @property
    def model_tag(self) -> str:
        """Structured provenance string stored in frames.caption_model.

        Format: "{model}:{prompt_version}:{precision}"
        Example: "florence-2-large:v1:fp16"

        Bump FLORENCE_PROMPT_VERSION (env var) whenever the task prompt or
        post-processing changes so that existing captions remain distinguishable
        from ones generated under a different prompt.
        """
        precision = "fp16" if (self.device == "cuda" and settings.USE_FP16) else "fp32"
        return f"{_MODEL_BASE_NAME}:{settings.FLORENCE_PROMPT_VERSION}:{precision}"

    # ── internals ─────────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        return resolve_device()

    def _caption_batch_chunk(
        self,
        images: List[Image.Image],
        batch_size: int,
    ) -> List[Tuple[str, float]]:
        """Caption one chunk, with OOM fallback to batch=1."""
        try:
            return self._run_inference(images)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 1:
                logger.warning(
                    "Florence OOM on batch_size=%d; falling back to batch=1", batch_size
                )
                torch.cuda.empty_cache()
                return [self._caption_single(img) for img in images]
            raise

    def _run_inference(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        """Run Florence inference on a list of images (same batch)."""
        prompts = [_TASK_PROMPT] * len(images)

        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        model_dtype = next(self._model.parameters()).dtype
        inputs = {
            k: v.to(model_dtype) if v.is_floating_point() else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=256,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        sequences = generated.sequences
        input_ids_len = inputs["input_ids"].shape[1]
        generated_ids = sequences[:, input_ids_len:]

        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions = []
        for raw, img in zip(decoded, images):
            parsed = self._processor.post_process_generation(
                raw,
                task=_TASK_PROMPT,
                image_size=(img.width, img.height),
            )
            text = parsed.get(_TASK_PROMPT, raw)
            captions.append(text.strip() if isinstance(text, str) else "")

        confidences = _compute_confidences(generated.scores, generated_ids)
        return list(zip(captions, confidences))

    def _caption_single(self, image: Image.Image) -> Tuple[str, float]:
        """Caption one image, returning ("", 0.5) on any error."""
        try:
            results = self._run_inference([image])
            return results[0]
        except Exception:
            logger.warning("Florence failed on a single image; returning empty caption", exc_info=True)
            return ("", 0.5)


def _resolve_local_model_source(model_id: str) -> str | Path:
    """Prefer an already-cached HF snapshot to avoid noisy HEAD retries on startup."""
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
        )
        path = Path(local_dir)
        if path.exists():
            logger.info("Florence cache hit: %s → %s", model_id, path)
            return path
    except Exception:
        pass
    return model_id


# ── confidence computation ────────────────────────────────────────────────────


def _compute_confidences(
    scores: tuple | None,
    generated_ids: torch.Tensor,
) -> List[float]:
    """Compute mean token probability for each sequence in the batch.

    scores: tuple of (vocab_size,) or (batch, vocab_size) tensors, one per step.
    generated_ids: (batch, seq_len) tensor of chosen token ids.

    Returns a list of floats in [0.0, 1.0], one per batch item.
    Falls back to 0.5 when scores are unavailable or seq_len is zero.
    """
    batch_size = generated_ids.shape[0]

    if scores is None or len(scores) == 0:
        return [0.5] * batch_size

    seq_len = generated_ids.shape[1]
    if seq_len == 0:
        return [0.5] * batch_size

    # Accumulate per-token probabilities for each sequence
    # sum_probs[b] sums the chosen-token probability across all non-padding positions
    sum_probs = [0.0] * batch_size
    count = [0] * batch_size

    for step_idx, step_logits in enumerate(scores):
        if step_idx >= seq_len:
            break
        # step_logits: (batch, vocab_size)
        if step_logits.dim() == 1:
            # single-item batch flattened — expand
            step_logits = step_logits.unsqueeze(0)

        probs = torch.softmax(step_logits.float(), dim=-1)  # (batch, vocab)
        chosen_ids = generated_ids[:, step_idx]            # (batch,)

        for b in range(batch_size):
            tok = chosen_ids[b].item()
            if tok == 1:  # padding / EOS token typically 1 for Florence
                continue
            prob = probs[b, tok].item()
            sum_probs[b] += prob
            count[b] += 1

    confidences = []
    for b in range(batch_size):
        if count[b] == 0:
            confidences.append(0.5)
        else:
            raw = sum_probs[b] / count[b]
            confidences.append(float(max(0.0, min(1.0, raw))))

    return confidences
