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

import torch
from PIL import Image

from selfsuvis.pipeline.core import get_logger, resolve_device, settings
from selfsuvis.pipeline.vision._quiet import suppress_runtime_noise

_TASK_PROMPT = "<MORE_DETAILED_CAPTION>"
_MODEL_ID = "microsoft/Florence-2-large"
_MODEL_BASE_NAME = "florence-2-large"

logger = get_logger(__name__)

_TRANSFORMERS5_PATCHES_APPLIED = False


def _apply_florence2_transformers5_patches() -> None:
    """Shim transformers 4.x APIs that Florence-2's trust_remote_code still uses.

    transformers 5.0 made two breaking changes that affect Florence-2's cached
    Python files (processing_florence2.py / configuration_florence2.py):

    1. ``PretrainedConfig.forced_bos_token_id`` removed — add None class default.
    2. ``tokenizer.additional_special_tokens`` renamed to ``extra_special_tokens``
       — patch PreTrainedTokenizerBase.__getattr__ / __setattr__ to redirect the
       old name to the new backing store (_extra_special_tokens).

    Patches are applied at most once per process (guarded by the module-level flag).
    """
    global _TRANSFORMERS5_PATCHES_APPLIED
    if _TRANSFORMERS5_PATCHES_APPLIED:
        return
    _TRANSFORMERS5_PATCHES_APPLIED = True

    # 1. forced_bos_token_id — removed from PretrainedConfig in transformers 5.0.
    try:
        from transformers import PretrainedConfig as _PC

        if not hasattr(_PC, "forced_bos_token_id"):
            _PC.forced_bos_token_id = None
    except Exception:
        pass

    # 2. additional_special_tokens → extra_special_tokens rename.
    #    Florence-2's processing_florence2.py reads `tokenizer.additional_special_tokens`.
    #    In transformers 5.x, PreTrainedTokenizerBase.__getattr__ only handles the new
    #    name `extra_special_tokens`; the old name falls through to AttributeError.
    #    We wrap __getattr__ and __setattr__ to redirect the deprecated name.
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTTB

        if getattr(_PTTB, "_florence2_compat_patched", False):
            return  # already patched (e.g. second FlorenceModel instance)

        _orig_getattr = _PTTB.__getattr__
        _orig_setattr = _PTTB.__setattr__

        def _compat_getattr(self, key: str):
            if key == "additional_special_tokens":
                return [str(t) for t in self.__dict__.get("_extra_special_tokens", [])]
            if key == "additional_special_tokens_ids":
                toks = [str(t) for t in self.__dict__.get("_extra_special_tokens", [])]
                return self.convert_tokens_to_ids(toks)
            return _orig_getattr(self, key)

        def _compat_setattr(self, key: str, value):
            if key == "additional_special_tokens":
                key = "extra_special_tokens"
            _orig_setattr(self, key, value)

        _PTTB.__getattr__ = _compat_getattr
        _PTTB.__setattr__ = _compat_setattr
        _PTTB._florence2_compat_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass


def _best_attn_impl() -> str:
    """Return the best available attention backend for Florence-2.

    Florence-2's trust_remote_code model class does not implement _supports_sdpa,
    so sdpa triggers an AttributeError on load.  Eager is the only safe choice.
    """
    return "eager"


def _sanitize_model_inputs(inputs: object, *, device: str, dtype: torch.dtype) -> dict:
    """Drop None-valued processor fields and move tensors to the target device/dtype."""
    if hasattr(inputs, "items"):
        raw_items = dict(inputs.items())
    else:
        raw_items = dict(inputs)

    cleaned: dict = {}
    for key, value in raw_items.items():
        if value is None:
            continue
        if hasattr(value, "to"):
            if value.is_floating_point():
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        cleaned[key] = value
    return cleaned


def _extract_generated_token_ids(
    sequences: torch.Tensor,
    scores: tuple | None,
) -> torch.Tensor:
    """Return only the newly generated token ids.

    Florence variants can behave like seq2seq models where ``sequences`` already
    contains only generated tokens, while some decoder-style paths may include a
    prompt prefix. ``generated.scores`` reliably reports how many new tokens were
    produced, so use that length when available instead of trimming by
    ``input_ids`` length.
    """
    if scores is None or len(scores) == 0:
        return sequences[:, 0:0]
    generated_len = min(len(scores), int(sequences.shape[1]))
    if generated_len <= 0:
        return sequences[:, 0:0]
    return sequences[:, -generated_len:]


def _normalise_sequences(sequences: object) -> torch.Tensor:
    """Validate and normalise generation output sequences."""
    if sequences is None:
        raise RuntimeError("Florence generate returned no sequences")
    if not isinstance(sequences, torch.Tensor):
        raise RuntimeError(
            f"Florence generate returned unsupported sequences type: {type(sequences)!r}"
        )
    if sequences.ndim != 2:
        raise RuntimeError(
            f"Florence generate returned unexpected sequences shape: {tuple(sequences.shape)!r}"
        )
    return sequences


def _scores_are_usable(scores: object) -> bool:
    """Return True when scores look like per-step logits tensors."""
    if not isinstance(scores, tuple) or len(scores) == 0:
        return False
    return all(isinstance(step, torch.Tensor) for step in scores)


def _build_generate_kwargs(inputs: dict, *, include_scores: bool) -> dict:
    """Build Florence generation kwargs for the current runtime mode.

    Florence eager-attention fallback has been unstable in beam-search/cache
    paths on some transformers builds. Force a simple greedy decode with cache
    disabled so local runs prefer reliability over richer generation metadata.
    """
    kwargs = {
        **inputs,
        "max_new_tokens": 256,
        "return_dict_in_generate": True,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": False,
        "output_scores": include_scores,
    }
    # Some Florence checkpoints inherit stale generation_config fields such as
    # early_stopping; explicitly remove them from the runtime kwargs path.
    kwargs["generation_config"] = None
    return kwargs


def _is_recoverable_batch_assertion(exc: BaseException) -> bool:
    """Return True for Florence batch-shape assertions that single-image retries can avoid."""
    if not isinstance(exc, AssertionError):
        return False
    return "only support square feature maps for now" in str(exc).lower()


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad a PIL image to square with black borders (letterbox/pillarbox).

    Florence-2's vision encoder asserts h*w == num_tokens which only holds for
    square inputs.  Padding preserves aspect ratio and is lossless — the model
    simply ignores the padding region.
    """
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    square = Image.new(image.mode, (side, side), 0)
    square.paste(image, ((side - w) // 2, (side - h) // 2))
    return square


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

        # -- transformers 5.x compatibility patches -----------------------------
        # Florence-2 uses trust_remote_code whose cached Python files were written
        # for transformers 4.x.  Two attributes were removed/renamed in 5.x and
        # must be shimmed before AutoProcessor/AutoModelForCausalLM are called.
        _apply_florence2_transformers5_patches()

        # Clear any leftover VRAM fragmentation before loading the model.
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self._generation_mode = "scored"

        # dtype: FP16 on CUDA, FP32 on CPU
        torch_dtype = (
            torch.float16 if (self.device == "cuda" and settings.USE_FP16) else torch.float32
        )
        source = _resolve_local_model_source(_MODEL_ID)
        source_label = str(source) if isinstance(source, Path) else _MODEL_ID
        load_common_kwargs = {
            "trust_remote_code": True,
            "local_files_only": isinstance(source, Path),
        }

        with suppress_runtime_noise(
            r".*Florence2Processor.*image_processor_class = 'CLIPImageProcessor'.*deprecated.*",
            r".*doesn't directly inherit from `GenerationMixin`.*",
            r".*will lose the ability to call `generate` and other related functions.*",
            logger_levels={"transformers": logging.ERROR},
        ):
            self._processor = AutoProcessor.from_pretrained(source_label, **load_common_kwargs)
        attn_impl = _best_attn_impl()
        # Flash Attention 2.0 requires float16 or bfloat16; fall back to sdpa for float32.
        if attn_impl == "flash_attention_2" and torch_dtype == torch.float32:
            attn_impl = "sdpa"
        load_kwargs: dict = {
            # transformers 5.x now prefers `dtype`; older trust_remote_code
            # Florence paths may still expect `torch_dtype`, so we fall back.
            "dtype": torch_dtype,
            "attn_implementation": attn_impl,
        }
        load_kwargs.update(load_common_kwargs)
        # Flash Attention 2.0 requires weights to land on GPU at load time.
        # Single-GPU: use {"": 0} to skip accelerate's conservative 90/10 memory
        # estimation (which emits an INFO log and unnecessarily caps usable VRAM).
        # Multi-GPU: fall back to "auto" so accelerate distributes layers.
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto" if torch.cuda.device_count() > 1 else {"": 0}
        try:
            with suppress_runtime_noise(
                r".*doesn't directly inherit from `GenerationMixin`.*",
                r".*will lose the ability to call `generate` and other related functions.*",
                logger_levels={"transformers": logging.ERROR},
            ):
                self._model = AutoModelForCausalLM.from_pretrained(source_label, **load_kwargs)
        except TypeError as exc:
            # Older Florence-2 trust_remote_code paths may still reject `dtype`
            # and require the legacy `torch_dtype` kwarg instead.
            if "torch_dtype" not in str(exc) and "dtype" not in str(exc):
                raise
            fallback_kwargs = {
                k: v for k, v in load_kwargs.items() if k not in ("dtype", "attn_implementation")
            }
            fallback_kwargs["torch_dtype"] = torch_dtype
            fallback_kwargs["attn_implementation"] = "eager"
            with suppress_runtime_noise(
                r".*doesn't directly inherit from `GenerationMixin`.*",
                r".*will lose the ability to call `generate` and other related functions.*",
                logger_levels={"transformers": logging.ERROR},
            ):
                self._model = AutoModelForCausalLM.from_pretrained(source_label, **fallback_kwargs)
        except AttributeError as exc:
            # Florence-2 custom code (trust_remote_code) may lack _supports_sdpa in
            # some transformers versions — retry without attn_implementation.
            if "_supports_sdpa" not in str(exc) and "attn_implementation" not in str(exc):
                raise
            logger.warning("Florence-2 SDPA check failed (%s) — retrying with eager attention", exc)
            fallback_kwargs = {k: v for k, v in load_kwargs.items() if k != "attn_implementation"}
            fallback_kwargs["attn_implementation"] = "eager"
            with suppress_runtime_noise(
                r".*doesn't directly inherit from `GenerationMixin`.*",
                r".*will lose the ability to call `generate` and other related functions.*",
                logger_levels={"transformers": logging.ERROR},
            ):
                self._model = AutoModelForCausalLM.from_pretrained(source_label, **fallback_kwargs)
            self._generation_mode = "eager"
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "early_stopping"):
            generation_config.early_stopping = None

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

    # -- public API ------------------------------------------------------------

    def caption_batch(
        self,
        images: list[Image.Image],
        batch_size: int | None = None,
    ) -> list[tuple[str, float]]:
        """Caption a list of PIL images.

        Returns a list of (caption, confidence) tuples, one per input image.
        Stable order: result[i] corresponds to images[i].
        Any per-image failure returns ("", 0.5) for that index.
        """
        if not images:
            return []

        effective_batch = batch_size if batch_size is not None else settings.FLORENCE_BATCH_SIZE
        results: list[tuple[str, float]] = []

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

    @property
    def runtime_mode(self) -> str:
        return str(getattr(self, "_generation_mode", "scored"))

    # -- internals -------------------------------------------------------------

    def _resolve_device(self) -> str:
        return resolve_device()

    def _caption_batch_chunk(
        self,
        images: list[Image.Image],
        batch_size: int,
    ) -> list[tuple[str, float]]:
        """Caption one chunk, with OOM fallback to batch=1."""
        try:
            return self._run_inference(images)
        except AssertionError as exc:
            if _is_recoverable_batch_assertion(exc) and batch_size > 1:
                logger.warning(
                    "Florence batch assertion on batch_size=%d (%s); retrying one image at a time",
                    batch_size,
                    exc,
                )
                return [self._caption_single(img) for img in images]
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 1:
                from selfsuvis.pipeline.core.gpu_utils import log_oom_banner

                log_oom_banner(
                    logger,
                    "Florence-2",
                    f"batch_size={batch_size} → clearing cache, retrying one image at a time",
                )
                torch.cuda.empty_cache()
                return [self._caption_single(img) for img in images]
            raise

    def _run_inference(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        """Run Florence inference on a list of images (same batch)."""
        prompts = [_TASK_PROMPT] * len(images)
        # Florence-2 requires square feature maps; pad non-square frames in place.
        images = [_pad_to_square(img) for img in images]

        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        model_dtype = next(self._model.parameters()).dtype
        inputs = _sanitize_model_inputs(inputs, device=self.device, dtype=model_dtype)
        if "pixel_values" not in inputs:
            raise RuntimeError("Florence processor did not produce pixel_values for image batch")
        if "input_ids" not in inputs:
            raise RuntimeError("Florence processor did not produce input_ids for prompt batch")

        generated = self._generate_with_fallback(inputs)
        sequences = _normalise_sequences(getattr(generated, "sequences", None))
        scores = getattr(generated, "scores", None)
        generated_ids = _extract_generated_token_ids(
            sequences, scores if _scores_are_usable(scores) else None
        )

        decoded = self._processor.batch_decode(sequences, skip_special_tokens=True)
        captions = []
        for raw, img in zip(decoded, images):
            parsed = self._processor.post_process_generation(
                raw,
                task=_TASK_PROMPT,
                image_size=(img.width, img.height),
            )
            text = parsed.get(_TASK_PROMPT, raw)
            captions.append(text.strip() if isinstance(text, str) else "")

        confidences = _compute_confidences(
            scores if _scores_are_usable(scores) else None, generated_ids
        )
        return list(zip(captions, confidences))

    def _generate_with_fallback(self, inputs: dict):
        """Generate captions, retrying without scores when the scored path is unstable."""
        should_try_scored = self._generation_mode != "eager"
        if should_try_scored:
            try:
                with (
                    suppress_runtime_noise(
                        r"The following generation flags are not valid and may be ignored: .*",
                        logger_levels={
                            "transformers": logging.ERROR,
                            "transformers.generation.configuration_utils": logging.ERROR,
                        },
                    ),
                    torch.no_grad(),
                ):
                    generated = self._model.generate(
                        **_build_generate_kwargs(inputs, include_scores=True)
                    )
                if _normalise_sequences(getattr(generated, "sequences", None)).shape[0] == 0:
                    raise RuntimeError(
                        "Florence scored generation returned an empty sequence batch"
                    )
                self._generation_mode = "scored"
                return generated
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    raise
                logger.warning(
                    "Florence scored generation failed (%s) — retrying in caption-only mode",
                    exc,
                )

        with (
            suppress_runtime_noise(
                r"The following generation flags are not valid and may be ignored: .*",
                logger_levels={
                    "transformers": logging.ERROR,
                    "transformers.generation.configuration_utils": logging.ERROR,
                },
            ),
            torch.no_grad(),
        ):
            generated = self._model.generate(**_build_generate_kwargs(inputs, include_scores=False))
        if _normalise_sequences(getattr(generated, "sequences", None)).shape[0] == 0:
            raise RuntimeError("Florence caption-only generation returned an empty sequence batch")
        self._generation_mode = (
            "eager-noscores" if self._generation_mode == "eager" else "caption-only"
        )
        return generated

    def _caption_single(self, image: Image.Image) -> tuple[str, float]:
        """Caption one image, returning ("", 0.5) on any error."""
        try:
            results = self._run_inference([image])
            return results[0]
        except Exception:
            logger.warning(
                "Florence failed on a single image; returning empty caption", exc_info=True
            )
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


# -- confidence computation ----------------------------------------------------


def _compute_confidences(
    scores: tuple | None,
    generated_ids: torch.Tensor,
) -> list[float]:
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
        chosen_ids = generated_ids[:, step_idx]  # (batch,)

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
