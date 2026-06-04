"""HuggingFace model downloaders: OpenCLIP, Whisper, Florence-2, OCR, Depth, Detection,
WorldModel, UniDriveVLA."""

import time

from selfsuvis.pipeline.core.logging import get_logger

from ._cache import _is_hf_cached, _is_florence2_complete
from ._utils import _quiet_hf, _label, _capture_hf_load_report

log = get_logger("prepare_models")


def _download_openclip(model: str, pretrained: str, device: str) -> None:
    from ._cache import _is_openclip_cached

    log.info("OpenCLIP — model=%s  pretrained=%s  device=%s", model, pretrained, device)
    if _is_openclip_cached(model, pretrained):
        log.info("  [ok] OpenCLIP already cached — skipping load")
        return
    import open_clip

    t0 = time.monotonic()
    open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    log.info("  [ok] OpenCLIP ready  (%.1fs)", time.monotonic() - t0)


def _download_whisper(model_id: str) -> None:
    log.info("Whisper ASR — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  [ok] Whisper already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from transformers import pipeline as _hf_pipeline

        with _capture_hf_load_report(model_id):
            _hf_pipeline("automatic-speech-recognition", model=model_id, device="cpu")
        log.info("  [ok] Whisper ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Whisper download failed: %s", exc)
        raise


def _download_florence(model_id: str = "microsoft/Florence-2-large") -> None:
    log.info("Florence-2 — model=%s", model_id)
    if _is_hf_cached(model_id) and _is_florence2_complete(model_id):
        log.info("  [ok] Florence-2 already cached — skipping load")
        return
    if _is_hf_cached(model_id):
        log.info(
            "  Partial Florence-2 cache detected (modeling_florence2.py missing) — re-downloading"
        )
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        log.info("  [ok] Florence-2 cached  (%.1fs)  cache=%s", time.monotonic() - t0, local_dir)
    except Exception as exc:
        log.warning("  Florence-2 download failed: %s", exc)
        raise


def _download_ocr(model_id: str) -> None:
    """Download (or verify) OCR model weights.

    Dispatches to the correct loader based on model family:
    - TrOCR (microsoft/trocr-*): TrOCRProcessor + VisionEncoderDecoderModel
    - GOT-OCR2 (ucaslcl/GOT-*): AutoTokenizer + AutoModel with trust_remote_code
    - VLM family (Phi-3.5-vision, Qwen2.5-VL, DeepSeek-OCR-2, llava-hf/*):
      AutoProcessor + AutoModelForCausalLM with trust_remote_code
    - Florence-2 (microsoft/Florence-*): AutoProcessor + AutoModelForCausalLM
    """
    log.info("OCR model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  [ok] OCR model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        if model_id.startswith("microsoft/trocr-"):
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            TrOCRProcessor.from_pretrained(model_id)
            VisionEncoderDecoderModel.from_pretrained(model_id)
        elif model_id.startswith("ucaslcl/GOT-"):
            from transformers import AutoModel, AutoTokenizer

            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor

            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            for _attn_impl in ("sdpa", "eager"):
                try:
                    AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        dtype="auto",
                        low_cpu_mem_usage=True,
                        attn_implementation=_attn_impl,
                    )
                    break
                except (ValueError, NotImplementedError) as _fa2_exc:
                    if "Flash Attention 2" not in str(_fa2_exc):
                        raise
                    if _attn_impl != "eager":
                        log.info(
                            "  %s: sdpa blocked by FA2 guard, retrying with eager",
                            model_id,
                        )
                        continue
                    log.warning(
                        "  %s: FA2 guard fires even with eager attn"
                        " (flash-attn installed + model lacks FA2 support)."
                        " Weights cached; load-verify skipped."
                        " Will load at inference time with attn_implementation='eager'.",
                        model_id,
                    )
                    break
        log.info("  [ok] OCR model ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  OCR model download failed: %s", exc)
        raise


def _download_depth(model_id: str) -> None:
    """Download depth-estimation model weights via HF transformers pipeline."""
    log.info("Depth model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  [ok] Depth model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from transformers import pipeline as _hf_pipeline

        _hf_pipeline("depth-estimation", model=model_id, device="cpu")
        log.info("  [ok] Depth model ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Depth model download failed: %s", exc)
        raise


def _download_detection(model_id: str) -> None:
    """Download object-detection model weights via HF transformers pipeline."""
    log.info("Detection model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  [ok] Detection model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from transformers import pipeline as _hf_pipeline

        with _quiet_hf():
            _hf_pipeline("object-detection", model=model_id, device="cpu")
        log.info("  [ok] Detection model ready  (%.1fs)", time.monotonic() - t0)
    except Exception as exc:
        log.warning("  Detection model download failed: %s", exc)
        raise


def _download_world_model(model_id: str) -> None:
    """Download world-model weights.

    Tries AutoFeatureExtractor + AutoModel first (works for VideoMAE, VJEPA2, etc.).
    Falls back to snapshot_download for models that lack preprocessor_config.json
    (e.g. nvidia/Cosmos-1.0-Autoregressive-4B which is a generative autoregressive model).
    """
    log.info("World model — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  [ok] World model already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoFeatureExtractor, AutoModel

        try:
            AutoFeatureExtractor.from_pretrained(model_id)
        except (OSError, ValueError) as feat_exc:
            feat_msg = str(feat_exc)
            if (
                "does not appear to have a file named" in feat_msg
                or "Unrecognized feature extractor" in feat_msg
                or "Unrecognized" in feat_msg
            ):
                log.info("  No compatible feature extractor — downloading repo via snapshot_download")
                local_dir = snapshot_download(
                    repo_id=model_id,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
                )
                log.info(
                    "  [ok] World model cached at %s  (%.1fs)", local_dir, time.monotonic() - t0
                )
                return
            raise
        AutoModel.from_pretrained(model_id, torch_dtype="auto")
        local_dir = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        log.info("  [ok] World model ready  (%.1fs)  cache=%s", time.monotonic() - t0, local_dir)
    except Exception as exc:
        log.warning("  World model download failed: %s", exc)
        raise


def _download_unidrive(model_id: str) -> None:
    """Download UniDriveVLA model assets for local bridges / sidecars."""
    log.info("UniDriveVLA — model=%s", model_id)
    if _is_hf_cached(model_id):
        log.info("  [ok] UniDriveVLA already cached — skipping load")
        return
    t0 = time.monotonic()
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            log.info(
                "  Transformers warmup skipped for %s (%s); repository cache is still ready",
                model_id,
                exc,
            )
        log.info("  [ok] UniDriveVLA ready  (%.1fs)  cache=%s", time.monotonic() - t0, local_dir)
    except Exception as exc:
        log.warning("  UniDriveVLA download failed: %s", exc)
        raise
