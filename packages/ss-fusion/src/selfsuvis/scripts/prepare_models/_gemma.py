"""Gemma open-weight model download."""

import os

from selfsuvis.pipeline.core.logging import get_logger

from ._auth import _with_auth_retry
from ._cache import _is_hf_cached

log = get_logger("prepare_models")


def _is_gemma_cached(model_id: str) -> bool:
    """Return True if Gemma weights are in the local HuggingFace cache."""
    return _is_hf_cached(model_id)


def _download_gemma(model_id: str) -> None:
    """Download Gemma weights from HuggingFace and warm up the processor/tokenizer.

    Gemma is a gated model — requires accepting the license on HuggingFace
    and setting HF_TOKEN in .env (or running ``huggingface-cli login``).

    Setup (one-time):
      1. Accept the license at https://huggingface.co/google/gemma-3-4b-it
      2. Add HF_TOKEN=hf_... to your .env file
      3. Run: selfsuvis-models --gemma

    The function uses :func:`_with_auth_retry` so it prints interactive
    authentication instructions and retries on access errors.
    """
    from selfsuvis.pipeline.core.config import mask_secret

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or None
    log.info("Gemma — downloading %s  (token: %s) …", model_id, mask_secret(token or ""))

    def _do_download() -> None:
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        log.info("  Downloading processor/tokenizer …")
        try:
            AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
            log.info("  AutoProcessor loaded (multimodal model)")
        except OSError:
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
            log.info("  AutoTokenizer loaded (text-only model — no vision support)")
        log.info("  Downloading model weights (this may take a while) …")
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            token=token,
        )
        log.info("  [ok] Gemma %s cached", model_id)

    _with_auth_retry(f"Gemma ({model_id})", model_id, _do_download)
