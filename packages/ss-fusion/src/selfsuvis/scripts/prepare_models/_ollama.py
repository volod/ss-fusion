"""Ollama / backend helpers and UniDrive/reasoning model constants."""

import importlib.util
import shutil
import subprocess
import time

from selfsuvis.pipeline.core.logging import get_logger

log = get_logger("prepare_models")

_UNIDRIVE_DEFAULT_MODEL = "owl10/UniDriveVLA_Nusc_Base_Stage3"
_UNIDRIVE_COLLECTION_URL = "https://huggingface.co/collections/owl10/unidrivevla"
_UNIDRIVE_OLLAMA_FALLBACK_MODEL = "qwen2.5vl:7b"

# Reasoning / agentic-audit model (step 24).  Served via Ollama by default.
# deepseek-r1:14b is a strong alternative if qwen3 is not available.
_REASONING_DEFAULT_MODEL = "qwen3:14b"


def _has_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def _has_vllm_installed() -> bool:
    return importlib.util.find_spec("vllm") is not None


def _is_ollama_model_name(model_id: str) -> bool:
    mid = (model_id or "").strip()
    return bool(mid) and "/" not in mid


def _resolve_unidrive_backend(requested_backend: str, model_id: str) -> str:
    """Choose the backend used to prepare UniDrive assets.

    UniDriveVLA is published on HuggingFace only — it is not available on Ollama.
    vllm is required to serve HF UniDrive repos.  Ollama is only valid for
    Ollama-native model tags (no slash in name).
    """
    have_ollama = _has_ollama_installed()
    have_vllm = _has_vllm_installed()

    backend = (requested_backend or "auto").strip().lower()
    if backend not in {"", "auto", "ollama", "vllm"}:
        raise ValueError(f"Unsupported UniDrive backend: {requested_backend}")
    if backend == "ollama":
        if not have_ollama:
            raise RuntimeError(
                "UniDrive backend 'ollama' requested, but 'ollama' is not installed on this machine."
            )
        return "ollama"
    if backend == "vllm":
        if not have_vllm:
            raise RuntimeError(
                "UniDrive requires vllm (UniDriveVLA is not available on Ollama). "
                "Install vllm: pip install vllm — or disable UniDrive by setting "
                "UNIDRIVE_ENABLED=false in .env."
            )
        return "vllm"

    # Auto mode: Ollama-tagged models (no slash) can run on Ollama.
    # HuggingFace UniDriveVLA repos require vLLM — they are not published on Ollama.
    if _is_ollama_model_name(model_id):
        if have_ollama:
            return "ollama"
        if have_vllm:
            return "vllm"
    if have_vllm:
        return "vllm"
    raise RuntimeError(
        "UniDriveVLA is not available on Ollama. "
        "Install vllm (pip install vllm) to use the HuggingFace model, "
        "or set UNIDRIVE_ENABLED=false in .env to skip this step."
    )


def _resolve_unidrive_prepare_model(model_id: str, backend: str) -> str:
    """Resolve the model artifact to warm up for the chosen UniDrive backend."""
    requested = (model_id or "").strip()
    if backend == "ollama":
        if requested and _is_ollama_model_name(requested):
            return requested
        if requested and requested.startswith("owl10/UniDriveVLA"):
            log.warning(
                "UniDriveVLA repo '%s' is published on Hugging Face, not in the Ollama library. "
                "Using Ollama fallback model '%s' for UniDrive-style sidecar serving.",
                requested,
                _UNIDRIVE_OLLAMA_FALLBACK_MODEL,
            )
        return _UNIDRIVE_OLLAMA_FALLBACK_MODEL
    if requested:
        return requested
    return _UNIDRIVE_DEFAULT_MODEL


def _is_ollama_model_cached(model: str) -> bool:
    if not _has_ollama_installed():
        return False
    result = subprocess.run(
        ["ollama", "show", model],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _download_ollama_model(model: str) -> None:
    if not _has_ollama_installed():
        raise RuntimeError("Cannot pull Ollama model because 'ollama' is not installed.")
    log.info("Ollama — model=%s", model)
    if _is_ollama_model_cached(model):
        log.info("  [ok] Ollama model already present — skipping pull")
        return
    t0 = time.monotonic()
    result = subprocess.run(["ollama", "pull", model], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama pull {model!r} failed. Ensure the Ollama daemon is running and the model tag exists."
        )
    log.info("  [ok] Ollama model ready  (%.1fs)", time.monotonic() - t0)
