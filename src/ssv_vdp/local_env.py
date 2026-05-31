"""Local full-analysis environment setup helpers.

IMPORTANT — import order matters
---------------------------------
selfsuvis.pipeline.core.__init__ creates ``settings = Settings()`` the first time
any sub-module of selfsuvis.pipeline.core is imported.  Settings() reads every
``os.environ`` value exactly once at instantiation.

``apply_local_env`` must therefore set all CLI-driven env vars (enable/disable
flags, API URLs, model IDs) **before** any selfsuvis.pipeline import.  The function
is split into three phases to make this invariant explicit.
"""

import os
from pathlib import Path
from typing import Any


def normalize_local_orchestration_args(args: Any) -> Any:
    """Apply default-on local orchestration flags unless explicitly overridden."""
    if getattr(args, "mode", "") != "local":
        return args

    default_true_flags = (
        "asr",
        "ocr",
        "depth",
        "detection",
        "world_model",
        "qwen",
        "unidrive",
    )
    for flag in default_true_flags:
        if getattr(args, flag, None) is None:
            setattr(args, flag, True)
    return args


def _phase1_force_cli_env(args: Any) -> None:
    """Phase 1 — force-set every CLI-driven env var BEFORE selfsuvis is imported.

    Uses direct assignment (not setdefault) so these values are guaranteed
    to be visible to Settings() when it is instantiated inside the
    ``from selfsuvis.pipeline.core.env import ...`` import in apply_local_env.
    """
    # ── Enable / disable flags ────────────────────────────────────────────────
    os.environ["ASR_ENABLED"]         = "true" if args.asr       else "false"
    os.environ["OCR_ENABLED"]         = "true" if args.ocr       else "false"
    os.environ["DEPTH_ENABLED"]       = "true" if args.depth     else "false"
    os.environ["DETECTION_ENABLED"]   = "true" if args.detection else "false"
    os.environ["WORLD_MODEL_ENABLED"] = "true" if args.world_model else "false"
    os.environ["UNIDRIVE_ENABLED"]    = "true" if args.unidrive  else "false"

    os.environ["YOLO_ENABLED"]   = "false" if getattr(args, "no_yolo",  False) else "true"
    os.environ["SAM_ENABLED"]    = "false" if getattr(args, "no_sam",   False) else "true"
    os.environ["RFDETR_ENABLED"] = "false" if getattr(args, "no_rfdetr",False) else "true"

    # ── Model tier overrides ──────────────────────────────────────────────────
    _rfdetr_model = getattr(args, "rfdetr_model", "base") or "base"
    os.environ["RFDETR_MODEL"] = _rfdetr_model          # always honour CLI flag
    _yolo_model = getattr(args, "yolo_model", "yolo11l") or "yolo11l"
    os.environ.setdefault("YOLO_MODEL", _yolo_model)    # yield to shell/env
    _sam_model = getattr(args, "sam_model", "auto") or "auto"
    os.environ.setdefault("SAM_MODEL", _sam_model)

    if getattr(args, "world_model_id", "auto") != "auto":
        os.environ["WORLD_MODEL"] = args.world_model_id

    # ── Gemma sidecar ─────────────────────────────────────────────────────────
    if getattr(args, "gemma_api_url", ""):
        os.environ["GEMMA_API_URL"] = args.gemma_api_url
    if getattr(args, "gemma_api_model", ""):
        os.environ["GEMMA_API_MODEL"] = args.gemma_api_model
    if getattr(args, "gemma_api_backend", ""):
        os.environ["GEMMA_API_BACKEND"] = args.gemma_api_backend

    # ── Qwen sidecar ──────────────────────────────────────────────────────────
    if getattr(args, "qwen_api_url", ""):
        os.environ["QWEN_API_URL"] = args.qwen_api_url
    if getattr(args, "qwen_model", ""):
        os.environ["QWEN_MODEL"] = args.qwen_model
    if getattr(args, "qwen_backend", ""):
        os.environ["QWEN_BACKEND"] = args.qwen_backend

    # ── UniDrive sidecar ──────────────────────────────────────────────────────
    if getattr(args, "unidrive_api_url", ""):
        os.environ["UNIDRIVE_API_URL"] = args.unidrive_api_url
    if getattr(args, "unidrive_model", ""):
        os.environ["UNIDRIVE_MODEL"] = args.unidrive_model
    if getattr(args, "unidrive_backend", ""):
        os.environ["UNIDRIVE_BACKEND"] = args.unidrive_backend

    # Fallback: when --unidrive is enabled without an explicit API URL, reuse
    # the Qwen endpoint (same Ollama instance, compatible v1 endpoint).
    if (
        args.unidrive
        and not os.environ.get("UNIDRIVE_API_URL")
        and os.environ.get("QWEN_API_URL")
    ):
        os.environ["UNIDRIVE_API_URL"] = os.environ["QWEN_API_URL"]
        if not os.environ.get("UNIDRIVE_BACKEND"):
            os.environ["UNIDRIVE_BACKEND"] = os.environ.get("QWEN_BACKEND", "ollama")

    # ── Reasoning sidecar ─────────────────────────────────────────────────────
    if getattr(args, "reasoning_api_url", ""):
        os.environ["REASONING_API_URL"] = args.reasoning_api_url
    if getattr(args, "reasoning_model", ""):
        os.environ["REASONING_MODEL"] = args.reasoning_model
    if getattr(args, "reasoning_backend", ""):
        os.environ["REASONING_BACKEND"] = args.reasoning_backend

    # ── Florence sidecar ──────────────────────────────────────────────────────
    if getattr(args, "florence_api_url", ""):
        os.environ["FLORENCE_API_URL"] = args.florence_api_url
    if getattr(args, "florence_model", ""):
        os.environ["FLORENCE_MODEL"] = args.florence_model

    # ── SceneTok sidecar ──────────────────────────────────────────────────────
    _scenetok = getattr(args, "scenetok", None)
    if _scenetok is not None:
        os.environ["SCENETOK_ENABLED"] = "true" if _scenetok else "false"
    if getattr(args, "scenetok_api_url", ""):
        os.environ["SCENETOK_API_URL"] = args.scenetok_api_url
    if getattr(args, "scenetok_checkpoint", ""):
        os.environ["SCENETOK_CHECKPOINT"] = args.scenetok_checkpoint


def apply_local_env(args: Any) -> None:
    """Set environment variables for local full-analysis mode.

    Args:
        args: Parsed argparse namespace for local full-analysis mode.
    """
    # Step 1 — apply defaults to tri-state flags still at None.
    normalize_local_orchestration_args(args)

    # Step 2 — force-set all CLI-driven values BEFORE any selfsuvis import.
    # This ensures Settings() (instantiated during the import below) sees
    # the correct values rather than frozen-in defaults from the .env.
    _phase1_force_cli_env(args)

    # Step 3 — import and run layered .env loader.
    # load_layered_env uses "if key not in os.environ" so it cannot override
    # anything we set in phase 1.
    from selfsuvis.pipeline.core.env import load_layered_env  # noqa: PLC0415

    load_layered_env(anchor_file=__file__)

    # Step 4 — resolve SceneTok tri-state from .env if still None after phase 1.
    if getattr(args, "scenetok", None) is None:
        _env_val = os.environ.get("SCENETOK_ENABLED", "").strip().lower()
        if _env_val in {"true", "1", "yes", "on"}:
            args.scenetok = True
            os.environ["SCENETOK_ENABLED"] = "true"
        elif _env_val in {"false", "0", "no", "off"}:
            args.scenetok = False

    # Step 5 — setdefault fills for values not covered by CLI or .env.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    _hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if _hf:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DATA_DIR", str(output_dir))
    os.environ.setdefault("MODEL_NAME", "gemma")
    os.environ.setdefault("GEMMA_MODEL_ID", "google/gemma-4-it-2b")
    os.environ.setdefault("GEMMA_USE_BF16", "true" if args.device != "cpu" else "false")
    os.environ.setdefault("QDRANT_HOST", "localhost")
    os.environ.setdefault("QDRANT_PORT", "6333")
    os.environ.setdefault("QDRANT_COLLECTION", "local_video_semantic")
    os.environ.setdefault("DEVICE", args.device)
    os.environ.setdefault("USE_FP16", "false")
    os.environ.setdefault("SAMPLE_FPS_MAX", str(args.fps))
    os.environ.setdefault("SFM_FPS", "1")
    os.environ.setdefault("ALLOWED_INDEX_PATHS", "")
    os.environ.setdefault("API_KEY", "")

    os.environ.setdefault("ASR_MODEL", args.asr_model)
    os.environ.setdefault("ASR_LANGUAGE", args.asr_language)
    os.environ.setdefault("OCR_MODEL", args.ocr_model)
    os.environ.setdefault("DEPTH_MODEL", args.depth_model)
    os.environ.setdefault("DETECTION_MODEL", args.detection_model)
    os.environ.setdefault("DETECTION_LABELS", args.detection_labels)
    os.environ.setdefault("WORLD_MODEL", "auto")

    os.environ.setdefault("GEMMA_API_URL", "")
    os.environ.setdefault("QWEN_API_URL", "")
    os.environ.setdefault("UNIDRIVE_API_URL", "")
    os.environ.setdefault("UNIDRIVE_MODEL", "")
    os.environ.setdefault("UNIDRIVE_BACKEND", "")
    os.environ.setdefault("SCENETOK_API_URL", "")
    os.environ.setdefault("REASONING_API_URL", "")

    if args.unidrive and not os.environ.get("UNIDRIVE_BACKEND"):
        os.environ.setdefault("UNIDRIVE_BACKEND", "ollama")
