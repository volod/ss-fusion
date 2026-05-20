"""Runtime configuration loaded from layered .env files and environment variables."""

import os
from pathlib import Path as _Path

from selfsuvis.pipeline.core.env import load_layered_env as _load_layered_env
from selfsuvis.pipeline.core.logging import get_logger

# Load .env files before any _env() calls execute in the mixin class bodies.
# anchor_file points at core/env.py so project_roots() resolves the same
# package_root and repo_root as the old core/config.py did.
_load_layered_env(anchor_file=str(_Path(__file__).parent.parent / "env.py"))

from ._helpers import (  # noqa: E402
    _env,
    _env_float,
    _env_int,
    _env_json_dict,
    _parse_allowed_paths,
    get_dino_model_name,
    mask_secret,
)
from ._models import _ModelSettings
from ._pipeline import _PipelineSettings
from ._security import _SecuritySettings
from ._services import _ServiceSettings
from ._training import _TrainingSettings

logger = get_logger(__name__)


class Settings(
    _ModelSettings,
    _PipelineSettings,
    _ServiceSettings,
    _TrainingSettings,
    _SecuritySettings,
):
    """Flat settings namespace populated from environment variables at import time."""

    APP_ENV = _env("APP_ENV", "dev").strip().lower()
    DATA_DIR = _env("DATA_DIR", "./data")
    FRAMES_DIR = _env("FRAMES_DIR", os.path.join(DATA_DIR, "frames"))
    TILES_DIR = _env("TILES_DIR", os.path.join(DATA_DIR, "tiles"))
    VIDEOS_DIR = _env("VIDEOS_DIR", os.path.join(DATA_DIR, "videos"))

    # HuggingFace authentication -- required for gated models (Gemma, Llama, etc.).
    # Set HF_TOKEN=hf_... in .env or export it before running.
    # Alternatively run: huggingface-cli login
    HF_TOKEN = _env("HF_TOKEN", _env("HUGGING_FACE_HUB_TOKEN", ""))

    MODEL_NAME = _env("MODEL_NAME", "openclip")

    DEVICE = _env("DEVICE", "auto")
    USE_FP16 = _env("USE_FP16", "true").lower() == "true"
    LOCAL_CUDA_STAGE_MIN_FREE_VRAM_GB = _env_float("LOCAL_CUDA_STAGE_MIN_FREE_VRAM_GB", 6.0)

    # Robot identity (used to tag Qdrant payloads for multi-robot filtering)
    ROBOT_ID = _env("ROBOT_ID", "robot_0")

    # Worker identity (used in gpu_jobs table for resource isolation)
    WORKER_ID = _env("WORKER_ID", __import__("socket").gethostname())
    # Maximum seconds a gpu_jobs entry may be held before it is considered stale.
    GPU_JOB_TIMEOUT_SEC = _env_int("GPU_JOB_TIMEOUT_SEC", 3600)

    # Model version provenance: stored in Qdrant frame payloads at embed/re-embed time.
    # Set automatically by the worker after a successful supervised fine-tune.
    MODEL_VERSION_ID = _env("MODEL_VERSION_ID", "base")

    VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".avi"})


settings = Settings()


def validate_settings() -> None:
    """Validate critical settings at startup. Raises ValueError on invalid config."""
    if settings.QDRANT_PORT < 1 or settings.QDRANT_PORT > 65535:
        raise ValueError("QDRANT_PORT must be 1-65535")
    if settings.MODEL_NAME not in {"openclip", "dinov2", "dinov3", "gemma"}:
        raise ValueError("MODEL_NAME must be openclip, dinov2, dinov3, or gemma")
    if settings.MAX_UPLOAD_BYTES < 0 or settings.MAX_DOWNLOAD_BYTES < 0:
        raise ValueError("MAX_UPLOAD_BYTES and MAX_DOWNLOAD_BYTES must be non-negative")
    if settings.FFMPEG_TIMEOUT_SEC < 1:
        raise ValueError("FFMPEG_TIMEOUT_SEC must be >= 1")
    if settings.TILE_SIZE < 1 or settings.STRIDE < 1:
        raise ValueError("TILE_SIZE and STRIDE must be >= 1")
    if (
        settings.MOTION_LOW < 0
        or settings.MOTION_HIGH < 0
        or settings.MOTION_LOW > settings.MOTION_HIGH
    ):
        raise ValueError(
            "MOTION_LOW and MOTION_HIGH must be non-negative and MOTION_LOW <= MOTION_HIGH"
        )
    if settings.SAMPLE_FPS_MIN <= 0 or settings.SAMPLE_FPS_MAX < settings.SAMPLE_FPS_MIN:
        raise ValueError("SAMPLE_FPS_MIN must be > 0 and SAMPLE_FPS_MAX >= SAMPLE_FPS_MIN")
    if settings.MAX_REDIRECTS < 0:
        raise ValueError("MAX_REDIRECTS must be >= 0")
    if settings.RATE_LIMIT_PER_MIN < 0 or settings.RATE_LIMIT_BURST < 0:
        raise ValueError("RATE_LIMIT_PER_MIN and RATE_LIMIT_BURST must be >= 0")
    if settings.MAX_IMAGE_PIXELS < 0:
        raise ValueError("MAX_IMAGE_PIXELS must be >= 0")
    if settings.MAX_DIR_FILES < 0 or settings.MAX_DIR_BYTES < 0 or settings.MAX_DIR_DEPTH < 0:
        raise ValueError("MAX_DIR_FILES/MAX_DIR_BYTES/MAX_DIR_DEPTH must be >= 0")
    if settings.HF_TOKEN:
        logger.info("HF_TOKEN configured: %s", mask_secret(settings.HF_TOKEN))
    if settings.API_AUTH_REQUIRED and not settings.API_KEY:
        raise ValueError("API_KEY must be set when API_AUTH_REQUIRED is true")
    if not settings.API_KEY:
        logger.warning(
            "API_KEY is not set; the API is unauthenticated and open to any caller. "
            "Set the API_KEY environment variable or enable API_AUTH_REQUIRED."
        )
    if not settings.CVAT_WEBHOOK_SECRET:
        logger.warning(
            "CVAT_WEBHOOK_SECRET is not set; POST /webhook/cvat will reject all requests "
            "(fail-closed). Set CVAT_WEBHOOK_SECRET to the secret configured in CVAT's "
            "webhook settings to enable annotation webhook delivery."
        )
    if not settings.ALLOWED_INDEX_PATHS:
        logger.warning(
            "ALLOWED_INDEX_PATHS is not set; path-based indexing endpoints "
            "(/index/video path=, /index/dir, /index/precheck path=, /index/precheck_dir) "
            "are disabled. Set ALLOWED_INDEX_PATHS to a comma-separated list of allowed "
            "base directories."
        )
    logger.info("Settings validated successfully")


__all__ = [
    "get_dino_model_name",
    "mask_secret",
    "Settings",
    "settings",
    "validate_settings",
]
