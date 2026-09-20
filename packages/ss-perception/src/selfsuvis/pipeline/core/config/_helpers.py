"""Env-reading helpers and small utilities shared by all settings modules."""

import logging as _logging

from selfsuvis.pipeline.core.env import (
    env_float,
    env_int,
    env_json_dict,
    env_str,
)
from ss_kit.security import parse_path_allowlist
from ss_kit.settings import mask_secret

_log = _logging.getLogger(__name__)

__all__ = ["get_dino_model_name", "mask_secret"]


def _env(key: str, default: str) -> str:
    return env_str(key, default)


def _env_int(key: str, default: int) -> int:
    return env_int(key, default)


def _env_float(key: str, default: float) -> float:
    return env_float(key, default)


def _env_json_dict(key: str, default: dict[str, str] | None = None) -> dict[str, str]:
    """Parse a JSON object from env, returning the safe default on invalid values."""
    return env_json_dict(key, default=default, on_error=_log.warning)


def _parse_allowed_paths(val: str | None) -> list[str]:
    """Parse ALLOWED_INDEX_PATHS as a comma-separated list. Empty disables path endpoints."""
    return parse_path_allowlist(val)


def get_dino_model_name(model_name: str) -> str | None:
    """Resolve configured model family to a concrete DINO backbone name."""
    if model_name == "dinov2":
        return "dinov2_vitb14"
    if model_name == "dinov3":
        return "dinov3_vitb14"
    return None
