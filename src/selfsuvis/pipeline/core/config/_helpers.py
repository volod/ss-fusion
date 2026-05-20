"""Env-reading helpers and small utilities shared by all settings modules."""

import logging as _logging
import os

from selfsuvis.pipeline.core.env import (
    env_float,
    env_int,
    env_json_dict,
    env_str,
)

_log = _logging.getLogger(__name__)


def mask_secret(value: str, visible_suffix: int = 4) -> str:
    """Return *value* with all but the last *visible_suffix* chars replaced by '*'.

    Safe to pass to any logger. Examples::

        mask_secret("hf_abcdefghijklmnopqrstuv")  -> "*********************stuv"
        mask_secret("")                            -> "<not set>"
        mask_secret("hi")                          -> "*i"
    """
    if not value:
        return "<not set>"
    if len(value) <= visible_suffix:
        return "*" * (len(value) - 1) + value[-1]
    return "*" * (len(value) - visible_suffix) + value[-visible_suffix:]


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
    """Parse ALLOWED_INDEX_PATHS as comma-separated list. Empty means no restriction."""
    if val is None or not val.strip():
        return []
    return [p.strip() for p in val.split(",") if p.strip()]


def get_dino_model_name(model_name: str) -> str | None:
    """Resolve configured model family to a concrete DINO backbone name."""
    if model_name == "dinov2":
        return "dinov2_vitb14"
    if model_name == "dinov3":
        return "dinov3_vitb14"
    return None
