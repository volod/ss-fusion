"""Shared environment loading and typed access helpers."""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


def project_roots(anchor_file: str) -> tuple[Path, Path]:
    current = Path(anchor_file).resolve()
    package_root = current.parents[2]
    repo_root = current.parents[4]
    return package_root, repo_root


def load_layered_env(
    *,
    anchor_file: str,
    app_env: str | None = None,
    package_env_dir: str = "env",
    root_env_filename: str = ".env",
) -> None:
    """Load packaged and repo-root env files without overriding existing vars."""
    env_name = app_env or os.getenv("APP_ENV", "dev")
    package_root, repo_root = project_roots(anchor_file)
    package_env = package_root / package_env_dir / f"{env_name}.env"
    root_env = repo_root / root_env_filename

    packaged_values: dict[str, str | None] = (
        dotenv_values(package_env) if package_env.exists() else {}
    )
    root_values: dict[str, str | None] = dotenv_values(root_env) if root_env.exists() else {}
    for key, value in {**packaged_values, **root_values}.items():
        if key not in os.environ:
            os.environ[key] = value if value is not None else ""


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, "true" if default else "false").strip().lower() == "true"


def env_int(key: str, default: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    raw = os.getenv(key, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def env_csv(key: str, default: Iterable[str] = ()) -> list[str]:
    raw = os.getenv(key, "")
    if not raw.strip():
        return [str(item).strip() for item in default if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_json_dict(
    key: str,
    *,
    default: dict[str, str] | None = None,
    on_error=None,
) -> dict[str, str]:
    fallback = dict(default or {})
    raw = os.getenv(key, "")
    if not raw:
        return fallback
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if on_error is not None:
            on_error("%s contains invalid JSON; using default value", key)
        return fallback
    if not isinstance(parsed, dict):
        if on_error is not None:
            on_error("%s must be a JSON object; using default value", key)
        return fallback
    return {str(k): str(v) for k, v in parsed.items()}


def set_env_if_present(key: str, value: Any) -> None:
    if value not in (None, ""):
        os.environ[key] = str(value)

