"""Shared environment loading and typed access helpers."""

import os
from pathlib import Path

from ss_kit.env import (
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_json_dict,
    env_str,
    set_env_if_present,
)
from ss_kit.settings import load_layered_env as load_kit_layered_env

__all__ = [
    "env_bool",
    "env_csv",
    "env_float",
    "env_int",
    "env_json_dict",
    "env_str",
    "load_layered_env",
    "load_script_env",
    "project_roots",
    "set_env_if_present",
]


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
    root_env_filename: str = ".data/.env",
) -> None:
    """Load packaged defaults and repo-local .env files without overriding existing vars.

    Load order (later entries win over earlier, os.environ always wins):
      1. {package}/env/{app_env}.env  -- packaged defaults
      2. {repo_root}/.env             -- top-level user overrides (HF_TOKEN, etc.)
      3. {repo_root}/.data/.env       -- stack env (sencoop / test infra overrides)
      4. {repo_root}/.data/.env.local -- machine-local dev overrides (highest precedence,
                                        written by `make env`; never committed)
    """
    env_name = app_env or os.getenv("APP_ENV", "dev")
    package_root, repo_root = project_roots(anchor_file)
    load_kit_layered_env(
        repo_root,
        app_env=env_name,
        package_env_dir=package_root / package_env_dir,
        root_env_files=(".env", root_env_filename, ".data/.env.local"),
    )


def load_script_env(*, anchor_file: str, default_app_env: str = "prod") -> None:
    """Load layered env for CLI/script entrypoints.

    This keeps a single canonical behavior across scripts:
    - If APP_ENV is set, honor it.
    - Otherwise default to *default_app_env* (prod by default).
    """
    load_layered_env(
        anchor_file=anchor_file,
        app_env=os.getenv("APP_ENV", default_app_env),
    )
