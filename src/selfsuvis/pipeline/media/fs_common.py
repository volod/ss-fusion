"""Shared filesystem helpers for media pipelines."""

import os

from selfsuvis.pipeline.core import ensure_dir


def ensure_parent_dir(path: str) -> None:
    ensure_dir(os.path.dirname(path))


def remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def output_path_with_suffix(source_path: str, output_dir: str, suffix: str) -> str:
    base = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(output_dir, f"{base}{suffix}")
