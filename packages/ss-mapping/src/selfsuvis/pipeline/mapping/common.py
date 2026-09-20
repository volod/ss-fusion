"""Shared filesystem helpers for mapping outputs and reports."""

import json
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import ensure_dir

PathLike = str | Path


def ensure_output_parent(path: PathLike) -> Path:
    output_path = Path(path)
    ensure_dir(str(output_path.parent))
    return output_path


def write_json_report(path: PathLike, payload: Any, *, ensure_ascii: bool = True) -> str:
    output_path = ensure_output_parent(path)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    return str(output_path)


def write_markdown_report(path: PathLike, lines: list[str]) -> str:
    output_path = ensure_output_parent(path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)
