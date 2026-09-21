"""Deterministic demo bindings for kernel unit tests and the shed fixture."""

from pathlib import Path
from typing import Any


def write_marker(context: dict[str, Any]) -> dict[str, Any]:
    """Write `{step_id}.txt` under the run directory."""
    run_dir = Path(context["run_dir"])
    step_id = str(context["step_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{step_id}.txt"
    path.write_text("ok\n", encoding="ascii")
    return {"path": str(path)}
