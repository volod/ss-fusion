"""ssv_vdp — standalone local video-data pipeline (VDP).

Offline analysis and training orchestration:
  Phase 1  — input validation, frame extraction
  Phase 2  — vision inference (YOLO, SAM, depth, Florence, Qwen, RF-DETR)
  Phase 3  — SSL fine-tuning and distillation
  Phase 4  — state fusion, threat assessment, HTML/JSON report

Depends on selfsuvis for shared pipeline infrastructure.
"""

import os

# Suppress transformers lazy-loader __warningregistry__ noise.
# Must be set before the first transformers import; runner.py triggers that
# import, so we set it here — the earliest point in the package load order.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

__all__ = ["run_local"]


def __getattr__(name: str):
    if name == "run_local":
        from .pipeline.runner import run_local  # noqa: PLC0415
        return run_local
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
