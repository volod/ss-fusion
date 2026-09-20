"""Which unit tests GitHub CI collects.

Heavy tests import torch, OpenCV, ffmpeg, ONNX Runtime, or other vision/ML
stacks. `make test-unit` (full local venv) runs them. `make ci` passes
`--ci-light`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAVY_UNIT_PATHS = (
    "tests/unit/models/",
    "tests/unit/pipeline/training/",
    "tests/unit/pipeline/analysis/test_distribution_shift.py",
    "tests/unit/pipeline/media/test_frame_extractor.py",
    "tests/unit/pipeline/media/test_heuristics.py",
    "tests/unit/pipeline/vision/test_florence_model.py",
    "tests/unit/pipeline/vision/test_ocr_model.py",
    "tests/unit/pipeline/vision/test_unidrive.py",
    "tests/unit/pipeline/vision/test_world_model.py",
    "tests/unit/pipeline/workflows/local/test_agentic_audit.py",
    "tests/unit/pipeline/workflows/local/test_agentic_processor.py",
    "tests/unit/pipeline/workflows/local/test_local_sidecar_cleanup.py",
    "tests/unit/pipeline/workflows/local/test_run_analytics.py",
    "tests/unit/pipeline/workflows/local/test_steps_embed.py",
    "tests/unit/pipeline/workflows/local/test_steps_ssl.py",
)

BENCHMARK_UNIT_PATHS: tuple[str, ...] = ()

CI_LIGHT_MARKEXPR = (
    "not slow and not heavy and not benchmark and not gpu and not integration and not load"
)


def repo_relative(path: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def path_matches(relative: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            if relative == pattern[:-1] or relative.startswith(pattern):
                return True
        elif relative == pattern:
            return True
    return False


def is_heavy_unit_path(path: Path) -> bool:
    relative = repo_relative(path)
    return bool(relative) and path_matches(relative, HEAVY_UNIT_PATHS)


def is_benchmark_unit_path(path: Path) -> bool:
    relative = repo_relative(path)
    return bool(relative) and path_matches(relative, BENCHMARK_UNIT_PATHS)
