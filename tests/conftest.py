"""Shared pytest configuration and custom marks."""

import pytest
from tests.support.ci_light import (
    CI_LIGHT_MARKEXPR,
    is_benchmark_unit_path,
    is_heavy_unit_path,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ci-light",
        action="store_true",
        default=False,
        help="GitHub CI profile: collect only tests that do not need torch/cv2/ffmpeg",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA GPU (skip with -m 'not gpu')",
    )
    if config.getoption("ci_light"):
        current = (getattr(config.option, "markexpr", None) or "").strip()
        config.option.markexpr = (
            f"({current}) and ({CI_LIGHT_MARKEXPR})" if current else CI_LIGHT_MARKEXPR
        )


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool:
    if not config.getoption("ci_light"):
        return False
    return is_heavy_unit_path(collection_path)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            continue
        if is_heavy_unit_path(path):
            item.add_marker(pytest.mark.heavy)
        if is_benchmark_unit_path(path):
            item.add_marker(pytest.mark.benchmark)
