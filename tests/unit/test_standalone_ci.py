"""Standalone CI: locked local gate plus slim fusion-rt Docker suite."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"
FUSION_COMPOSE = ROOT / "docker" / "fusion-rt" / "docker-compose.yml"
FUSION_DF = ROOT / "docker" / "fusion-rt" / "Dockerfile"
TESTS_DF = ROOT / "docker" / "fusion-rt" / "Dockerfile.tests"
INSTALL_SH = ROOT / "docker" / "install-python.sh"
ROOT_PYPROJECT = ROOT / "pyproject.toml"
FUSION_RT_PYPROJECT = ROOT / "apps" / "fusion-rt" / "pyproject.toml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ss_common_pin_is_published_tag() -> None:
    text = _read(ROOT_PYPROJECT)
    assert 'ss-common = { git = "https://github.com/volod/ss-common", tag = "v0.1.0" }' in text
    makefile = _read(MAKEFILE)
    assert "SS_COMMON_TAG ?= v0.1.0" in makefile
    assert "ss-common.git@$(SS_COMMON_TAG)" in makefile


def test_workspace_member_versions_are_021() -> None:
    assert 'version = "0.2.1"' in _read(ROOT_PYPROJECT)
    assert 'version = "0.2.1"' in _read(FUSION_RT_PYPROJECT)


def test_fusion_rt_does_not_depend_on_ss_fusion() -> None:
    text = _read(FUSION_RT_PYPROJECT)
    assert "ss-fusion" not in text
    assert "ss-perception" in text
    assert "ss-common[web,mqtt]" in text


def test_makefile_has_fusion_rt_docker_and_standalone_build() -> None:
    text = _read(MAKEFILE)
    assert "\ntest-fusion-rt:" in text
    assert "\nstandalone-build:" in text
    assert "docker/fusion-rt/docker-compose.yml" in text
    assert "docker/fusion-rt/docker-compose.test.yml" in text


def test_fusion_rt_images_are_slim() -> None:
    compose = _read(FUSION_COMPOSE)
    fusion_df = _read(FUSION_DF)
    tests_df = _read(TESTS_DF)
    install = _read(INSTALL_SH)
    assert "dockerfile: docker/fusion-rt/Dockerfile" in compose
    assert "install-python.sh --runtime" in fusion_df
    assert "torch" not in fusion_df.lower()
    assert "torch" not in tests_df.lower()
    assert ".[vision]" not in fusion_df
    assert ".[vision]" not in tests_df
    assert ".[dev]" not in tests_df
    assert "tests/test_fusion_rt.py" in tests_df
    assert "--runtime" in install
    assert "no torch" in install
    assert "packages/ss-fusion" not in install
    assert "v0.1.0" in install
    assert "v0.1.0" in tests_df
