"""Workspace layout smoke test for ss-fusion staging."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_workspace_members_declared() -> None:
    assert (ROOT / "packages" / "ss-perception" / "pyproject.toml").is_file()
    assert (ROOT / "packages" / "ss-mapping" / "pyproject.toml").is_file()
    assert (ROOT / "packages" / "ss-fusion" / "pyproject.toml").is_file()
    assert (ROOT / "apps" / "fusion-rt" / "pyproject.toml").is_file()
    assert (ROOT / "AGENTS.md").is_file()
    assert (ROOT / "docs" / "design" / "spec.md").is_file()
