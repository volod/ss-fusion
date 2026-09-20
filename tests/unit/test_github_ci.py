"""GitHub CI for ss-fusion stays a minutes-long light job."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_github_workflow_is_light_and_single_python() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "make ci-github" in text
    assert "timeout-minutes: 10" in text
    assert "PYTHON_VERSION=3.11" in text
    assert "3.12" not in text
    assert "3.13" not in text
    assert "uv sync --locked" not in text


def test_makefile_ci_github_is_not_the_locked_install() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\nci-github: github-bootstrap" in text
    assert "test-github:" in text
    assert "tests/unit/fusion_rt" in text
    assert "\nci: bootstrap lint" in text


def test_github_bootstrap_skips_locked_sync() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    start = text.index("github-bootstrap:")
    end = text.index("\nvenv:", start)
    recipes = "\n".join(
        line
        for line in text[start:end].splitlines()
        if line.startswith("\t") and not line.lstrip().startswith("#")
    )
    assert "uv sync" not in recipes
    assert "--no-deps" in recipes
    assert "--clear" in recipes
