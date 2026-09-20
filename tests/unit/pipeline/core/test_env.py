import os
from pathlib import Path

from selfsuvis.pipeline.core.env import (
    env_csv,
    env_json_dict,
    kit_project_root,
    project_roots,
    set_env_if_present,
)


def test_env_csv_parses_trimmed_values(monkeypatch):
    monkeypatch.setenv("TEST_CSV", " alpha, beta ,gamma ")
    assert env_csv("TEST_CSV") == ["alpha", "beta", "gamma"]


def test_env_json_dict_uses_fallback_and_error_callback(monkeypatch):
    seen: list[str] = []

    def _on_error(message: str, key: str) -> None:
        seen.append(message % key)

    monkeypatch.setenv("TEST_JSON", "{bad")
    parsed = env_json_dict("TEST_JSON", default={"fallback": "yes"}, on_error=_on_error)

    assert parsed == {"fallback": "yes"}
    assert seen == ["TEST_JSON contains invalid JSON; using default value"]


def test_set_env_if_present_does_not_write_empty(monkeypatch):
    monkeypatch.delenv("MAYBE_SET", raising=False)
    set_env_if_present("MAYBE_SET", "")
    assert "MAYBE_SET" not in os.environ

    set_env_if_present("MAYBE_SET", Path("/tmp/value"))
    assert os.environ["MAYBE_SET"] == "/tmp/value"


def test_kit_project_root_finds_markers(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "AGENTS.md").write_text("# agents\n")
    nested = tmp_path / "pkg" / "nested"
    nested.mkdir(parents=True)
    assert kit_project_root(nested) == tmp_path


def test_kit_project_root_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    assert kit_project_root(orphan) == tmp_path.resolve()


def test_project_roots_walks_staged_package_layout(tmp_path):
    repo = tmp_path / "ss-fusion"
    pkg = repo / "packages" / "ss-perception" / "src" / "selfsuvis" / "pipeline" / "core"
    pkg.mkdir(parents=True)
    (pkg / "env.py").write_text("# stub\n")
    (repo / "pyproject.toml").write_text("[project]\nname = 'ss-fusion'\n")
    (repo / "AGENTS.md").write_text("# agents\n")
    (pkg.parent.parent / "env").mkdir()
    package_root, repo_root = project_roots(str(pkg / "env.py"))
    assert package_root == pkg.parent.parent
    assert repo_root == repo
