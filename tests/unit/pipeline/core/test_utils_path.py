"""Unit tests for path validation (resolve_allowed_path, resolve_allowed_paths_for_walk)."""

from selfsuvis.pipeline.core import config
from selfsuvis.pipeline.core.utils import resolve_allowed_path, resolve_allowed_paths_for_walk


def test_resolve_allowed_path_must_be_file_rejects_dir(tmp_path, monkeypatch):
    """When must_be_file=True, directory returns None."""
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(tmp_path)])
    d = tmp_path / "subdir"
    d.mkdir()
    result = resolve_allowed_path(str(d), must_be_file=True)
    assert result is None


def test_resolve_allowed_path_must_be_dir_rejects_file(tmp_path, monkeypatch):
    """When must_be_dir=True, file returns None."""
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(tmp_path)])
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = resolve_allowed_path(str(f), must_be_dir=True)
    assert result is None


def test_resolve_allowed_path_within_base_no_type_check(tmp_path, monkeypatch):
    """Path inside allowed base with no type check resolves."""
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(tmp_path)])
    d = tmp_path / "dir"
    d.mkdir()
    result = resolve_allowed_path(str(d), must_be_file=False, must_be_dir=False)
    assert result == str(d.resolve())


def test_resolve_allowed_paths_for_walk_delegates(tmp_path, monkeypatch):
    """resolve_allowed_paths_for_walk returns same as resolve_allowed_path with must_be_dir."""
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(tmp_path)])
    d = tmp_path / "subdir"
    d.mkdir()
    r1 = resolve_allowed_paths_for_walk(str(d))
    r2 = resolve_allowed_path(str(d), must_be_dir=True)
    assert r1 == r2
    assert r1 == str(d.resolve())


def test_resolve_allowed_path_inside_base(tmp_path, monkeypatch):
    """Path inside allowed base is accepted."""
    base = tmp_path / "allowed"
    base.mkdir()
    sub = base / "subdir"
    sub.mkdir()
    f = base / "file.txt"
    f.write_text("x")

    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(base)])

    resolved_file = resolve_allowed_path(str(f), must_be_file=True)
    resolved_dir = resolve_allowed_path(str(sub), must_be_dir=True)
    assert resolved_file == str(f.resolve())
    assert resolved_dir == str(sub.resolve())


def test_resolve_allowed_path_outside_base(tmp_path, monkeypatch):
    """Path outside allowed base returns None."""
    base = tmp_path / "allowed"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    f = outside / "file.txt"
    f.write_text("x")

    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(base)])

    result = resolve_allowed_path(str(f), must_be_file=True)
    assert result is None
    result_dir = resolve_allowed_path(str(outside), must_be_dir=True)
    assert result_dir is None


def test_resolve_allowed_path_traversal_blocked(tmp_path, monkeypatch):
    """Path traversal (..) outside base is rejected."""
    base = tmp_path / "allowed"
    base.mkdir()
    sub = base / "sub"
    sub.mkdir()
    # sub/../.. goes above base - resolved will be tmp_path
    traversal = str(sub / ".." / "..")

    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(base)])

    result = resolve_allowed_path(traversal, must_be_dir=True)
    # tmp_path is parent of base, so traversal resolves to tmp_path which is outside base
    assert result is None


def test_resolve_allowed_path_empty_allowed_denies_all(tmp_path, monkeypatch):
    """When ALLOWED_INDEX_PATHS is empty, all paths are denied (fail-closed)."""
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [])
    f = tmp_path / "file.txt"
    f.write_text("x")
    d = tmp_path / "subdir"
    d.mkdir()
    assert resolve_allowed_path(str(f), must_be_file=True) is None
    assert resolve_allowed_path(str(d), must_be_dir=True) is None
    assert resolve_allowed_path(str(f)) is None
