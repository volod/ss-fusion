"""Unit tests for pipeline.utils."""

import hashlib
import tempfile
from pathlib import Path

from selfsuvis.pipeline.core import config
from selfsuvis.pipeline.core.utils import (
    RateTimer,
    clamp,
    file_sha256,
    file_sha256_bytes,
    resolve_allowed_path,
    stable_point_id,
)


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
    assert clamp(0.5, 0.0, 1.0) == 0.5
    assert clamp(-0.1, 0.0, 1.0) == 0.0
    assert clamp(1.5, 0.0, 1.0) == 1.0


def test_stable_point_id():
    id1 = stable_point_id("vid1", 0, 1000, "frame")
    id2 = stable_point_id("vid1", 0, 1000, "frame")
    assert id1 == id2
    id3 = stable_point_id("vid1", 0, 1001, "frame")
    assert id1 != id3
    id4 = stable_point_id("vid1", 0, 1000, "tile", 10, 20)
    assert id1 != id4


def test_file_sha256_bytes():
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    assert file_sha256_bytes(data) == expected
    assert file_sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()


def test_file_sha256():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content")
        path = f.name
    try:
        expected = hashlib.sha256(b"test content").hexdigest()
        assert file_sha256(path) == expected
    finally:
        Path(path).unlink(missing_ok=True)


def test_resolve_allowed_path_empty_allowed_denies_all(tmp_path, monkeypatch):
    """When ALLOWED_INDEX_PATHS is empty, all paths are denied (fail-closed)."""
    f = tmp_path / "file.txt"
    f.write_text("x")
    d = tmp_path / "subdir"
    d.mkdir()
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [])
    assert resolve_allowed_path(str(f), must_be_file=True) is None
    assert resolve_allowed_path(str(d), must_be_dir=True) is None


def test_resolve_allowed_path_within_explicit_base(tmp_path, monkeypatch):
    """When ALLOWED_INDEX_PATHS is set, paths inside the base resolve correctly."""
    base = tmp_path / "allowed"
    base.mkdir()
    f = base / "file.txt"
    f.write_text("x")
    d = base / "subdir"
    d.mkdir()
    monkeypatch.setattr(config.settings, "ALLOWED_INDEX_PATHS", [str(base)])
    assert resolve_allowed_path(str(f), must_be_file=True) == str(f.resolve())
    assert resolve_allowed_path(str(d), must_be_dir=True) == str(d.resolve())


def test_rate_timer():
    timer = RateTimer()
    assert timer.rate() >= 0
    timer.tick()
    timer.tick(5)
    assert timer.count == 6
    assert timer.rate() > 0
