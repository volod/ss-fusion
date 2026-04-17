from __future__ import annotations

from pathlib import Path

from selfsuvis.pipeline.vision import rfdetr


def test_rfdetr_weights_path_uses_data_dir_and_migrates_legacy(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    legacy = repo_root / "rf-detr-base.pth"
    legacy.write_bytes(b"stub")

    monkeypatch.setattr(rfdetr.settings, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(rfdetr.Path, "cwd", lambda: repo_root)

    resolved = Path(rfdetr._rfdetr_weights_path("base"))

    assert resolved == data_dir.resolve() / "models" / "rfdetr" / "rf-detr-base.pth"
    assert resolved.exists()
    assert not legacy.exists()
