"""Unit tests for pipeline.ffmpeg_utils."""

from unittest.mock import MagicMock, patch

import pytest

from selfsuvis.pipeline.ffmpeg_utils import extract_frames


def test_extract_frames_uses_timeout(tmp_path, monkeypatch):
    """extract_frames passes FFMPEG_TIMEOUT_SEC to subprocess.run."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    monkeypatch.setattr("selfsuvis.pipeline.media.ffmpeg.settings", MagicMock(
        FRAMES_DIR=str(frames_dir),
        SAMPLE_FPS_MAX=5,
        FFMPEG_TIMEOUT_SEC=123,
    ))

    mock_run = MagicMock()
    with patch("selfsuvis.pipeline.media.ffmpeg.subprocess.run", mock_run):
        with patch("selfsuvis.pipeline.media.ffmpeg.os.listdir", return_value=[]):
            result = extract_frames(str(tmp_path / "video.mp4"), "vid1")

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs.get("timeout") == 123
    assert result == []
