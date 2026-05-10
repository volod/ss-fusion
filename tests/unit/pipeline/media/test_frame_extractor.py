"""Unit tests for pipeline.media.frames.extract_frames_fixed.

Requires a working cv2 installation. Skipped automatically when cv2 is not
importable (e.g. NumPy 2.x / OpenCV binary incompatibility in the test venv).
"""

import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", reason="cv2 not available (NumPy 2.x incompatibility)")

from selfsuvis.pipeline.media.frames import extract_frames_fixed  # noqa: E402

ASSETS = Path(__file__).resolve().parents[3] / "assets"
# vid_testsrc.mp4: 640×360, 10 fps, 20 frames (2 s)
TEST_VIDEO = str(ASSETS / "vid_testsrc.mp4")


def test_extract_frames_fixed_count(tmp_path):
    frames = extract_frames_fixed(TEST_VIDEO, str(tmp_path), interval_sec=1.0, max_frames=2)
    assert len(frames) == 2


def test_extract_frames_fixed_files_exist(tmp_path):
    frames = extract_frames_fixed(TEST_VIDEO, str(tmp_path), interval_sec=1.0, max_frames=2)
    for rec in frames:
        assert os.path.exists(rec.path)
        assert rec.path.endswith(".png")


def test_extract_frames_fixed_metadata(tmp_path):
    frames = extract_frames_fixed(TEST_VIDEO, str(tmp_path), interval_sec=1.0, max_frames=2)
    assert frames[0].t_sec == pytest.approx(0.0)
    assert frames[1].t_sec == pytest.approx(1.0)
    assert frames[0].index == 0
    assert frames[1].index == 1
    assert frames[0].width == 640
    assert frames[0].height == 360


def test_extract_frames_fixed_max_frames_respected(tmp_path):
    frames = extract_frames_fixed(TEST_VIDEO, str(tmp_path), interval_sec=0.1, max_frames=3)
    assert len(frames) == 3


def test_extract_frames_fixed_bad_path_raises(tmp_path):
    with pytest.raises(RuntimeError, match="failed to open video"):
        extract_frames_fixed("/nonexistent/video.mp4", str(tmp_path))
