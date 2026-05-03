"""Unit tests for pipeline.media.gps."""
import json
from unittest import mock

import pytest

from selfsuvis.pipeline.media.gps import (
    _extract_from_ffprobe_atoms,
    _interpolate_gps,
    _parse_srt_file,
    extract_gps,
)

# ── fixtures / helpers ────────────────────────────────────────────────────────

# Minimal DJI SRT content with two blocks
SRT_TWO_BLOCKS = """\
1
00:00:01,000 --> 00:00:02,000
SrtCnt : 1
latitude : 47.5577 longitude : 8.4697 altitude : 482m

2
00:00:02,000 --> 00:00:03,000
SrtCnt : 2
latitude : 47.5578 longitude : 8.4698 altitude : 483m
"""

SRT_NO_GPS = """\
1
00:00:01,000 --> 00:00:02,000
No GPS data here.
"""


# ── _parse_srt_file ───────────────────────────────────────────────────────────

def test_parse_srt_valid(tmp_path):
    srt = tmp_path / "video.srt"
    srt.write_text(SRT_TWO_BLOCKS)
    records = _parse_srt_file(str(srt))
    assert len(records) == 2


def test_parse_srt_first_record_lat_lon(tmp_path):
    srt = tmp_path / "video.srt"
    srt.write_text(SRT_TWO_BLOCKS)
    records = _parse_srt_file(str(srt))
    assert records[0]["lat"] == pytest.approx(47.5577)
    assert records[0]["lon"] == pytest.approx(8.4697)
    assert records[0]["alt"] == pytest.approx(482.0)


def test_parse_srt_first_record_timestamp(tmp_path):
    srt = tmp_path / "video.srt"
    srt.write_text(SRT_TWO_BLOCKS)
    records = _parse_srt_file(str(srt))
    assert records[0]["timestamp_ms"] == pytest.approx(1000.0)


def test_parse_srt_second_record(tmp_path):
    srt = tmp_path / "video.srt"
    srt.write_text(SRT_TWO_BLOCKS)
    records = _parse_srt_file(str(srt))
    assert records[1]["lat"] == pytest.approx(47.5578)
    assert records[1]["timestamp_ms"] == pytest.approx(2000.0)


def test_parse_srt_missing_file(tmp_path):
    records = _parse_srt_file(str(tmp_path / "nonexistent.srt"))
    assert records == []


def test_parse_srt_no_gps(tmp_path):
    srt = tmp_path / "nogps.srt"
    srt.write_text(SRT_NO_GPS)
    assert _parse_srt_file(str(srt)) == []


def test_parse_srt_sorted_by_timestamp(tmp_path):
    """Records are returned sorted by timestamp_ms."""
    # Write blocks in reverse timestamp order
    srt_content = """\
2
00:00:02,000 --> 00:00:03,000
latitude : 48.0 longitude : 9.0 altitude : 500m

1
00:00:01,000 --> 00:00:02,000
latitude : 47.0 longitude : 8.0 altitude : 400m
"""
    srt = tmp_path / "reverse.srt"
    srt.write_text(srt_content)
    records = _parse_srt_file(str(srt))
    assert records[0]["timestamp_ms"] < records[1]["timestamp_ms"]


# ── _extract_from_ffprobe_atoms ───────────────────────────────────────────────

def _ffprobe_json(location: str) -> str:
    return json.dumps({"format": {"tags": {"com.apple.quicktime.location.ISO6709": location}}})


def test_ffprobe_atoms_valid_iso6709():
    with mock.patch(
        "selfsuvis.pipeline.media.gps._run_ffprobe",
        return_value=_ffprobe_json("+47.5577+008.4697+482.123/"),
    ):
        fix = _extract_from_ffprobe_atoms("/fake/video.mp4")
    assert fix is not None
    assert fix["lat"] == pytest.approx(47.5577)
    assert fix["lon"] == pytest.approx(8.4697)
    assert fix["alt"] == pytest.approx(482.123)


def test_ffprobe_atoms_negative_coords():
    with mock.patch(
        "selfsuvis.pipeline.media.gps._run_ffprobe",
        return_value=_ffprobe_json("-33.8688+151.2093+10.0/"),
    ):
        fix = _extract_from_ffprobe_atoms("/fake/video.mp4")
    assert fix is not None
    assert fix["lat"] == pytest.approx(-33.8688)
    assert fix["lon"] == pytest.approx(151.2093)


def test_ffprobe_atoms_no_location_tag():
    out = json.dumps({"format": {"tags": {}}})
    with mock.patch("selfsuvis.pipeline.media.gps._run_ffprobe", return_value=out):
        assert _extract_from_ffprobe_atoms("/fake/video.mp4") is None


def test_ffprobe_atoms_ffprobe_fails():
    with mock.patch("selfsuvis.pipeline.media.gps._run_ffprobe", return_value=None):
        assert _extract_from_ffprobe_atoms("/fake/video.mp4") is None


def test_ffprobe_atoms_invalid_json():
    with mock.patch("selfsuvis.pipeline.media.gps._run_ffprobe", return_value="not json"):
        assert _extract_from_ffprobe_atoms("/fake/video.mp4") is None


# ── _interpolate_gps ─────────────────────────────────────────────────────────

def test_interpolate_empty_records():
    assert _interpolate_gps([], [1000.0, 2000.0]) == [None, None]


def test_interpolate_exact_match():
    records = [
        {"lat": 47.0, "lon": 8.0, "alt": 100.0, "timestamp_ms": 1000.0},
        {"lat": 48.0, "lon": 9.0, "alt": 200.0, "timestamp_ms": 2000.0},
    ]
    result = _interpolate_gps(records, [1000.0, 2000.0])
    assert result[0]["lat"] == pytest.approx(47.0)
    assert result[1]["lat"] == pytest.approx(48.0)


def test_interpolate_midpoint():
    records = [
        {"lat": 47.0, "lon": 8.0, "alt": 100.0, "timestamp_ms": 0.0},
        {"lat": 49.0, "lon": 10.0, "alt": 300.0, "timestamp_ms": 2000.0},
    ]
    result = _interpolate_gps(records, [1000.0])
    assert result[0]["lat"] == pytest.approx(48.0)
    assert result[0]["lon"] == pytest.approx(9.0)
    assert result[0]["alt"] == pytest.approx(200.0)


def test_interpolate_before_first_clamps():
    records = [{"lat": 10.0, "lon": 20.0, "alt": 0.0, "timestamp_ms": 5000.0}]
    result = _interpolate_gps(records, [0.0])
    assert result[0]["lat"] == pytest.approx(10.0)


def test_interpolate_after_last_clamps():
    records = [{"lat": 10.0, "lon": 20.0, "alt": 0.0, "timestamp_ms": 1000.0}]
    result = _interpolate_gps(records, [9999.0])
    assert result[0]["lat"] == pytest.approx(10.0)


def test_interpolate_preserves_timestamp_ms():
    records = [
        {"lat": 47.0, "lon": 8.0, "alt": 0.0, "timestamp_ms": 0.0},
        {"lat": 48.0, "lon": 9.0, "alt": 0.0, "timestamp_ms": 2000.0},
    ]
    result = _interpolate_gps(records, [500.0])
    assert result[0]["timestamp_ms"] == pytest.approx(500.0)


def test_interpolate_empty_timestamps():
    records = [{"lat": 47.0, "lon": 8.0, "alt": 0.0, "timestamp_ms": 1000.0}]
    assert _interpolate_gps(records, []) == []


# ── extract_gps integration ───────────────────────────────────────────────────

def test_extract_gps_uses_srt_when_present(tmp_path, monkeypatch):
    srt = tmp_path / "video.srt"
    srt.write_text(SRT_TWO_BLOCKS)
    video = tmp_path / "video.mp4"
    video.write_text("")

    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "GPS_SIDECAR_PATH", "")

    result = extract_gps(str(video), [1000.0, 2000.0])
    assert len(result) == 2
    assert result[0] is not None
    assert result[0]["lat"] == pytest.approx(47.5577)


def test_extract_gps_falls_back_to_ffprobe_atoms(tmp_path, monkeypatch):
    """No SRT sidecar → falls back to ffprobe ISO 6709 atom."""
    video = tmp_path / "video.mp4"
    video.write_text("")

    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "GPS_SIDECAR_PATH", "")

    ffprobe_out = json.dumps(
        {"format": {"tags": {"com.apple.quicktime.location.ISO6709": "+47.5000+008.0000+100.0/"}}}
    )
    with mock.patch("selfsuvis.pipeline.media.gps._run_ffprobe", return_value=ffprobe_out):
        result = extract_gps(str(video), [500.0, 1000.0])

    assert len(result) == 2
    assert all(r is not None for r in result)
    assert result[0]["lat"] == pytest.approx(47.5)


def test_extract_gps_null_fallback(tmp_path, monkeypatch):
    """No GPS anywhere → all frames return None."""
    video = tmp_path / "video.mp4"
    video.write_text("")

    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "GPS_SIDECAR_PATH", "")

    with mock.patch("selfsuvis.pipeline.media.gps._run_ffprobe", return_value=None):
        result = extract_gps(str(video), [1000.0, 2000.0, 3000.0])

    assert result == [None, None, None]


def test_extract_gps_sidecar_path_override(tmp_path, monkeypatch):
    """GPS_SIDECAR_PATH setting overrides auto-detection path."""
    custom_srt = tmp_path / "custom_gps.srt"
    custom_srt.write_text(SRT_TWO_BLOCKS)
    video = tmp_path / "othervideo.mp4"
    video.write_text("")

    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "GPS_SIDECAR_PATH", str(custom_srt))

    result = extract_gps(str(video), [1000.0])
    assert result[0] is not None
    assert result[0]["lat"] == pytest.approx(47.5577)


def test_extract_gps_returns_same_length(tmp_path, monkeypatch):
    """Return list always matches len(frame_timestamps_ms)."""
    video = tmp_path / "video.mp4"
    video.write_text("")

    from selfsuvis.pipeline.core import config
    monkeypatch.setattr(config.settings, "GPS_SIDECAR_PATH", "")

    with mock.patch("selfsuvis.pipeline.media.gps._run_ffprobe", return_value=None):
        timestamps = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = extract_gps(str(video), timestamps)
    assert len(result) == len(timestamps)
