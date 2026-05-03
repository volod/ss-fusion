"""Unit tests for pipeline/audio_extractor.py — no ffmpeg/GPU required."""
from selfsuvis.pipeline.media.audio import _normalise_segments, map_subtitles_to_frames

# ── _normalise_segments ────────────────────────────────────────────────────────

def test_normalise_hf_format():
    """HuggingFace pipeline format: {"timestamp": (start, end), "text": ...}"""
    segs = [{"timestamp": (0.0, 2.5), "text": "hello world"}]
    result = _normalise_segments(segs)
    assert result == [{"text": "hello world", "start": 0.0, "end": 2.5}]


def test_normalise_direct_format():
    """Direct format: {"start": float, "end": float, "text": ...}"""
    segs = [{"start": 5.0, "end": 8.0, "text": "convoy spotted"}]
    result = _normalise_segments(segs)
    assert result == [{"text": "convoy spotted", "start": 5.0, "end": 8.0}]


def test_normalise_skips_empty_text():
    segs = [{"timestamp": (0.0, 1.0), "text": "  "}, {"timestamp": (2.0, 3.0), "text": "ok"}]
    result = _normalise_segments(segs)
    assert len(result) == 1
    assert result[0]["text"] == "ok"


def test_normalise_unknown_format_skipped():
    segs = [{"unknown_key": "value", "text": "test"}]
    result = _normalise_segments(segs)
    assert result == []


def test_normalise_null_end_gets_default():
    """When end is None, default to start + 3."""
    segs = [{"timestamp": (5.0, None), "text": "test"}]
    result = _normalise_segments(segs)
    assert result[0]["start"] == 5.0
    assert result[0]["end"] == 8.0  # start + 3


def test_normalise_empty_input():
    assert _normalise_segments([]) == []


# ── map_subtitles_to_frames ────────────────────────────────────────────────────

def test_exact_overlap():
    """Frame at 5 s with segment [4, 6] — within window."""
    segs = [{"timestamp": (4.0, 6.0), "text": "trucks moving"}]
    result = map_subtitles_to_frames(segs, [5.0], window_sec=3.0)
    assert result[5.0] == "trucks moving"


def test_no_overlap_outside_window():
    """Frame at 0 s, segment at [10, 12] — too far."""
    segs = [{"timestamp": (10.0, 12.0), "text": "distant"}]
    result = map_subtitles_to_frames(segs, [0.0], window_sec=3.0)
    assert 0.0 not in result


def test_multiple_segments_concatenated():
    """Two segments near the same frame should be joined by a space."""
    segs = [
        {"timestamp": (3.0, 4.0), "text": "first part"},
        {"timestamp": (4.0, 5.0), "text": "second part"},
    ]
    result = map_subtitles_to_frames(segs, [4.0], window_sec=2.0)
    assert result[4.0] == "first part second part"


def test_multiple_frames_separate_subtitles():
    segs = [
        {"timestamp": (0.0, 1.0), "text": "intro"},
        {"timestamp": (30.0, 31.0), "text": "later"},
    ]
    result = map_subtitles_to_frames(segs, [0.5, 30.5], window_sec=2.0)
    assert result[0.5] == "intro"
    assert result[30.5] == "later"


def test_empty_segments_returns_empty():
    result = map_subtitles_to_frames([], [1.0, 2.0, 3.0])
    assert result == {}


def test_empty_frames_returns_empty():
    segs = [{"timestamp": (0.0, 5.0), "text": "something"}]
    result = map_subtitles_to_frames(segs, [])
    assert result == {}


def test_boundary_frame_exactly_at_window_edge():
    """Frame at 0 s, segment ends at exactly window_sec (3.0)."""
    segs = [{"timestamp": (3.0, 3.0), "text": "edge"}]
    result = map_subtitles_to_frames(segs, [0.0], window_sec=3.0)
    # seg starts at 3.0, window hi = 3.0 → seg_start (3.0) <= hi (3.0): overlaps
    assert 0.0 in result


def test_direct_format_segments():
    """Accepts the faster-whisper {start, end, text} format directly."""
    segs = [{"start": 10.0, "end": 12.0, "text": "armored vehicle"}]
    result = map_subtitles_to_frames(segs, [11.0], window_sec=3.0)
    assert result[11.0] == "armored vehicle"
