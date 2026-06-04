import math


def test_step_labels_include_audio_drau_and_advisor_order():
    from ssv_vdp.steps.report_helpers._stats import _STEP_LABELS

    labels = [label for _key, label, _kind in _STEP_LABELS]

    assert "33 Train: Drone audio" in labels
    assert "34 Eval: drau range" in labels
    assert "35 Optimize: Model/run advisor" in labels
    assert labels.index("33 Train: Drone audio") < labels.index("34 Eval: drau range")
    assert labels.index("34 Eval: drau range") < labels.index("35 Optimize: Model/run advisor")


# ---------------------------------------------------------------------------
# _fmt_sec boundary conditions
# ---------------------------------------------------------------------------


def test_fmt_sec_nan_returns_dash():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(float("nan")) == "—"


def test_fmt_sec_negative_returns_dash():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(-1.0) == "—"
    assert _fmt_sec(-0.001) == "—"


def test_fmt_sec_zero():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(0.0) == "0.0s"


def test_fmt_sec_below_sixty_seconds():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(1.2) == "1.2s"
    assert _fmt_sec(59.9) == "59.9s"


def test_fmt_sec_exactly_sixty_seconds():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(60.0) == "1m 00.0s"


def test_fmt_sec_minutes_range():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(90.0) == "1m 30.0s"
    assert _fmt_sec(65.5) == "1m 05.5s"


def test_fmt_sec_exactly_one_hour():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(3600.0) == "1h 00m 00s"


def test_fmt_sec_hours_range():
    from ssv_vdp.steps.report_helpers._stats import _fmt_sec

    assert _fmt_sec(3661.0) == "1h 01m 01s"
    assert _fmt_sec(7384.0) == "2h 03m 04s"


# ---------------------------------------------------------------------------
# Analytics formatter functions
# ---------------------------------------------------------------------------


def test_fmt_analytics_coverage_formats_percentages():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_coverage

    summary = {
        "run_health": {
            "florence_caption_coverage": 1.0,
            "qwen_caption_coverage": 0.5,
            "asr_coverage": 0.0,
            "ocr_coverage": 0.75,
        }
    }
    result = _fmt_analytics_coverage(summary)
    assert result == "100/50/0/75%"


def test_fmt_analytics_coverage_appends_parse_errors():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_coverage

    summary = {
        "run_health": {
            "florence_caption_coverage": 1.0,
            "qwen_caption_coverage": 1.0,
            "asr_coverage": 1.0,
            "ocr_coverage": 1.0,
            "qwen_parse_error_count": 3,
        }
    }
    result = _fmt_analytics_coverage(summary)
    assert result == "100/100/100/100% (Qwen parse=3)"


def test_fmt_analytics_detections_no_total_returns_dash():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_detections

    assert _fmt_analytics_detections({}) == "—"
    assert _fmt_analytics_detections({"detection_stats": {}}) == "—"
    assert _fmt_analytics_detections({"detection_stats": {"total_objects": None}}) == "—"


def test_fmt_analytics_detections_no_mean_returns_total_only():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_detections

    summary = {"detection_stats": {"total_objects": 15}}
    assert _fmt_analytics_detections(summary) == "15"


def test_fmt_analytics_detections_full_format():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_detections

    summary = {"detection_stats": {"total_objects": 30, "mean_per_frame": 2.5}}
    assert _fmt_analytics_detections(summary) == "30 (2.5/fr)"


def test_fmt_analytics_temporal_no_data_returns_dash():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_temporal

    assert _fmt_analytics_temporal({}) == "—"
    assert _fmt_analytics_temporal({"temporal_stats": {"mean_surprise": None}}) == "—"


def test_fmt_analytics_temporal_with_data():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_temporal

    summary = {"temporal_stats": {"mean_surprise": 0.154, "peak_frames": [1, 5, 9]}}
    assert _fmt_analytics_temporal(summary) == "0.154 / 3 peaks"


def test_fmt_analytics_map_empty_returns_dash():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_map

    assert _fmt_analytics_map({}) == "—"
    assert _fmt_analytics_map({"map_stats": {}}) == "—"


def test_fmt_analytics_map_sfm_anchors_equal_shows_poses():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_map

    summary = {"map_stats": {"points": 500, "poses": 40}}
    result = _fmt_analytics_map(summary)
    assert result == "ok (500p/40 poses)"


def test_fmt_analytics_map_sfm_anchors_differ_shows_both():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_map

    summary = {
        "map_stats": {
            "points": 500,
            "poses": 50,
            "sfm_poses": 40,
            "frame_anchor_count": 50,
        }
    }
    result = _fmt_analytics_map(summary)
    assert result == "ok (500p/40 SfM, 50 anchors)"


def test_fmt_analytics_warnings_none_returns_dash():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_warnings

    assert _fmt_analytics_warnings({}) == "—"
    assert _fmt_analytics_warnings({"run_health": {"warnings": []}}) == "—"


def test_fmt_analytics_warnings_truncates_beyond_two():
    from ssv_vdp.steps.report_helpers._stats import _fmt_analytics_warnings

    summary = {"run_health": {"warnings": ["w1", "w2", "w3", "w4"]}}
    result = _fmt_analytics_warnings(summary)
    assert result == "w1, w2 +2"


# ---------------------------------------------------------------------------
# write_final_stats_md — concurrent overlap calculation
# ---------------------------------------------------------------------------


def test_write_final_stats_md_concurrent_overlap_positive(tmp_path):
    from ssv_vdp.steps.report_helpers._stats import write_final_stats_md

    per_video = [
        {"name": "v1", "timings": {"A_extract": 60.0, "I_3dmap": 80.0}},
    ]
    # step_sum = 140, total_elapsed = 100 → overlap = 40
    out = tmp_path / "stats.md"
    write_final_stats_md(out, per_video, total_elapsed=100.0)

    content = out.read_text()
    assert "Concurrent overlap: 40.0s" in content


def test_write_final_stats_md_concurrent_overlap_zero_when_no_concurrency(tmp_path):
    from ssv_vdp.steps.report_helpers._stats import write_final_stats_md

    per_video = [
        {"name": "v1", "timings": {"A_extract": 20.0, "B_index": 15.0}},
    ]
    # step_sum = 35, total_elapsed = 100 → overlap = max(0, 35-100) = 0
    out = tmp_path / "stats.md"
    write_final_stats_md(out, per_video, total_elapsed=100.0)

    content = out.read_text()
    assert "Concurrent overlap: 0.0s" in content
