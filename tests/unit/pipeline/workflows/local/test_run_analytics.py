from types import SimpleNamespace

from selfsuvis.pipeline.workflows.local import runner
from selfsuvis.pipeline.workflows.local import steps_report
from selfsuvis.pipeline.workflows.local import steps_caption


def test_build_local_run_analytics_payload_includes_high_signal_fields():
    summary = SimpleNamespace(
        video_name="mission_a",
        n_frames=51,
        duration_sec=25.5,
        fps=2.0,
        domain="aerial_reconnaissance",
        top_category="road or highway from above",
        scene_complexity="low",
        n_scene_clusters=1,
        has_3d_map=True,
        has_edge_model=True,
        artifact_inventory=SimpleNamespace(total_files=17, total_bytes=4096),
        run_health=SimpleNamespace(
            florence_caption_coverage=1.0,
            qwen_caption_coverage=0.4,
            qwen_parse_error_count=0,
            asr_coverage=1.0,
            ocr_coverage=0.0,
            world_model_ok=True,
            tracking_ok=True,
            tracking_filter_fallback_used=False,
            florence_runtime_mode="qwen_fallback",
            restore_failures=0,
            vram_wait_time_sec=1.2,
            warnings=["map_degraded"],
        ),
        detection_stats=SimpleNamespace(
            model="yolo11l",
            total_objects=94,
            mean_per_frame=1.84,
            max_per_frame=6,
            by_class={"vehicle": 80, "human": 10, "sign": 4},
        ),
        temporal_stats=SimpleNamespace(
            method="rssm",
            mean_surprise=0.663,
            peak_frames=[3, 12, 18],
        ),
        training_stats=SimpleNamespace(
            ssl_best_loss=1.9236,
            distill_best_loss=1.5464,
            distill_best_r1=0.529,
            distill_compression=3.9,
            ssl_epochs=3,
        ),
        tracking_stats=SimpleNamespace(
            model="rfdetr_base",
            gemma_scene_type="urban_street",
            tracking_priority=["vehicle"],
            tracking_targets_effective=["vehicle"],
            unique_track_ids=371,
            total_detections=1048,
            sam_masks_total=36,
        ),
        embedding_stats=SimpleNamespace(
            n_embeddings=51,
            embedding_dim=384,
            mean_neighbour_similarity=0.995,
        ),
        map_stats=SimpleNamespace(
            method="sfm",
            points=10,
            poses=10,
            sfm_poses=10,
            frame_anchor_count=10,
            degraded=True,
            quality_note="",
        ),
    )

    payload = runner._build_local_run_analytics_payload(summary)

    assert payload["video_name"] == "mission_a"
    assert payload["artifact_count"] == 17
    assert payload["run_health"]["world_model_ok"] is True
    assert payload["detection_stats"]["top_classes"][0] == ("vehicle", 80)
    assert payload["temporal_stats"]["peak_frames"] == [3, 12, 18]
    assert payload["tracking_stats"]["unique_track_ids"] == 371
    assert payload["embedding_stats"]["embedding_dim"] == 384
    assert payload["map_stats"]["degraded"] is True


def test_steps_report_analytics_formatters_render_compact_cells():
    summary = {
        "run_health": {
            "florence_caption_coverage": 1.0,
            "qwen_caption_coverage": 0.4,
            "asr_coverage": 1.0,
            "ocr_coverage": 0.0,
            "world_model_ok": False,
            "warnings": ["map_degraded", "world_model_unavailable", "ocr_zero_coverage"],
        },
        "detection_stats": {
            "total_objects": 94,
            "mean_per_frame": 1.84,
        },
        "temporal_stats": {
            "mean_surprise": 0.663,
            "peak_frames": [3, 12, 18],
        },
        "tracking_stats": {
            "unique_track_ids": 371,
        },
        "map_stats": {
            "degraded": True,
            "points": 10,
            "poses": 10,
        },
    }

    assert steps_report._fmt_analytics_coverage(summary) == "100/40/100/0%"
    assert steps_report._fmt_analytics_detections(summary) == "94 (1.8/fr)"
    assert steps_report._fmt_analytics_temporal(summary) == "0.663 / 3 peaks"
    assert steps_report._fmt_analytics_world_tracking(summary) == "degraded / 371 tracks"
    assert steps_report._fmt_analytics_map(summary) == "degraded (10p/10 poses)"
    assert steps_report._fmt_analytics_warnings(summary) == "map_degraded, world_model_unavailable +1"


def test_fallback_ocr_frame_sample_selects_evenly_spaced_subset():
    frame_list = [(f"frame_{i:02d}.jpg", float(i)) for i in range(20)]

    selected = steps_caption._fallback_ocr_frame_sample(frame_list, max_samples=5)

    assert len(selected) == 5
    assert selected[0][0] == "frame_00.jpg"
    assert selected[-1][0] == "frame_19.jpg"
