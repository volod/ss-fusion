"""Tests for local-run analytics loader."""

from __future__ import annotations

import json

import numpy as np

from selfsuvis.analytics import loader


def _write(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_local_run_loader_parses_key_artifacts(tmp_path):
    from selfsuvis.analytics import LocalRunLoader

    run_dir = tmp_path / "mission_a"
    _write(
        run_dir / "frames_metadata.json",
        json.dumps(
            {
                "video_id": "mission_a",
                "fps": 2.0,
                "frame_count": 2,
                "duration_sec": 1.0,
                "frames": [
                    {"path": "frame_1.jpg", "t_sec": 0.0},
                    {"path": "frame_2.jpg", "t_sec": 0.5},
                ],
            }
        ),
    )
    _write(
        run_dir / "yolo_sam_results.json",
        json.dumps(
            {
                "model": "yolo11l",
                "n_frames": 2,
                "total_objects": 3,
                "by_priority": {"vehicle": 3},
                "frames": [
                    {"t_sec": 0.0, "n_detections": 1},
                    {"t_sec": 0.5, "n_detections": 2},
                ],
            }
        ),
    )
    _write(
        run_dir / "rssm_temporal.json",
        json.dumps(
            {
                "method": "rssm",
                "n_frames": 2,
                "surprise_scores": [0.1, 0.9],
            }
        ),
    )
    _write(
        run_dir / "video_ontology.json",
        json.dumps({"domain": "aerial_reconnaissance", "scene_complexity": "low"}),
    )
    _write(
        run_dir / "scene_captions.md",
        "\n".join(
            [
                "| Frame | t (s) | Seg | Sim | Confidence | Caption |",
                "| `frame_1.jpg` | 0.0 | 1 | — | 0.800 | road |",
                "| `frame_2.jpg` | 0.5 | 1 | 1.00 | 0.700 | highway |",
            ]
        ),
    )
    _write(
        run_dir / "detailed_captions.md",
        "\n".join(
            [
                "| Frame | t (s) | Seg | Δ Changes | Caption / Scene Facts | Audio Context |",
                "| `frame_1.jpg` | 0.0 | 1 | first | detailed road caption | hello |",
                "| `frame_2.jpg` | 0.5 | 1 | — | detailed highway caption | hello |",
            ]
        ),
    )
    _write(
        run_dir / "asr_subtitles.md",
        "\n".join(
            [
                "| Start (s) | End (s) | Text |",
                "| 0.00 | 1.00 | hello |",
            ]
        ),
    )
    _write(
        run_dir / "gemma_tracking_results.json",
        json.dumps(
            {
                "model": "rfdetr_base",
                "gemma_scene_type": "urban_street",
                "tracking_priority": ["vehicle"],
                "tracking_targets_effective": ["vehicle"],
                "tracking_filter_retry_mode": "reduced",
                "sam_masks_total": 4,
                "n_unique_track_ids": 2,
                "total_detections": 2,
                "mean_track_length_frames": 2.0,
                "median_track_length_frames": 2.0,
                "elapsed_sec": 12.0,
                "frames": [
                    {"t_sec": 0.0, "n_detections": 1, "sam_masks": [{}, {}]},
                    {"t_sec": 0.5, "n_detections": 1, "sam_masks": [{}, {}]},
                ],
            }
        ),
    )
    _write(
        run_dir / "finetune_stats.md",
        "\n".join(
            [
                "| Metric | Value |",
                "| Best loss | 1.2345 |",
                "| Checkpoint size | 123.4 MB |",
                "| Epoch | Loss |",
                "| 1 | 1.4000 |",
                "| 2 | 1.2345 |",
            ]
        ),
    )
    _write(
        run_dir / "distill_stats.md",
        "\n".join(
            [
                "| Metric | Value |",
                "| Best total loss | 0.9876 |",
                "| Best Recall@1 (student vs teacher) | 0.321 |",
                "| Compression ratio | 3.9× (86M → 22M params) |",
            ]
        ),
    )
    _write(run_dir / "multimodal_features.md", "OCR: 1/2 frames have text\nWorld model unavailable\n")
    _write(run_dir / "3d_map" / "map_stats.json", json.dumps({"method": "sfm", "points": 10, "poses": 5}))
    _write(run_dir / "3d_map" / "gaussian_splat.ply", "ply")
    _write(run_dir / "edge_models" / "dino_local.onnx", "onnx")
    np.savez(run_dir / "edge_models" / "gallery.npz", embeddings=np.ones((2, 4), dtype=np.float32))

    summary = LocalRunLoader(run_dir).load()

    assert summary.video_name == "mission_a"
    assert summary.detection_stats is not None
    assert summary.detection_stats.total_objects == 3
    assert summary.frames[0].caption == "road"
    assert summary.frames[1].qwen_caption == "detailed highway caption"
    assert summary.frames[0].asr_text == "hello"
    assert summary.tracking_stats is not None
    assert summary.tracking_stats.unique_track_ids == 2
    assert summary.run_health.tracking_filter_fallback_used is True
    assert summary.embedding_stats is not None
    assert summary.embedding_stats.embedding_dim == 4
    assert summary.map_stats is not None
    assert summary.map_stats.points == 10
    assert summary.has_3d_map is True
    assert summary.has_edge_model is True
    assert summary.artifact_inventory.total_files >= 10
    assert summary.run_health.world_model_ok is False
    assert summary.run_health.ocr_coverage == 0.5
    assert summary.diagnostics.modality_completeness == 5.5 / 8.0
    assert summary.diagnostics.detection_density_per_frame == 1.5
    assert round(summary.diagnostics.detection_count_cv, 3) == 0.333
    assert summary.diagnostics.tracking_fragmentation == 1.0
    assert summary.diagnostics.track_persistence == 1.0
    assert summary.diagnostics.map_points_per_pose == 2.0
    assert summary.diagnostics.map_pose_coverage == 1.0
    assert round(summary.diagnostics.adaptation_efficiency, 3) == 0.082
    assert summary.diagnostics.quality_score > 0


def test_local_run_loader_accepts_map_builder_key_names(tmp_path):
    from selfsuvis.analytics import LocalRunLoader

    run_dir = tmp_path / "mission_b"
    _write(
        run_dir / "frames_metadata.json",
        json.dumps(
            {
                "video_id": "mission_b",
                "fps": 2.0,
                "frame_count": 1,
                "duration_sec": 0.5,
                "frames": [{"path": "frame_1.jpg", "t_sec": 0.0}],
            }
        ),
    )
    _write(
        run_dir / "3d_map" / "map_stats.json",
        json.dumps({"method": "sfm", "point_count": 10, "sfm_poses": 10}),
    )

    summary = LocalRunLoader(run_dir).load()

    assert summary.map_stats is not None
    assert summary.map_stats.points == 10
    assert summary.map_stats.poses == 10


def test_local_run_loader_parses_ocr_coverage_from_markdown_table(tmp_path):
    from selfsuvis.analytics import LocalRunLoader

    run_dir = tmp_path / "mission_c"
    _write(
        run_dir / "frames_metadata.json",
        json.dumps(
            {
                "video_id": "mission_c",
                "fps": 2.0,
                "frame_count": 51,
                "duration_sec": 25.5,
                "frames": [{"path": "frame_1.jpg", "t_sec": 0.0}],
            }
        ),
    )
    _write(
        run_dir / "multimodal_features.md",
        "\n".join(
            [
                "# Multimodal Features",
                "",
                "Total frames: 51",
                "",
                "| Step | Status | Detail |",
                "|------|--------|--------|",
                "| OCR | ✓ | 24 frames with text |",
            ]
        ),
    )

    summary = LocalRunLoader(run_dir).load()

    assert summary.run_health.ocr_coverage == 24 / 51


def test_local_run_loader_does_not_treat_pca_anchors_as_sfm_pose_coverage(tmp_path):
    from selfsuvis.analytics import LocalRunLoader

    run_dir = tmp_path / "mission_pca"
    _write(
        run_dir / "frames_metadata.json",
        json.dumps(
            {
                "video_id": "mission_pca",
                "fps": 2.0,
                "frame_count": 4,
                "duration_sec": 2.0,
                "frames": [
                    {"path": f"frame_{idx}.jpg", "t_sec": idx * 0.5}
                    for idx in range(4)
                ],
            }
        ),
    )
    _write(
        run_dir / "3d_map" / "map_stats.json",
        json.dumps(
            {
                "method": "pca_pixels",
                "points": 51,
                "poses": 4,
                "sfm_poses": 0,
                "frame_anchor_count": 4,
                "quality_degraded": True,
            }
        ),
    )

    summary = LocalRunLoader(run_dir).load()

    assert summary.map_stats is not None
    assert summary.map_stats.poses == 4
    assert summary.map_stats.sfm_poses == 0
    assert summary.diagnostics.map_pose_coverage == 0.0
    assert summary.diagnostics.map_points_per_pose == 0.0
    assert any(
        "0 SfM poses, 4 frame anchors" in warning
        for warning in summary.run_health.warnings
    )


def test_targeted_coverage_score_saturates_sparse_expert_passes():
    score = loader._targeted_coverage_score(0.39, n_frames=51, target_frames=20)

    assert 0.99 <= score <= 1.0
