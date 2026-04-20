from __future__ import annotations

import json

from selfsuvis.pipeline.fusion import run_platform_state_fusion
from selfsuvis.pipeline.workflows.local._common import VideoKnowledge


def test_platform_fusion_runs_with_gps_only(tmp_path):
    video_path = tmp_path / "mission.mp4"
    video_path.write_bytes(b"")

    frame_times = [0.0, 1.0, 2.0]
    gps_samples = [
        {"lat": 47.0, "lon": 8.0, "alt": 100.0},
        {"lat": 47.0, "lon": 8.00001, "alt": 100.5},
        {"lat": 47.0, "lon": 8.00002, "alt": 101.0},
    ]

    result = run_platform_state_fusion(
        video_path=str(video_path),
        frame_times_sec=frame_times,
        gps_samples=gps_samples,
    )

    assert result.status == "ok"
    assert len(result.posterior_samples) == 3
    assert result.telemetry_sources == ["gps"]
    assert result.summary()["final_state"] is not None


def test_platform_fusion_uses_imu_and_baro_sidecars(tmp_path):
    video_path = tmp_path / "mission.mp4"
    video_path.write_bytes(b"")

    imu_path = tmp_path / "mission.imu.jsonl"
    imu_path.write_text(
        "\n".join(
            [
                json.dumps({"t": 0.0, "ax": 0.1, "ay": 0.0, "az": -9.81}),
                json.dumps({"t": 0.5, "ax": 0.1, "ay": 0.0, "az": -9.81}),
                json.dumps({"t": 1.0, "ax": 0.0, "ay": 0.0, "az": -9.81}),
            ]
        ),
        encoding="utf-8",
    )
    baro_path = tmp_path / "mission.baro.jsonl"
    baro_path.write_text(
        "\n".join(
            [
                json.dumps({"t": 0.0, "alt_m": 100.0}),
                json.dumps({"t": 1.0, "alt_m": 100.6}),
                json.dumps({"t": 2.0, "alt_m": 101.1}),
            ]
        ),
        encoding="utf-8",
    )

    frame_times = [0.0, 1.0, 2.0]
    gps_samples = [
        {"lat": 47.0, "lon": 8.0, "alt": 100.0},
        {"lat": 47.0, "lon": 8.00001, "alt": 100.5},
        {"lat": 47.0, "lon": 8.00002, "alt": 101.0},
    ]

    result = run_platform_state_fusion(
        video_path=str(video_path),
        frame_times_sec=frame_times,
        gps_samples=gps_samples,
    )

    assert result.status == "ok"
    assert "imu" in result.telemetry_sources
    assert "barometer" in result.telemetry_sources
    assert result.measurement_counts["imu_accel"] == 3
    assert result.measurement_counts["barometer_altitude"] == 3


def test_video_knowledge_includes_fused_state_context():
    frame_times = [0.0, 1.0]
    gps_samples = [
        {"lat": 47.0, "lon": 8.0, "alt": 100.0},
        {"lat": 47.0, "lon": 8.00001, "alt": 100.5},
    ]
    result = run_platform_state_fusion(
        video_path="ignored.mp4",
        frame_times_sec=frame_times,
        gps_samples=gps_samples,
    )

    knowledge = VideoKnowledge(video_name="mission", duration_sec=2.0, frame_count=2)
    knowledge.add_state_fusion(result.posterior_samples)

    context = knowledge.context_for_frame(1.0)

    assert "[Fused platform state]:" in context
