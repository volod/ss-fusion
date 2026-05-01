from selfsuvis.pipeline.workflows.local.steps_model_advisor import build_model_run_advisor


def test_model_run_advisor_recommends_stronger_qwen_for_parse_errors():
    advisor = build_model_run_advisor(
        [
            {
                "name": "drone_mission",
                "analysis_summary": {
                    "run_health": {
                        "qwen_parse_error_count": 10,
                        "qwen_caption_coverage": 0.04,
                    },
                    "diagnostics": {"map_pose_coverage": 0.0, "artifact_mb_per_min": 12000},
                    "map_stats": {"degraded": True},
                },
            }
        ],
        resources={"vram_gb": 11.9, "free_vram_gb": 11.4, "ram_gb": 62.0},
        env_values={"GEMMA_API_MODEL": "gemma4:e4b", "QWEN_MODEL": "qwen2.5vl:3b"},
    )

    assert advisor["recommended_env_updates"]["QWEN_MODEL"] == "qwen2.5vl:7b"
    assert advisor["recommended_env_updates"]["REASONING_MODEL"] == "qwen3:14b"
    assert advisor["recommended_env_updates"]["UNIDRIVE_ENABLED"] == "true"
    assert any(
        finding["code"] == "qwen_structured_captioning_degraded"
        for finding in advisor["findings"]
    )
    assert any(finding["code"] == "sfm_pose_recovery_degraded" for finding in advisor["findings"])
