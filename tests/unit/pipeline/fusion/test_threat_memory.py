import json
import tempfile
import unittest
from pathlib import Path


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestThreatMemoryPersistence(unittest.TestCase):

    def test_persist_threat_memory_summarizes_repeated_conflicts(self):
        from selfsuvis.pipeline.fusion.threat_memory import persist_threat_memory

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for video_name in ("video_a", "video_b"):
                video_dir = output_dir / video_name
                _write_json(
                    video_dir / "local_threat_assessment.json",
                    {
                        "local_threat_score": 0.62,
                        "automation_confidence": 0.44,
                        "trust_penalty": 0.31,
                        "source_pair_conflicts": [
                            {"pattern": "occupancy_vs_unidrive_clear", "count": 1, "severity": 0.30},
                            {"pattern": "caption_confidence_vs_depth_failure", "count": 1, "severity": 0.26},
                        ],
                        "top_threats": [
                            {"type": "collision_risk", "score": 0.62, "evidence": {"evidence_sources": ["near_field_occupancy", "object_velocity"]}}
                        ],
                    },
                )
                _write_json(
                    video_dir / "threat_primitives.json",
                    {
                        "contradiction_signals": [
                            {"pattern": "occupancy_vs_unidrive_clear", "severity": 0.30},
                            {"pattern": "caption_confidence_vs_depth_failure", "severity": 0.26},
                        ],
                        "primitives": [
                            {"type": "collision_risk", "evidence_sources": ["near_field_occupancy", "object_velocity"]},
                        ],
                    },
                )
                _write_json(
                    video_dir / "physical_state_summary.json",
                    {
                        "platform_pose_confidence": 0.66,
                        "tracking_used": True,
                        "confirmed_tracks": 4,
                    },
                )
                _write_json(
                    video_dir / "full_state_fusion.json",
                    {
                        "platform": {"origin_lla": {"lat": 50.4501, "lon": 30.5234, "alt": 100.0}},
                        "map_state": {
                            "smoothed_samples": [
                                {"t_sec": 0.0, "position_enu_m": {"x": 0.0, "y": 0.0, "z": 0.0}},
                                {"t_sec": 1.0, "position_enu_m": {"x": 20.0, "y": 5.0, "z": 0.0}},
                            ]
                        },
                    },
                )

            payload = persist_threat_memory(
                output_dir,
                [
                    {"name": "video_a", "video_dir": str(output_dir / "video_a")},
                    {"name": "video_b", "video_dir": str(output_dir / "video_b")},
                ],
                {"route_advisories": [{"route_id": "route_test", "recommended_action": "inspect_sensor"}]},
            )

            summary = payload["history_summary"]
            self.assertEqual(payload["record_count"], 2)
            self.assertIn("unidrive_perception_consistency_check", summary["sensor_health_flags"])
            self.assertIn("depth_caption_crosscheck_required", summary["sensor_health_flags"])
            self.assertTrue(summary["route_level_trust_warnings"])
            self.assertTrue((output_dir / "threat_memory").exists())


if __name__ == "__main__":
    unittest.main()
