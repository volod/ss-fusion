import json
import tempfile
import unittest
from pathlib import Path


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestGlobalThreatAggregation(unittest.TestCase):
    def test_batch_aggregator_merges_shared_sectors(self):
        from ssv_vdp.steps.global_threat import step_global_threat

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            video_a = output_dir / "video_a"
            video_b = output_dir / "video_b"

            local_threat_a = {
                "local_threat_score": 0.62,
                "recommended_action": "reduce_speed",
                "top_threats": [
                    {
                        "type": "visibility_degradation",
                        "score": 0.62,
                        "evidence": {
                            "evidence_sources": ["depth_failure_rate", "caption_confidence"]
                        },
                    }
                ],
            }
            local_threat_b = {
                "local_threat_score": 0.71,
                "recommended_action": "reroute",
                "top_threats": [
                    {
                        "type": "collision_risk",
                        "score": 0.71,
                        "evidence": {
                            "evidence_sources": ["near_field_occupancy", "object_velocity"]
                        },
                    }
                ],
            }
            primitives_a = {
                "primitives": [
                    {
                        "type": "visibility_degradation",
                        "score": 0.62,
                        "uncertainty": 0.2,
                        "evidence_sources": ["depth_failure_rate", "caption_confidence"],
                    }
                ]
            }
            primitives_b = {
                "primitives": [
                    {
                        "type": "collision_risk",
                        "score": 0.71,
                        "uncertainty": 0.15,
                        "evidence_sources": ["near_field_occupancy", "object_velocity"],
                    }
                ]
            }
            physical = {"platform_pose_confidence": 0.8}
            full_fusion_a = {
                "platform": {"origin_lla": {"lat": 50.4501, "lon": 30.5234, "alt": 100.0}},
                "map_state": {
                    "smoothed_samples": [
                        {"t_sec": 0.0, "position_enu_m": {"x": 0.0, "y": 0.0, "z": 0.0}},
                        {"t_sec": 1.0, "position_enu_m": {"x": 15.0, "y": 5.0, "z": 0.0}},
                    ]
                },
            }
            full_fusion_b = {
                "platform": {"origin_lla": {"lat": 50.4501, "lon": 30.5234, "alt": 100.0}},
                "map_state": {
                    "smoothed_samples": [
                        {"t_sec": 0.0, "position_enu_m": {"x": 10.0, "y": 4.0, "z": 0.0}},
                        {"t_sec": 1.0, "position_enu_m": {"x": 25.0, "y": 8.0, "z": 0.0}},
                    ]
                },
            }

            _write_json(video_a / "local_threat_assessment.json", local_threat_a)
            _write_json(video_a / "threat_primitives.json", primitives_a)
            _write_json(video_a / "physical_state_summary.json", physical)
            _write_json(video_a / "full_state_fusion.json", full_fusion_a)

            _write_json(video_b / "local_threat_assessment.json", local_threat_b)
            _write_json(video_b / "threat_primitives.json", primitives_b)
            _write_json(video_b / "physical_state_summary.json", physical)
            _write_json(video_b / "full_state_fusion.json", full_fusion_b)

            result = step_global_threat(
                output_dir,
                [
                    {"name": "video_a", "video_dir": str(video_a)},
                    {"name": "video_b", "video_dir": str(video_b)},
                ],
            )

            self.assertFalse(result["skipped"])
            self.assertTrue((output_dir / "global_threat_summary.json").exists())
            self.assertGreaterEqual(len(result["sector_risk_levels"]), 1)
            overlapping = [
                row
                for row in result["sector_risk_levels"]
                if len(row.get("supporting_videos", [])) >= 2
            ]
            self.assertTrue(overlapping)
            self.assertTrue(result["route_advisories"])
            self.assertTrue(result["threat_corridor_graph"])


if __name__ == "__main__":
    unittest.main()
