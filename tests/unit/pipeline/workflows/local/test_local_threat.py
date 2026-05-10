import tempfile
import unittest
from pathlib import Path


class TestLocalThreatContradictions(unittest.TestCase):
    def test_contradictions_reduce_automation_confidence_and_policy_biases_inspect_sensor(self):
        from selfsuvis.pipeline.workflows.local.steps_local_threat import step_local_threat
        from selfsuvis.pipeline.workflows.local.steps_policy import step_policy

        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp)
            primitive = {
                "type": "collision_risk",
                "score": 0.68,
                "uncertainty": 0.10,
                "spatial_support": [str(video_dir / "frame_0001.jpg")],
                "temporal_persistence": 4,
                "evidence_sources": ["near_field_occupancy", "object_velocity"],
            }
            result = step_local_threat(
                threat_primitives_result={
                    "skipped": False,
                    "primitives": [primitive],
                    "contradiction_signals": [],
                },
                video_dir=video_dir,
                video_name="sample",
                unidrive_rows=[
                    {
                        "frame_path": str(video_dir / "frame_0001.jpg"),
                        "understanding": {"risk_level": "low"},
                        "perception": {"drivable_area": "clear"},
                        "planning": {"recommended_action": "continue"},
                    }
                ],
                physical_state={"near_field_occupancy_density": 0.30},
            )

            self.assertGreater(result["local_threat_score"], 0.0)
            self.assertLess(result["automation_confidence"], 0.75)
            self.assertGreater(result["trust_penalty"], 0.0)
            self.assertNotIn("recommended_action", result)
            self.assertTrue(result["source_pair_conflicts"])

            policy = step_policy(result, video_dir, "sample")
            self.assertEqual(policy["recommended_action"], "inspect_sensor")


if __name__ == "__main__":
    unittest.main()
