import json
import tempfile
import unittest
from pathlib import Path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TestThreatEvaluationArtifacts(unittest.TestCase):
    def test_calibration_and_eval_outputs_are_written(self):
        from ssv_vdp.steps.threat_eval import (
            write_threat_calibration,
            write_threat_eval_summary,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for video_name, action, score, disagreement in (
                ("video_a", "inspect_sensor", 0.66, 0.30),
                ("video_b", "reduce_speed", 0.41, 0.10),
            ):
                video_dir = output_dir / video_name
                _write_json(
                    video_dir / "local_threat_assessment.json",
                    {
                        "local_threat_score": score,
                        "automation_confidence": 0.72 if video_name == "video_a" else 0.85,
                        "disagreement_rate": disagreement,
                        "top_threats": [
                            {
                                "type": "collision_risk"
                                if video_name == "video_a"
                                else "track_anomaly",
                                "score": score,
                                "evidence": {"evidence_sources": ["a", "b"]},
                            }
                        ],
                    },
                )
                _write_json(
                    video_dir / "policy_decision.json",
                    {"recommended_action": action},
                )
                _write_json(
                    video_dir / "threat_primitives.json",
                    {
                        "primitives": [
                            {
                                "type": "collision_risk"
                                if video_name == "video_a"
                                else "track_anomaly",
                                "score": score,
                                "spatial_support": ["f1.jpg"],
                                "temporal_persistence": 3,
                            }
                        ]
                    },
                )

            _write_json(
                output_dir / "threat_eval_labels.json",
                {
                    "videos": {
                        "video_a": {
                            "threat_types": ["collision_risk"],
                            "recommended_action": "inspect_sensor",
                            "outcome": "sensor_issue_confirmed",
                        },
                        "video_b": {
                            "threat_types": ["track_anomaly"],
                            "recommended_action": "reduce_speed",
                            "outcome": "safe_completion",
                        },
                    }
                },
            )

            stats = [
                {"name": "video_a", "video_dir": str(output_dir / "video_a")},
                {"name": "video_b", "video_dir": str(output_dir / "video_b")},
            ]
            calibration = write_threat_calibration(output_dir, stats)
            summary = write_threat_eval_summary(output_dir, stats)

            self.assertEqual(calibration["record_count"], 2)
            self.assertTrue(calibration["threat_score_histogram"])
            self.assertTrue(calibration["persistence_threshold_sweeps"])
            self.assertEqual(summary["matched_records"], 2)
            self.assertEqual(summary["action_policy_metrics"]["accuracy"], 1.0)
            self.assertTrue((output_dir / "threat_calibration.json").exists())
            self.assertTrue((output_dir / "threat_eval_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
