import tempfile
import unittest
from pathlib import Path

from PIL import Image


def _make_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(path)


class TestFieldStateStep(unittest.TestCase):
    def test_field_state_summary_contains_visibility(self):
        from selfsuvis.pipeline.workflows.local.steps_field_state import step_field_state

        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp) / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / "sample.mp4"
            video_path.write_bytes(b"")

            frame_list = []
            for idx in range(3):
                frame_path = video_dir / f"frame_{idx:04d}.jpg"
                _make_frame(frame_path)
                frame_list.append((str(frame_path), float(idx)))

            depth_result = {
                "skipped": False,
                "depth_results": [
                    {"frame_path": frame_list[0][0], "depth_confidence": 0.45, "near_ratio": 0.30},
                    {"frame_path": frame_list[1][0], "depth_error": True},
                    {"frame_path": frame_list[2][0], "depth_confidence": 0.55, "near_ratio": 0.35},
                ],
            }
            physical_state_result = {
                "skipped": False,
                "near_field_occupancy_density": 0.22,
                "free_space_estimate": 0.68,
            }
            caption_results = [
                {
                    "frame_path": frame_list[1][0],
                    "caption": "road with smoke ahead",
                    "caption_confidence": 0.40,
                },
            ]
            unidrive_result = {"skipped": True, "results": []}

            result = step_field_state(
                video_path=video_path,
                video_dir=video_dir,
                video_name="sample",
                frame_list=frame_list,
                depth_result=depth_result,
                physical_state_result=physical_state_result,
                caption_results=caption_results,
                unidrive_result=unidrive_result,
            )

            self.assertFalse(result["skipped"])
            self.assertIn("visibility", result["field_types"])
            self.assertIn("visibility", result["clip_level_fields"])
            self.assertTrue((video_dir / "field_state_summary.json").exists())


class TestThreatPrimitivesFieldIntegration(unittest.TestCase):
    def test_field_state_enriches_visibility_and_rf_primitives(self):
        from selfsuvis.pipeline.workflows.local.steps_threat_primitives import (
            step_threat_primitives,
        )

        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp)
            frame_list = []
            for idx in range(3):
                frame_path = video_dir / f"frame_{idx:04d}.jpg"
                _make_frame(frame_path)
                frame_list.append((str(frame_path), float(idx)))

            result = step_threat_primitives(
                physical_state_result={
                    "skipped": False,
                    "platform_pose_confidence": 0.8,
                    "near_field_occupancy_density": 0.10,
                    "tracked_object_velocities": {"mean": 0.0},
                    "free_space_estimate": 0.85,
                    "mean_bbox_uncertainty": 0.02,
                    "tracking_used": False,
                },
                field_state_result={
                    "skipped": False,
                    "clip_level_fields": {
                        "visibility": {
                            "mean": 0.55,
                            "uncertainty": 0.18,
                            "trend": "worsening",
                            "evidence_sources": [
                                "depth_confidence_drop",
                                "unidrive_visibility_semantics",
                            ],
                            "support_frames": [frame_list[0][0], frame_list[1][0]],
                        },
                        "rf_interference": {
                            "mean": 0.62,
                            "uncertainty": 0.22,
                            "trend": "worsening",
                            "evidence_sources": ["rf_spectral_flatness", "rf_occupied_bandwidth"],
                            "support_frames": [frame_list[1][0], frame_list[2][0]],
                        },
                    },
                },
                depth_result={
                    "skipped": False,
                    "depth_results": [
                        {"frame_path": frame_list[0][0], "depth_error": True},
                        {"frame_path": frame_list[1][0], "depth_error": True},
                    ],
                },
                caption_results=[
                    {"frame_path": frame_list[0][0], "caption_confidence": 0.40},
                    {"frame_path": frame_list[1][0], "caption_confidence": 0.45},
                ],
                unidrive_result={"skipped": True, "results": []},
                gemma_tracking_result={"skipped": True, "tracking_results": []},
                full_fusion_result={"skipped": True, "per_frame_object_states": []},
                frame_list=frame_list,
                sfm_poses=3,
                map_degraded=False,
                video_dir=video_dir,
                video_name="sample",
            )

            types = {p["type"] for p in result["primitives"]}
            self.assertIn("visibility_degradation", types)
            self.assertIn("rf_anomaly", types)


if __name__ == "__main__":
    unittest.main()
