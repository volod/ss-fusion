import unittest


class TestMultimodalPairMining(unittest.TestCase):
    def test_pair_mining_adds_auxiliary_pairs_deterministically(self):
        from ssv_vdp.steps.ssl import _build_multimodal_pair_mining

        frame_list = [
            ("/tmp/frame_0000.jpg", 0.0),
            ("/tmp/frame_0001.jpg", 1.0),
            ("/tmp/frame_0002.jpg", 2.0),
            ("/tmp/frame_0003.jpg", 3.0),
            ("/tmp/frame_0004.jpg", 4.0),
        ]
        track_map = {
            11: [
                ("/tmp/frame_0000.jpg", [0.1, 0.1, 0.2, 0.2], 0.0),
                ("/tmp/frame_0002.jpg", [0.1, 0.1, 0.2, 0.2], 2.0),
                ("/tmp/frame_0004.jpg", [0.1, 0.1, 0.2, 0.2], 4.0),
            ]
        }
        depth_result = {
            "depth_results": [
                {"frame_path": "/tmp/frame_0000.jpg", "near_ratio": 0.20, "depth_confidence": 0.9},
                {"frame_path": "/tmp/frame_0001.jpg", "near_ratio": 0.22, "depth_confidence": 0.9},
                {"frame_path": "/tmp/frame_0002.jpg", "near_ratio": 0.21, "depth_confidence": 0.9},
                {"frame_path": "/tmp/frame_0003.jpg", "near_ratio": 0.23, "depth_confidence": 0.9},
            ]
        }
        platform_state_fusion = {
            "posterior_samples": [
                {
                    "t_sec": 0.0,
                    "velocity_enu_mps": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "covariance_trace": 4.0,
                },
                {
                    "t_sec": 1.0,
                    "velocity_enu_mps": {"x": 1.1, "y": 0.0, "z": 0.0},
                    "covariance_trace": 4.0,
                },
                {
                    "t_sec": 2.0,
                    "velocity_enu_mps": {"x": 0.9, "y": 0.0, "z": 0.0},
                    "covariance_trace": 5.0,
                },
                {
                    "t_sec": 3.0,
                    "velocity_enu_mps": {"x": 1.0, "y": 0.1, "z": 0.0},
                    "covariance_trace": 4.0,
                },
                {
                    "t_sec": 4.0,
                    "velocity_enu_mps": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "covariance_trace": 4.0,
                },
            ]
        }
        full_fusion_result = {
            "smoothed_trajectory": [
                {"position_enu_m": {"x": 0.0, "y": 0.0, "z": 0.0}},
                {"position_enu_m": {"x": 0.6, "y": 0.0, "z": 0.0}},
                {"position_enu_m": {"x": 1.1, "y": 0.1, "z": 0.0}},
                {"position_enu_m": {"x": 1.7, "y": 0.1, "z": 0.0}},
                {"position_enu_m": {"x": 2.2, "y": 0.1, "z": 0.0}},
            ]
        }
        physical_state_result = {
            "near_field_occupancy_density": 0.18,
            "free_space_estimate": 0.77,
            "platform_pose_confidence": 0.9,
        }

        pairs_a, stats_a, sfm_rows_a = _build_multimodal_pair_mining(
            frame_list,
            track_map,
            depth_result,
            platform_state_fusion,
            full_fusion_result,
            physical_state_result,
            min_gap=2,
            max_gap=5,
        )
        pairs_b, stats_b, sfm_rows_b = _build_multimodal_pair_mining(
            frame_list,
            track_map,
            depth_result,
            platform_state_fusion,
            full_fusion_result,
            physical_state_result,
            min_gap=2,
            max_gap=5,
        )

        self.assertEqual(stats_a, stats_b)
        self.assertEqual(sfm_rows_a, sfm_rows_b)
        self.assertGreater(stats_a["track_pairs"], 0)
        self.assertGreater(stats_a["depth_pairs"], 0)
        self.assertGreater(stats_a["motion_pairs"], 0)
        self.assertGreater(stats_a["pose_overlap_pairs"], 0)
        self.assertGreater(stats_a["total_pairs"], stats_a["track_pairs"])
        self.assertEqual(len(pairs_a), stats_a["total_pairs"])


if __name__ == "__main__":
    unittest.main()
