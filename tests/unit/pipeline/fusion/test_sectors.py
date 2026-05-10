import unittest


class TestSectors(unittest.TestCase):
    def test_nearby_positions_share_overlapping_sector_ids(self):
        from selfsuvis.pipeline.fusion.sectors import (
            sectorize_global_positions,
            unique_sector_sequence,
        )

        origin = {"lat": 50.4501, "lon": 30.5234, "alt": 100.0}
        traj_a = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 20.0, "y": 10.0, "z": 0.0},
            {"x": 45.0, "y": 15.0, "z": 0.0},
        ]
        traj_b = [
            {"x": 10.0, "y": 5.0, "z": 0.0},
            {"x": 30.0, "y": 12.0, "z": 0.0},
            {"x": 55.0, "y": 20.0, "z": 0.0},
        ]

        seq_a = unique_sector_sequence(sectorize_global_positions(origin, traj_a, tile_size_m=50.0))
        seq_b = unique_sector_sequence(sectorize_global_positions(origin, traj_b, tile_size_m=50.0))

        self.assertTrue(set(seq_a) & set(seq_b))


if __name__ == "__main__":
    unittest.main()
