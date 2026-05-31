from ssv_vdp.steps import caption as steps_caption, embed as steps_embed


def test_adaptive_sparse_budget_scales_down_short_clip_sampling():
    frame_list = [(f"frame_{i:03d}.jpg", i * 0.2) for i in range(51)]

    qwen_budget = steps_caption._adaptive_sparse_budget(
        frame_list,
        configured_max=20,
        seconds_per_sample=0.9,
        floor=8,
    )
    unidrive_budget = steps_caption._adaptive_sparse_budget(
        frame_list,
        configured_max=24,
        seconds_per_sample=1.4,
        floor=6,
    )

    assert qwen_budget == 12
    assert unidrive_budget == 8


def test_search_excludes_near_temporal_neighbours():
    class FakeStore:
        def search(self, _query_vec, limit):
            assert limit == 12
            return [
                {"score": 0.99, "payload": {"frame_path": "query.jpg", "t_sec": 5.0}},
                {"score": 0.98, "payload": {"frame_path": "near.jpg", "t_sec": 5.5}},
                {"score": 0.80, "payload": {"frame_path": "good_a.jpg", "t_sec": 7.0}},
                {"score": 0.70, "payload": {"frame_path": "good_b.jpg", "t_sec": 2.0}},
            ]

    results = steps_embed._search(
        query_vec=None,
        store=FakeStore(),
        is_qdrant=False,
        top_k=3,
        video_id="vid-1",
        exclude_frame_path="query.jpg",
        exclude_t_sec=5.0,
        min_time_gap_sec=1.0,
    )

    assert [row["payload"]["frame_path"] for row in results] == ["good_a.jpg", "good_b.jpg"]
