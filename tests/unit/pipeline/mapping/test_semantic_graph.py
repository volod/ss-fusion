"""Unit tests for the YOLO semantic scene graph builder."""

from selfsuvis.pipeline.mapping.semantic_graph import build_semantic_environment_graph


def test_semantic_graph_clusters_same_label_nearby_observations(tmp_path):
    frames = [
        {
            "id": "f1",
            "frame_path": "a.jpg",
            "t_sec": 0.0,
            "detections": [
                {
                    "label": "truck",
                    "confidence": 0.9,
                    "bbox_norm": [0.1, 0.1, 0.4, 0.5],
                    "priority": 2,
                    "priority_label": "vehicle",
                    "mask_area_norm": 0.08,
                }
            ],
        },
        {
            "id": "f2",
            "frame_path": "b.jpg",
            "t_sec": 1.0,
            "detections": [
                {
                    "label": "truck",
                    "confidence": 0.85,
                    "bbox_norm": [0.12, 0.1, 0.41, 0.49],
                    "priority": 2,
                    "priority_label": "vehicle",
                    "mask_area_norm": 0.07,
                }
            ],
        },
    ]
    anchors = [
        {"frame_id": "f1", "t_sec": 0.0, "position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        {"frame_id": "f2", "t_sec": 1.0, "position": {"x": 0.6, "y": 0.0, "z": 0.0}},
    ]

    graph = build_semantic_environment_graph(
        frames,
        graph_id="demo",
        frame_positions=anchors,
        output_path=tmp_path / "graph.json",
    )

    assert graph["summary"]["node_count"] == 1
    assert graph["summary"]["observation_count"] == 2
    assert graph["nodes"][0]["label"] == "truck"
    assert graph["nodes"][0]["observations"] == 2
    assert graph["frame_assignments"]["f1"] == graph["frame_assignments"]["f2"]


def test_semantic_graph_creates_near_edge_for_co_visible_nodes():
    frames = [
        {
            "id": "f1",
            "frame_path": "a.jpg",
            "t_sec": 0.0,
            "detections": [
                {
                    "label": "truck",
                    "confidence": 0.9,
                    "bbox_norm": [0.1, 0.1, 0.4, 0.5],
                    "priority": 2,
                    "priority_label": "vehicle",
                    "mask_area_norm": 0.08,
                },
                {
                    "label": "person",
                    "confidence": 0.87,
                    "bbox_norm": [0.5, 0.2, 0.7, 0.8],
                    "priority": 1,
                    "priority_label": "human",
                    "mask_area_norm": 0.04,
                },
            ],
        }
    ]
    anchors = [
        {"frame_id": "f1", "t_sec": 0.0, "position": {"x": 0.0, "y": 0.0, "z": 0.0}},
    ]

    graph = build_semantic_environment_graph(frames, graph_id="demo", frame_positions=anchors)

    assert graph["summary"]["node_count"] == 2
    assert graph["summary"]["edge_count"] == 1
    assert graph["edges"][0]["shared_frames"] == 1
