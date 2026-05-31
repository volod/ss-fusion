"""Demo helper for YOLO semantic scene graph (SSG) generation."""

from pathlib import Path
from typing import Any

from selfsuvis.pipeline.mapping import (
    build_semantic_environment_graph,
    write_semantic_graph_markdown,
)

from .common import _log


def step_build_semantic_environment_graph(
    *,
    video_id: str,
    video_name: str,
    video_dir: Path,
    yolo_sam_result: dict[str, Any],
    map_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a YOLO-driven semantic environment graph for local-run artifacts."""
    if yolo_sam_result.get("skipped"):
        return {"skipped": True, "reason": "yolo_step_skipped"}

    frame_positions = map_result.get("frame_positions") or []
    graph_dir = video_dir / "3d_map"
    graph_json = graph_dir / "semantic_environment_graph.json"
    graph_md = graph_dir / "semantic_environment_graph.md"
    graph = build_semantic_environment_graph(
        yolo_sam_result.get("detection_results", []),
        graph_id=video_id,
        frame_positions=frame_positions,
        output_path=graph_json,
    )
    write_semantic_graph_markdown(
        graph,
        graph_md,
        title=f"{video_name} — YOLO Semantic Environment Graph",
    )
    _log.info(
        "  [ok] YOLO SSG → %s (%d nodes, %d edges)",
        graph_json,
        graph.get("summary", {}).get("node_count", 0),
        graph.get("summary", {}).get("edge_count", 0),
    )
    return {
        "skipped": False,
        "graph": graph,
        "json_path": str(graph_json),
        "markdown_path": str(graph_md),
    }
