"""Build lightweight YOLO-driven semantic scene graphs over 3D frame anchors.

The graph is intentionally observation-centric rather than metrically exact:
every detection is anchored to the best available 3D frame position
(ENU/GPS-derived pose in production, SfM/PCA frame point in local mode, timeline
fallback otherwise). Detections with the same label that recur near one
another are merged into persistent object nodes; co-visible or spatially close
nodes are connected with ``near`` edges.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from selfsuvis.pipeline.core import ensure_dir, settings


def _euclidean(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _bbox_area(bbox_norm: Optional[Iterable[float]]) -> float:
    if not bbox_norm:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _position_tuple(anchor: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(anchor.get("x", 0.0)),
        float(anchor.get("y", 0.0)),
        float(anchor.get("z", 0.0)),
    )


def _anchor_from_global_pose(global_pose_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not isinstance(global_pose_json, dict):
        return None
    if not {"tx", "ty", "tz"}.issubset(global_pose_json):
        return None
    return {
        "x": float(global_pose_json["tx"]),
        "y": float(global_pose_json["ty"]),
        "z": float(global_pose_json["tz"]),
    }


def _build_frame_anchor_lookup(
    frame_positions: Optional[List[Dict[str, Any]]],
    frame_observations: List[Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Tuple[float, Dict[str, float]]], str]:
    by_frame_path: Dict[str, Dict[str, float]] = {}
    by_frame_id: Dict[str, Dict[str, float]] = {}
    by_time: List[Tuple[float, Dict[str, float]]] = []

    for item in frame_positions or []:
        if not isinstance(item, dict):
            continue
        anchor = item.get("position")
        if not isinstance(anchor, dict):
            continue
        norm_anchor = _position_tuple(anchor)
        entry = {"x": norm_anchor[0], "y": norm_anchor[1], "z": norm_anchor[2]}
        frame_path = item.get("frame_path")
        frame_id = item.get("frame_id")
        t_sec = item.get("t_sec")
        if frame_path:
            by_frame_path[str(frame_path)] = entry
        if frame_id:
            by_frame_id[str(frame_id)] = entry
        if t_sec is not None:
            by_time.append((float(t_sec), entry))

    if by_time:
        return by_frame_path, by_frame_id, sorted(by_time, key=lambda pair: pair[0]), "map"

    has_pose = False
    for frame in frame_observations:
        anchor = _anchor_from_global_pose(frame.get("global_pose_json"))
        if anchor is None:
            continue
        has_pose = True
        if frame.get("frame_path"):
            by_frame_path[str(frame["frame_path"])] = anchor
        if frame.get("frame_id"):
            by_frame_id[str(frame["frame_id"])] = anchor
        if frame.get("id"):
            by_frame_id[str(frame["id"])] = anchor
        by_time.append((float(frame.get("t_sec", 0.0)), anchor))

    anchor_source = "enu" if has_pose else "timeline"
    return by_frame_path, by_frame_id, sorted(by_time, key=lambda pair: pair[0]), anchor_source


def _nearest_anchor(
    frame: Dict[str, Any],
    *,
    by_frame_path: Dict[str, Dict[str, float]],
    by_frame_id: Dict[str, Dict[str, float]],
    by_time: List[Tuple[float, Dict[str, float]]],
    anchor_source: str,
) -> Dict[str, float]:
    frame_path = frame.get("frame_path")
    if frame_path and frame_path in by_frame_path:
        return by_frame_path[frame_path]

    frame_id = frame.get("frame_id") or frame.get("id")
    if frame_id and str(frame_id) in by_frame_id:
        return by_frame_id[str(frame_id)]

    t_sec = float(frame.get("t_sec", 0.0))
    if by_time:
        anchor_time, anchor = min(by_time, key=lambda pair: abs(pair[0] - t_sec))
        tolerance = 1.5 if anchor_source == "map" else 5.0
        if abs(anchor_time - t_sec) <= tolerance:
            return anchor

    return {"x": t_sec, "y": 0.0, "z": 0.0}


def build_semantic_environment_graph(
    frame_observations: List[Dict[str, Any]],
    *,
    graph_id: str,
    frame_positions: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build a semantic environment graph from YOLO detections.

    Parameters
    ----------
    frame_observations:
        Per-frame dicts containing ``frame_path``, ``t_sec``, and a
        ``detections`` list (or ``frame_facts_json['yolo_detections']``).
    graph_id:
        Stable graph identifier such as a mission ID or local-run video ID.
    frame_positions:
        Optional list of frame anchor positions from SfM/PCA/local-map output.
    output_path:
        Optional JSON file destination.
    """
    by_frame_path, by_frame_id, by_time, anchor_source = _build_frame_anchor_lookup(
        frame_positions, frame_observations
    )
    coordinate_frame = "timeline" if anchor_source == "timeline" else anchor_source
    cluster_radius = (
        settings.YOLO_SSG_CLUSTER_RADIUS_PCA
        if anchor_source == "map"
        else settings.YOLO_SSG_CLUSTER_RADIUS_METERS
    )
    near_radius = (
        settings.YOLO_SSG_NEAR_EDGE_RADIUS_PCA
        if anchor_source == "map"
        else settings.YOLO_SSG_NEAR_EDGE_RADIUS_METERS
    )

    nodes: List[Dict[str, Any]] = []
    frame_assignments: Dict[str, List[str]] = defaultdict(list)
    observations: List[Dict[str, Any]] = []

    def assign_detection(frame: Dict[str, Any], detection: Dict[str, Any]) -> Optional[str]:
        label = str(detection.get("label", "")).strip()
        if not label:
            return None
        anchor = _nearest_anchor(
            frame,
            by_frame_path=by_frame_path,
            by_frame_id=by_frame_id,
            by_time=by_time,
            anchor_source=anchor_source,
        )
        pos = _position_tuple(anchor)
        candidate_idx = None
        candidate_dist = None
        for idx, node in enumerate(nodes):
            if node["label"] != label:
                continue
            dist = _euclidean(pos, _position_tuple(node["position"]))
            if dist > cluster_radius:
                continue
            if candidate_dist is None or dist < candidate_dist:
                candidate_idx = idx
                candidate_dist = dist

        obs = {
            "frame_id": str(frame.get("frame_id") or frame.get("id") or frame.get("frame_path") or ""),
            "frame_path": frame.get("frame_path"),
            "t_sec": float(frame.get("t_sec", 0.0)),
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "label": label,
            "priority": int(detection.get("priority", 4)),
            "priority_label": str(detection.get("priority_label", "other")),
            "confidence": float(detection.get("confidence", 0.0)),
            "bbox_area_norm": _bbox_area(detection.get("bbox_norm")),
            "mask_area_norm": float(detection.get("mask_area_norm") or 0.0),
        }
        observations.append(obs)

        if candidate_idx is None:
            node_id = f"{graph_id}:node:{len(nodes):04d}"
            nodes.append(
                {
                    "id": node_id,
                    "label": label,
                    "priority": obs["priority"],
                    "priority_label": obs["priority_label"],
                    "position": dict(obs["position"]),
                    "observations": 1,
                    "confidence_sum": obs["confidence"],
                    "bbox_area_sum": obs["bbox_area_norm"],
                    "mask_area_sum": obs["mask_area_norm"],
                    "first_seen_t_sec": obs["t_sec"],
                    "last_seen_t_sec": obs["t_sec"],
                    "frames": [obs["frame_id"]],
                }
            )
            return node_id

        node = nodes[candidate_idx]
        count = node["observations"]
        node["position"] = {
            "x": (node["position"]["x"] * count + pos[0]) / (count + 1),
            "y": (node["position"]["y"] * count + pos[1]) / (count + 1),
            "z": (node["position"]["z"] * count + pos[2]) / (count + 1),
        }
        node["observations"] += 1
        node["confidence_sum"] += obs["confidence"]
        node["bbox_area_sum"] += obs["bbox_area_norm"]
        node["mask_area_sum"] += obs["mask_area_norm"]
        node["first_seen_t_sec"] = min(node["first_seen_t_sec"], obs["t_sec"])
        node["last_seen_t_sec"] = max(node["last_seen_t_sec"], obs["t_sec"])
        if obs["frame_id"] not in node["frames"]:
            node["frames"].append(obs["frame_id"])
        return node["id"]

    for frame in frame_observations:
        detections = frame.get("detections")
        if detections is None:
            detections = (frame.get("frame_facts_json") or {}).get("yolo_detections", [])
        assigned: List[str] = []
        for detection in detections or []:
            node_id = assign_detection(frame, detection)
            if node_id:
                assigned.append(node_id)
        frame_key = str(frame.get("frame_id") or frame.get("id") or frame.get("frame_path") or "")
        if frame_key:
            frame_assignments[frame_key] = assigned

    filtered_nodes: List[Dict[str, Any]] = []
    kept_ids: set[str] = set()
    for node in nodes:
        if node["observations"] < settings.YOLO_SSG_MIN_OBSERVATIONS:
            continue
        count = node["observations"]
        node["confidence_mean"] = round(node.pop("confidence_sum") / count, 4)
        node["bbox_area_mean"] = round(node.pop("bbox_area_sum") / count, 6)
        node["mask_area_mean"] = round(node.pop("mask_area_sum") / count, 6)
        node["frame_count"] = len(node["frames"])
        filtered_nodes.append(node)
        kept_ids.add(node["id"])

    for frame_key, node_ids in list(frame_assignments.items()):
        frame_assignments[frame_key] = [node_id for node_id in node_ids if node_id in kept_ids]

    co_observed: Counter[Tuple[str, str]] = Counter()
    for node_ids in frame_assignments.values():
        unique = sorted(set(node_ids))
        for idx, source in enumerate(unique):
            for target in unique[idx + 1 :]:
                co_observed[(source, target)] += 1

    edges: List[Dict[str, Any]] = []
    for idx, source in enumerate(filtered_nodes):
        source_pos = _position_tuple(source["position"])
        for target in filtered_nodes[idx + 1 :]:
            target_pos = _position_tuple(target["position"])
            distance = _euclidean(source_pos, target_pos)
            shared_frames = co_observed.get((source["id"], target["id"]), 0)
            if distance > near_radius and shared_frames == 0:
                continue
            edges.append(
                {
                    "source": source["id"],
                    "target": target["id"],
                    "relation": "near",
                    "distance": round(distance, 3),
                    "shared_frames": shared_frames,
                }
            )

    label_counts = Counter(node["label"] for node in filtered_nodes)
    priority_counts = Counter(node["priority_label"] for node in filtered_nodes)
    graph = {
        "graph_id": graph_id,
        "builder": "yolo_ssg",
        "anchor_source": anchor_source,
        "coordinate_frame": coordinate_frame,
        "summary": {
            "node_count": len(filtered_nodes),
            "edge_count": len(edges),
            "observation_count": len(observations),
            "label_counts": dict(sorted(label_counts.items())),
            "priority_counts": dict(sorted(priority_counts.items())),
        },
        "nodes": filtered_nodes,
        "edges": edges,
        "frame_assignments": dict(frame_assignments),
    }

    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(str(output_path.parent))
        output_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        graph["output_path"] = str(output_path)

    return graph


def write_semantic_graph_markdown(
    graph: Dict[str, Any],
    output_path: str | Path,
    *,
    title: str,
) -> str:
    """Write a short markdown summary for local/report consumption."""
    output_path = Path(output_path)
    ensure_dir(str(output_path.parent))
    summary = graph.get("summary", {})
    lines = [
        f"# {title}",
        "",
        f"- Builder: `yolo_ssg`",
        f"- Anchor source: `{graph.get('anchor_source', 'unknown')}`",
        f"- Coordinate frame: `{graph.get('coordinate_frame', 'unknown')}`",
        f"- Nodes: **{summary.get('node_count', 0)}**",
        f"- Edges: **{summary.get('edge_count', 0)}**",
        f"- Observations: **{summary.get('observation_count', 0)}**",
        "",
        "## Priority counts",
    ]
    for label, count in sorted((summary.get("priority_counts") or {}).items()):
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    lines.append("## Top semantic nodes")
    for node in sorted(graph.get("nodes", []), key=lambda item: (-item["observations"], item["label"]))[:12]:
        pos = node["position"]
        lines.append(
            f"- `{node['label']}` at ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) "
            f"| obs={node['observations']} | mean_conf={node['confidence_mean']:.2f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)
