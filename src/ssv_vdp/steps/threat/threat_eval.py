"""Offline calibration and evaluation artifacts for threat outputs."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..common import write_json_artifact


def write_threat_calibration(
    output_dir: Path,
    per_video_stats: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    records = _collect_video_records(per_video_stats)
    histogram = _histogram([float(r.get("local_threat_score", 0.0) or 0.0) for r in records])
    disagreement_trend = [
        {
            "video_id": str(record.get("video_id", "")),
            "index": idx,
            "disagreement_rate": float(record.get("disagreement_rate", 0.0) or 0.0),
            "automation_confidence": float(record.get("automation_confidence", 1.0) or 1.0),
        }
        for idx, record in enumerate(records)
    ]
    persistence_sweeps = _persistence_threshold_sweeps(records)
    labels = _load_eval_labels(output_dir)
    reliability = _reliability_diagram(records, labels)
    payload = {
        "record_count": len(records),
        "reliability_diagram": reliability,
        "threat_score_histogram": histogram,
        "disagreement_rate_trends": disagreement_trend,
        "persistence_threshold_sweeps": persistence_sweeps,
    }
    out = output_dir / "threat_calibration.json"
    write_json_artifact(out, payload)
    return payload


def write_threat_eval_summary(
    output_dir: Path,
    per_video_stats: Sequence[dict[str, Any]],
    *,
    label_path: Path | None = None,
) -> dict[str, Any]:
    records = _collect_video_records(per_video_stats)
    labels = _load_eval_labels(output_dir, label_path=label_path)
    matched = []
    for record in records:
        video_id = str(record.get("video_id", ""))
        label = labels.get(video_id)
        if not label:
            continue
        matched.append((record, label))

    payload = {
        "label_source": str((label_path or (output_dir / "threat_eval_labels.json")).name),
        "matched_records": len(matched),
        "threat_detection_metrics": _threat_detection_metrics(matched),
        "action_policy_metrics": _action_policy_metrics(matched),
        "per_video": [
            {
                "video_id": str(record.get("video_id", "")),
                "predicted_threat_types": list(record.get("predicted_threat_types") or []),
                "label_threat_types": list(label.get("threat_types") or []),
                "recommended_action": str(record.get("recommended_action", "continue")),
                "label_action": str(label.get("recommended_action", "")),
                "outcome": str(label.get("outcome", "")),
            }
            for record, label in matched
        ],
    }
    out = output_dir / "threat_eval_summary.json"
    write_json_artifact(out, payload)
    return payload


def _collect_video_records(per_video_stats: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stats in per_video_stats:
        video_dir = Path(str(stats.get("video_dir", "")))
        if not video_dir.exists():
            continue
        local_threat = _load_json(video_dir / "local_threat_assessment.json")
        if not local_threat:
            continue
        primitives = _load_json(video_dir / "threat_primitives.json")
        policy = _load_json(video_dir / "policy_decision.json")
        records.append(
            {
                "video_id": video_dir.name,
                "video_dir": str(video_dir),
                "local_threat_score": float(local_threat.get("local_threat_score", 0.0) or 0.0),
                "automation_confidence": float(
                    local_threat.get("automation_confidence", 1.0) or 1.0
                ),
                "disagreement_rate": float(local_threat.get("disagreement_rate", 0.0) or 0.0),
                "top_threats": list(local_threat.get("top_threats") or []),
                "predicted_threat_types": [
                    str(item.get("type", ""))
                    for item in (local_threat.get("top_threats") or [])
                    if item.get("type")
                ],
                "recommended_action": str(
                    policy.get(
                        "recommended_action", local_threat.get("recommended_action", "continue")
                    )
                ),
                "primitives": list(primitives.get("primitives") or []),
            }
        )
    return records


def _histogram(values: Sequence[float], *, n_bins: int = 5) -> list[dict[str, Any]]:
    bins = [0 for _ in range(n_bins)]
    for value in values:
        clipped = max(0.0, min(0.999999, float(value or 0.0)))
        idx = min(n_bins - 1, int(clipped * n_bins))
        bins[idx] += 1
    out = []
    for idx, count in enumerate(bins):
        start = idx / n_bins
        end = (idx + 1) / n_bins
        out.append({"bin_start": round(start, 2), "bin_end": round(end, 2), "count": count})
    return out


def _reliability_diagram(
    records: Sequence[dict[str, Any]], labels: dict[str, dict[str, Any]], *, n_bins: int = 5
) -> dict[str, Any]:
    rows = []
    for idx in range(n_bins):
        start = idx / n_bins
        end = (idx + 1) / n_bins
        bucket = [
            record
            for record in records
            if start <= float(record.get("automation_confidence", 0.0) or 0.0) < end
            or (idx == n_bins - 1 and float(record.get("automation_confidence", 0.0) or 0.0) == 1.0)
        ]
        if not bucket:
            rows.append(
                {
                    "bin_start": round(start, 2),
                    "bin_end": round(end, 2),
                    "count": 0,
                    "mean_confidence": None,
                    "observed_accuracy": None,
                }
            )
            continue
        matched = [
            labels.get(str(record.get("video_id", "")))
            for record in bucket
            if labels.get(str(record.get("video_id", "")))
        ]
        if matched:
            accuracy = sum(
                1
                for record in bucket
                if _action_matches_label(record, labels.get(str(record.get("video_id", "")), {}))
            ) / len(bucket)
            label_source = "human_or_outcome_labels"
        else:
            accuracy = None
            label_source = "no_labels_available"
        rows.append(
            {
                "bin_start": round(start, 2),
                "bin_end": round(end, 2),
                "count": len(bucket),
                "mean_confidence": round(
                    sum(float(r.get("automation_confidence", 0.0) or 0.0) for r in bucket)
                    / len(bucket),
                    4,
                ),
                "observed_accuracy": round(float(accuracy), 4) if accuracy is not None else None,
                "label_source": label_source,
            }
        )
    return {"bins": rows}


def _persistence_threshold_sweeps(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    sweeps = []
    for threshold in range(1, 6):
        scores = []
        for record in records:
            active = []
            for primitive in record.get("primitives") or []:
                support = max(
                    len([fp for fp in (primitive.get("spatial_support") or []) if fp]),
                    int(primitive.get("temporal_persistence", 0) or 0),
                )
                if support >= threshold:
                    active.append(float(primitive.get("score", 0.0) or 0.0))
            remaining = 1.0
            for score in active:
                remaining *= 1.0 - max(0.0, min(1.0, score))
            scores.append(1.0 - remaining if active else 0.0)
        sweeps.append(
            {
                "persist_min_frames": threshold,
                "mean_local_threat_score": round(sum(scores) / max(1, len(scores)), 4),
                "active_video_count": sum(1 for score in scores if score > 0.0),
            }
        )
    return sweeps


def _threat_detection_metrics(
    matched: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    tp = fp = fn = 0
    for record, label in matched:
        predicted = set(record.get("predicted_threat_types") or [])
        expected = set(label.get("threat_types") or [])
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
    }


def _action_policy_metrics(
    matched: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    if not matched:
        return {"accuracy": None, "count": 0}
    correct = sum(1 for record, label in matched if _action_matches_label(record, label))
    return {"accuracy": round(correct / len(matched), 4), "count": len(matched)}


def _action_matches_label(record: dict[str, Any], label: dict[str, Any]) -> bool:
    return str(record.get("recommended_action", "") or "") == str(
        label.get("recommended_action", "") or ""
    )


def _load_eval_labels(
    output_dir: Path, *, label_path: Path | None = None
) -> dict[str, dict[str, Any]]:
    path = label_path or (output_dir / "threat_eval_labels.json")
    if not path.exists():
        return {}
    payload = _load_json(path)
    if isinstance(payload.get("videos"), dict):
        return {str(k): dict(v) for k, v in payload["videos"].items() if isinstance(v, dict)}
    return {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
