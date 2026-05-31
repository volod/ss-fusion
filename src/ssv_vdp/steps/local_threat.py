"""Aggregate threat primitives into a clip-level local threat assessment.

This step consumes the structured threat primitives emitted by
``steps_threat_primitives.py`` and collapses them across the full video window
into a single local threat estimate.

Output artifact: ``local_threat_assessment.json``

Schema:
    {
      "local_threat_score": float,
      "top_threats": [
        {
          "type": str,
          "score": float,
          "evidence": {
            "evidence_sources": list[str],
            "support_frames": int,
            "temporal_persistence": int,
            "uncertainty": float,
          },
        },
        ...
      ],
      "summary": {...},
      "skipped": bool,
      "elapsed_sec": float,
    }
"""

import json
import time
from pathlib import Path
from typing import Any

from .common import _log
from .threat_contradictions import contradiction_signals_for_threat, summarize_contradictions

_PERSIST_MIN_FRAMES = 3
_TYPE_WEIGHTS: dict[str, float] = {
    "collision_risk": 1.00,
    "visibility_degradation": 0.85,
    "rf_anomaly": 0.70,
    "track_anomaly": 0.60,
    "pose_uncertain": 0.75,
}


def _support_frame_count(primitive: dict[str, Any]) -> int:
    return len([fp for fp in (primitive.get("spatial_support") or []) if fp])


def _persistence_factor(
    primitive: dict[str, Any],
    min_frames: int,
) -> float:
    support_frames = _support_frame_count(primitive)
    persistence = int(primitive.get("temporal_persistence", 0) or 0)
    observed = max(support_frames, persistence)
    if observed <= 0:
        return 0.0
    return min(1.0, observed / max(1, min_frames))


def _effective_score(
    primitive: dict[str, Any],
    min_frames: int,
) -> float:
    ptype = str(primitive.get("type", ""))
    base = float(primitive.get("score", 0.0) or 0.0)
    weight = _TYPE_WEIGHTS.get(ptype, 0.5)
    return min(1.0, base * _persistence_factor(primitive, min_frames) * weight)


def _primitive_evidence(primitive: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_sources": list(primitive.get("evidence_sources") or []),
        "support_frames": _support_frame_count(primitive),
        "temporal_persistence": int(primitive.get("temporal_persistence", 0) or 0),
        "uncertainty": float(primitive.get("uncertainty", 0.0) or 0.0),
    }


def _empty_result() -> dict[str, Any]:
    return {
        "local_threat_score": 0.0,
        "automation_confidence": 1.0,
        "trust_penalty": 0.0,
        "disagreement_count": 0,
        "disagreement_rate": 0.0,
        "source_pair_conflicts": [],
        "top_threats": [],
        "summary": {
            "persist_min_frames": _PERSIST_MIN_FRAMES,
            "n_input_primitives": 0,
            "n_active_primitives": 0,
            "active_types": [],
        },
        "elapsed_sec": 0.0,
    }


def step_local_threat(
    threat_primitives_result: dict[str, Any],
    video_dir: Path,
    video_name: str,
    unidrive_rows: list[dict[str, Any]] | None = None,
    physical_state: dict[str, Any] | None = None,
    persist_min_frames: int = _PERSIST_MIN_FRAMES,
) -> dict[str, Any]:
    """Aggregate per-primitive signals into one clip-level threat assessment."""
    t0 = time.time()

    primitives = list(threat_primitives_result.get("primitives") or [])
    if threat_primitives_result.get("skipped", True) or not primitives:
        result = _empty_result()
        result["skipped"] = True
        result["elapsed_sec"] = round(time.time() - t0, 3)
        _write_json(result, video_dir)
        _log.info("  [local threat] no primitives available — defaulting to continue")
        return result

    unidrive_rows = unidrive_rows or []
    physical_state = physical_state or {}

    active: list[dict[str, Any]] = []
    top_threats: list[dict[str, Any]] = []
    remaining_risk = 1.0

    for primitive in primitives:
        evidence = _primitive_evidence(primitive)
        support_frames = max(
            int(evidence["support_frames"]),
            int(evidence["temporal_persistence"]),
        )
        if support_frames < persist_min_frames:
            continue
        eff = _effective_score(primitive, persist_min_frames)
        if eff <= 0.0:
            continue
        enriched = {
            **primitive,
            "aggregated_score": round(eff, 4),
            "evidence": evidence,
            "contradiction_signals": contradiction_signals_for_threat(
                str(primitive.get("type", "")),
                primitive,
                unidrive_rows,
                physical_state,
            ),
        }
        active.append(enriched)
        remaining_risk *= 1.0 - eff

    active.sort(key=lambda p: float(p.get("aggregated_score", 0.0)), reverse=True)
    for primitive in active[:3]:
        top_threats.append(
            {
                "type": str(primitive.get("type", "")),
                "score": round(float(primitive.get("aggregated_score", 0.0)), 4),
                "evidence": dict(primitive.get("evidence") or {}),
            }
        )

    contradiction_summary = summarize_contradictions(
        threat_rows=active,
        contradiction_signals=list(threat_primitives_result.get("contradiction_signals") or []),
    )
    trust_penalty = float(contradiction_summary.get("trust_penalty", 0.0) or 0.0)
    mean_uncertainty = (
        sum(
            float((p.get("evidence") or {}).get("uncertainty", p.get("uncertainty", 0.0)) or 0.0)
            for p in active
        )
        / len(active)
        if active
        else 0.0
    )
    automation_confidence = max(0.0, min(1.0, 1.0 - mean_uncertainty - trust_penalty))

    local_threat_score = round(1.0 - remaining_risk, 4) if active else 0.0
    result = {
        "skipped": False,
        "local_threat_score": local_threat_score,
        "automation_confidence": round(automation_confidence, 4),
        "trust_penalty": round(trust_penalty, 4),
        "disagreement_count": int(contradiction_summary.get("disagreement_count", 0)),
        "disagreement_rate": float(contradiction_summary.get("disagreement_rate", 0.0)),
        "source_pair_conflicts": contradiction_summary.get("source_pair_conflicts", []),
        "top_threats": top_threats,
        "summary": {
            "persist_min_frames": int(persist_min_frames),
            "n_input_primitives": len(primitives),
            "n_active_primitives": len(active),
            "active_types": [str(p.get("type", "")) for p in active],
            "mean_uncertainty": round(mean_uncertainty, 4),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    _write_json(result, video_dir)
    _log.info(
        "  [ok] Local threat: score=%.2f  confidence=%.2f  active=%s",
        local_threat_score,
        automation_confidence,
        result["summary"]["active_types"] or "none",
    )
    return result


def _write_json(result: dict[str, Any], video_dir: Path) -> None:
    out = video_dir / "local_threat_assessment.json"
    try:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        _log.warning("local_threat: could not write JSON: %s", exc)
