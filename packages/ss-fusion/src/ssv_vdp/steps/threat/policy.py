"""Policy layer for mapping threat estimates into operational actions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..common import _log

_DEFAULT_MISSION_OBJECTIVE = "safe_progress"
_DEFAULT_RISK_TOLERANCE = "balanced"


def step_policy(
    local_threat_result: dict[str, Any],
    video_dir: Path,
    video_name: str,
    *,
    mission_objective: str = _DEFAULT_MISSION_OBJECTIVE,
    risk_tolerance: str = _DEFAULT_RISK_TOLERANCE,
    sensor_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map threat estimates into a fixed action vocabulary without changing the score."""
    t0 = time.time()
    sensor_health = sensor_health or {}

    if local_threat_result.get("skipped", True):
        result = {
            "skipped": True,
            "mission_objective": mission_objective,
            "risk_tolerance": risk_tolerance,
            "recommended_action": "continue",
            "policy_reason": "no active threat estimate",
            "elapsed_sec": round(time.time() - t0, 3),
        }
        _write_json(result, video_dir)
        return result

    score = float(local_threat_result.get("local_threat_score", 0.0) or 0.0)
    confidence = float(local_threat_result.get("automation_confidence", 1.0) or 1.0)
    trust_penalty = float(local_threat_result.get("trust_penalty", 0.0) or 0.0)
    disagreement_rate = float(local_threat_result.get("disagreement_rate", 0.0) or 0.0)
    conflicts = list(local_threat_result.get("source_pair_conflicts") or [])
    active_types = set((local_threat_result.get("summary") or {}).get("active_types") or [])

    missing_sensors = list(sensor_health.get("missing_sensors") or [])
    health_warnings = list(sensor_health.get("health_warnings") or [])
    degraded = bool(sensor_health.get("degraded", False))

    policy_reason = "nominal"
    if (
        degraded
        or health_warnings
        or confidence < 0.45
        or trust_penalty >= 0.30
        or disagreement_rate >= 0.34
    ):
        action = "inspect_sensor"
        policy_reason = "automation confidence reduced by contradiction or sensor health"
    elif score >= 0.80:
        action = "abort"
        policy_reason = "severe local threat"
    elif score >= 0.65:
        action = "reroute"
        policy_reason = "high local threat"
    elif score >= 0.35 or "track_anomaly" in active_types:
        action = "reduce_speed"
        policy_reason = "moderate local threat"
    else:
        action = "continue"

    if risk_tolerance == "conservative":
        if action == "continue" and score >= 0.25:
            action = "reduce_speed"
        elif action == "reduce_speed" and score >= 0.60:
            action = "reroute"
    elif (
        risk_tolerance == "aggressive"
        and action == "reduce_speed"
        and score < 0.45
        and not degraded
    ):
        action = "continue"

    result = {
        "skipped": False,
        "mission_objective": mission_objective,
        "risk_tolerance": risk_tolerance,
        "recommended_action": action,
        "policy_reason": policy_reason,
        "automation_confidence": confidence,
        "sensor_health": {
            "degraded": degraded,
            "missing_sensors": missing_sensors,
            "health_warnings": health_warnings,
        },
        "source_pair_conflicts": conflicts,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    _write_json(result, video_dir)
    _log.info(
        "  [ok] Policy: action=%s  confidence=%.2f  degraded=%s",
        action,
        confidence,
        degraded,
    )
    return result


def _write_json(result: dict[str, Any], video_dir: Path) -> None:
    out = video_dir / "policy_decision.json"
    try:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        _log.warning("policy: could not write JSON: %s", exc)
