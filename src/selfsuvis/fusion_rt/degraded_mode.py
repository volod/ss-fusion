"""Realtime degraded-mode and automation-confidence policies."""

from collections.abc import Iterable, Sequence
from typing import Any

from selfsuvis.pipeline.core.freshness import downweight_score

from .event_access import (
    event_freshness_sec,
    event_node_id,
    event_payload,
    event_sensor_type,
    payload_float,
    payload_text,
)

_HIGH_VALUE_SENSORS = ("camera", "gps", "imu", "fusion")


def evaluate_degraded_mode(
    sensor_events: Sequence[dict[str, Any]],
    node_health_events: Sequence[dict[str, Any]],
    *,
    base_automation_confidence: float,
    required_sensors: Iterable[str] = _HIGH_VALUE_SENSORS,
    outage_warning_sec: float = 30.0,
) -> dict[str, Any]:
    required = [str(sensor).strip().lower() for sensor in required_sensors if str(sensor).strip()]
    freshest_by_sensor: dict[str, float] = {}
    warnings: list[str] = []

    for event in sensor_events:
        sensor = event_sensor_type(event)
        freshness_sec = event_freshness_sec(event)
        if sensor not in freshest_by_sensor or freshness_sec < freshest_by_sensor[sensor]:
            freshest_by_sensor[sensor] = freshness_sec

    missing = [sensor for sensor in required if sensor not in freshest_by_sensor]
    stale = [
        sensor
        for sensor, age in freshest_by_sensor.items()
        if sensor in required and age > outage_warning_sec
    ]

    penalty = 0.0
    for sensor in missing:
        penalty += 0.12
        warnings.append(f"missing high-value sensor: {sensor}")
    for sensor in stale:
        penalty += 0.08
        warnings.append(f"stale high-value sensor: {sensor}")

    max_outage_sec = 0.0
    for event in node_health_events:
        payload = event_payload(event)
        outage_sec = payload_float(payload, "outage_sec")
        if outage_sec > max_outage_sec:
            max_outage_sec = outage_sec
        if outage_sec >= outage_warning_sec:
            warnings.append(
                f"model outage on node {event_node_id(event)}: {payload_text(payload, 'model_name', 'unknown')} for {outage_sec:.1f}s"
            )
            penalty += 0.10

    automation_confidence = max(0.0, min(1.0, float(base_automation_confidence) - penalty))
    return {
        "automation_confidence": round(automation_confidence, 4),
        "degraded": bool(missing or stale or max_outage_sec >= outage_warning_sec),
        "missing_sensors": missing,
        "stale_sensors": stale,
        "health_warnings": warnings,
    }


def apply_degraded_mode_to_threat(
    threat_score: float,
    automation_confidence: float,
    *,
    trust_penalty: float = 0.0,
) -> dict[str, float]:
    confidence = max(0.0, min(1.0, float(automation_confidence) - float(trust_penalty or 0.0)))
    weighted_threat = downweight_score(
        threat_score,
        freshness_sec=max(0.0, (1.0 - confidence) * 60.0),
        soft_limit_sec=20.0,
        hard_expiry_sec=120.0,
    )
    return {
        "automation_confidence": round(confidence, 4),
        "weighted_threat_score": round(weighted_threat, 4),
    }
