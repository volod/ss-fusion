"""Correlator — background task that matches sensor events to fusion rules and creates incidents.

Design:
- Batched DB poll every 5s: one SELECT fetches all events in the max window across all rules.
- Python-side grouping by (rule_id, zone_id) checks modality presence and computes confidence.
- Dedup: skip INSERT if an incident already exists in the rule's window for that zone.
- On match: INSERT incident, emit to SSE subscribers, LPUSH to fusion:alert:retry.
- Correlator heartbeat: SET fusion:correlator:heartbeat <ISO> EX 60 on every tick.
- Restart catch-up: run the full query once immediately before entering the poll loop.
- Error resilience: try/except around the entire poll body.
- Single-process only: correlator uses in-process SSE subscribers dict; multi-worker
  deployment causes duplicate incidents (see docs/operations.md).
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import redis.asyncio as aioredis

from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.fusion.utils import probability_union

logger = get_logger(__name__)

_POLL_INTERVAL_S = 5.0
_SEED_YAML = Path(__file__).parents[4] / "docs" / "seed" / "fusion_rules.yaml"


def _risk_level(confidence: float) -> str:
    if confidence < 0.4:
        return "low"
    if confidence < 0.7:
        return "medium"
    if confidence < 0.9:
        return "high"
    return "critical"


async def _seed_rules(conn) -> None:
    """Seed fusion_rules from YAML if table is empty and YAML file exists."""
    count = await conn.fetchval("SELECT COUNT(*) FROM fusion_rules")
    if count > 0:
        return
    if not _SEED_YAML.exists():
        logger.debug("Correlator: no seed YAML at %s, idling with no rules", _SEED_YAML)
        return

    import yaml

    try:
        data = yaml.safe_load(_SEED_YAML.read_text())
        rules = data.get("rules", []) if data else []
        for rule in rules:
            await conn.execute(
                """
                INSERT INTO fusion_rules
                    (rule_id, label, modalities, zone_id, window_s, min_confidence, enabled)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (rule_id) DO NOTHING
                """,
                rule["rule_id"],
                rule["label"],
                rule.get("modalities", []),
                rule.get("zone_id"),
                rule.get("window_s", 30),
                rule.get("min_confidence", 0.5),
                rule.get("enabled", True),
            )
        logger.info("Correlator: seeded %d rules from %s", len(rules), _SEED_YAML)
    except Exception as exc:
        logger.warning("Correlator: YAML seed failed: %s", exc)


async def _poll(pool, redis_client, sse_subscribers: dict) -> None:
    async with pool.acquire() as conn:
        rules = await conn.fetch("SELECT * FROM fusion_rules WHERE enabled = TRUE")
        if not rules:
            return

        max_window_s = max(r["window_s"] for r in rules)
        rows = await conn.fetch(
            """
            SELECT event_id, zone_id, modality, confidence, ts
            FROM site_events
            WHERE ts > NOW() - ($1 || ' seconds')::INTERVAL
            """,
            str(max_window_s),
        )

    events_by_zone_modality: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["zone_id"], row["modality"])
        events_by_zone_modality.setdefault(key, []).append(
            {
                "event_id": str(row["event_id"]),
                "modality": row["modality"],
                "confidence": row["confidence"],
                "ts": row["ts"],
            }
        )

    all_zones: set[str] = {row["zone_id"] for row in rows}

    for rule in rules:
        rule_id = rule["rule_id"]
        required_modalities = list(rule["modalities"])
        window_s = rule["window_s"]
        rule_zone = rule["zone_id"]

        check_zones = {rule_zone} if rule_zone else all_zones

        for zone_id in check_zones:
            modality_events: dict[str, list[dict]] = {}
            for mod in required_modalities:
                evts = events_by_zone_modality.get((zone_id, mod), [])
                if evts:
                    modality_events[mod] = evts

            if set(modality_events.keys()) != set(required_modalities):
                continue

            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    """
                    SELECT incident_id FROM incidents
                    WHERE rule_id = $1 AND zone_id = $2
                      AND ts > NOW() - ($3 || ' seconds')::INTERVAL
                    LIMIT 1
                    """,
                    rule_id,
                    zone_id,
                    str(window_s),
                )
                if existing:
                    continue

                all_events = [e for evts in modality_events.values() for e in evts]
                scores = [e["confidence"] for e in all_events]
                confidence = probability_union(scores)

                if confidence < rule["min_confidence"]:
                    continue

                risk = _risk_level(confidence)
                modalities_sorted = sorted(required_modalities)
                summary = f"{rule['label']} — {', '.join(modalities_sorted)} in {zone_id} ({risk})"
                evidence_refs = [
                    {
                        "event_id": e["event_id"],
                        "modality": e["modality"],
                        "confidence": e["confidence"],
                    }
                    for e in all_events
                ]

                row = await conn.fetchrow(
                    """
                    INSERT INTO incidents
                        (ts, zone_id, modalities, confidence, risk_level,
                         summary_text, evidence_refs, rule_id)
                    VALUES (NOW(), $1, $2, $3, $4, $5, $6::jsonb, $7)
                    RETURNING *
                    """,
                    zone_id,
                    modalities_sorted,
                    confidence,
                    risk,
                    summary,
                    json.dumps(evidence_refs),
                    rule_id,
                )

            incident_id = str(row["incident_id"])
            logger.info(
                "Correlator: incident %s created (rule=%s zone=%s risk=%s conf=%.3f)",
                incident_id,
                rule_id,
                zone_id,
                risk,
                confidence,
            )

            # Emit to SSE subscribers
            event_payload_site = json.dumps(
                {
                    "event_id": evidence_refs[0]["event_id"] if evidence_refs else None,
                    "ts": row["ts"].isoformat(),
                    "zone_id": zone_id,
                    "modality": modalities_sorted[0] if modalities_sorted else "custom",
                    "confidence": confidence,
                    "incident_id": incident_id,
                }
            )
            event_payload_incident = json.dumps(
                {
                    "incident_id": incident_id,
                    "ts": row["ts"].isoformat(),
                    "zone_id": zone_id,
                    "risk_level": risk,
                }
            )
            for q in list(sse_subscribers.values()):
                try:
                    q.put_nowait({"event": "site_event", "data": event_payload_site})
                    q.put_nowait({"event": "incident_created", "data": event_payload_incident})
                except asyncio.QueueFull:
                    pass

            # LPUSH to webhook retry queue
            alert = json.dumps(
                {
                    "incident_id": incident_id,
                    "ts": row["ts"].isoformat(),
                    "zone_id": zone_id,
                    "modalities": modalities_sorted,
                    "confidence": confidence,
                    "risk_level": risk,
                    "summary_text": summary,
                    "rule_id": rule_id,
                    "attempt": 0,
                }
            )
            try:
                await redis_client.lpush("fusion:alert:retry", alert)
            except Exception as exc:
                logger.error("Correlator: LPUSH to retry queue failed: %s", exc)


async def run_correlator(app) -> None:
    """Main correlator loop. Called as asyncio.create_task from app lifespan."""
    pool = app.state.db_pool
    sse_subscribers: dict = app.state.sse_subscribers

    redis_client = aioredis.from_url(settings.CORRELATOR_REDIS_URL)

    async with pool.acquire() as conn:
        await _seed_rules(conn)

    # Startup catch-up
    try:
        await _poll(pool, redis_client, sse_subscribers)
    except Exception as exc:
        logger.error("Correlator: startup catch-up failed: %s", exc)

    while True:
        try:
            await _poll(pool, redis_client, sse_subscribers)
            await redis_client.set(
                "fusion:correlator:heartbeat",
                datetime.now(timezone.utc).isoformat(),
                ex=60,
            )
        except Exception as exc:
            logger.error("Correlator: poll error: %s", exc)
        await asyncio.sleep(_POLL_INTERVAL_S)
