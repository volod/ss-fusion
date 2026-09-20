"""GET /api/v1/site/state — DB-backed site state snapshot."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from selfsuvis.fusion_rt.db import get_db_pool
from selfsuvis.fusion_rt.deps import require_api_key
from selfsuvis.fusion_rt.routers.v1.schemas import (
    IncidentResponse,
    SiteStateSnapshot,
    SiteStateZone,
)
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _row_to_incident(row) -> IncidentResponse:
    return IncidentResponse(
        incident_id=str(row["incident_id"]),
        ts=row["ts"].isoformat(),
        zone_id=row["zone_id"],
        modalities=list(row["modalities"]),
        confidence=row["confidence"],
        risk_level=row["risk_level"],
        summary_text=row["summary_text"],
        evidence_refs=row["evidence_refs"] if isinstance(row["evidence_refs"], list) else [],
        rule_id=row["rule_id"],
        acknowledged_at=row["acknowledged_at"].isoformat() if row["acknowledged_at"] else None,
        dismissed_at=row["dismissed_at"].isoformat() if row["dismissed_at"] else None,
        dismissal_reason=row["dismissal_reason"],
        created_at=row["created_at"].isoformat(),
    )


@router.get("/site/state", response_model=SiteStateSnapshot, summary="Site state snapshot")
async def get_site_state(request: Request) -> SiteStateSnapshot:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        zones = await conn.fetch("SELECT zone_id, label FROM zones ORDER BY zone_id")
        active_rows = await conn.fetch(
            """
            SELECT * FROM incidents
            WHERE acknowledged_at IS NULL AND dismissed_at IS NULL
            ORDER BY ts DESC
            """,
        )

    incidents_by_zone: dict[str, list] = {}
    for row in active_rows:
        incidents_by_zone.setdefault(row["zone_id"], []).append(_row_to_incident(row))

    zone_list = []
    for zone in zones:
        zid = zone["zone_id"]
        zone_incidents = incidents_by_zone.get(zid, [])
        if zone_incidents:
            best = max(zone_incidents, key=lambda i: _RISK_ORDER.get(i.risk_level, 0))
            risk_level: str | None = best.risk_level
        else:
            risk_level = None
        zone_list.append(
            SiteStateZone(
                zone_id=zid,
                label=zone["label"],
                risk_level=risk_level,
                active_incidents=zone_incidents,
            )
        )

    return SiteStateSnapshot(
        ts=datetime.now(timezone.utc).isoformat(),
        zones=zone_list,
    )
