"""Zone CRUD + zone history endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from selfsuvis.fusion_rt.db import get_db_pool
from selfsuvis.fusion_rt.deps import require_api_key
from selfsuvis.fusion_rt.routers.v1.schemas import (
    IncidentResponse,
    ZoneCreate,
    ZoneResponse,
    ZoneUpdate,
)
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


def _zone_row_to_model(row) -> ZoneResponse:
    return ZoneResponse(
        zone_id=row["zone_id"],
        label=row["label"],
        description=row["description"],
        map_x=row["map_x"],
        map_y=row["map_y"],
        map_w=row["map_w"],
        map_h=row["map_h"],
        created_at=row["created_at"].isoformat(),
    )


def _incident_row_to_model(row) -> IncidentResponse:

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


@router.get("/zones", summary="List zones")
async def list_zones(request: Request) -> dict:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM zones ORDER BY created_at DESC")
    return {"zones": [_zone_row_to_model(r).model_dump() for r in rows]}


@router.post("/zones", status_code=201, summary="Create zone")
async def create_zone(request: Request, body: ZoneCreate) -> ZoneResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        existing = await conn.fetchval("SELECT zone_id FROM zones WHERE zone_id = $1", body.zone_id)
        if existing:
            raise HTTPException(status_code=409, detail="Zone already exists")
        row = await conn.fetchrow(
            """
            INSERT INTO zones (zone_id, label, description, map_x, map_y, map_w, map_h)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            body.zone_id,
            body.label,
            body.description,
            body.map_x,
            body.map_y,
            body.map_w,
            body.map_h,
        )
    return _zone_row_to_model(row)


@router.put("/zones/{zone_id}", summary="Update zone")
async def update_zone(request: Request, zone_id: str, body: ZoneUpdate) -> ZoneResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT * FROM zones WHERE zone_id = $1", zone_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Zone not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return _zone_row_to_model(existing)

        set_clauses = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates))
        values = [zone_id] + list(updates.values())
        row = await conn.fetchrow(
            f"UPDATE zones SET {set_clauses} WHERE zone_id = $1 RETURNING *",
            *values,
        )
    return _zone_row_to_model(row)


@router.delete("/zones/{zone_id}", status_code=204, summary="Delete zone")
async def delete_zone(request: Request, zone_id: str) -> None:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM zones WHERE zone_id = $1", zone_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Zone not found")


@router.get("/zones/{zone_id}/history", summary="Zone incident history")
async def zone_history(
    request: Request,
    zone_id: str,
    since: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        zone = await conn.fetchval("SELECT zone_id FROM zones WHERE zone_id = $1", zone_id)
        if not zone:
            raise HTTPException(status_code=404, detail="Zone not found")

        if since:
            rows = await conn.fetch(
                """
                SELECT * FROM incidents
                WHERE zone_id = $1 AND ts >= $2
                ORDER BY ts DESC LIMIT $3
                """,
                zone_id,
                since,
                limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM incidents WHERE zone_id = $1 ORDER BY ts DESC LIMIT $2",
                zone_id,
                limit,
            )

    return {
        "zone_id": zone_id,
        "incidents": [_incident_row_to_model(r).model_dump() for r in rows],
    }
