"""Incident management endpoints — list, detail, ack, dismiss, notes, search, export."""

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from selfsuvis.fusion_rt.db import get_db_pool
from selfsuvis.fusion_rt.deps import require_api_key
from selfsuvis.fusion_rt.routers.v1.schemas import (
    DismissBody,
    IncidentResponse,
    NoteCreate,
    NoteResponse,
)
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


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


def _status_clause(status: str) -> str:
    if status == "active":
        return "acknowledged_at IS NULL AND dismissed_at IS NULL"
    if status == "acknowledged":
        return "acknowledged_at IS NOT NULL AND dismissed_at IS NULL"
    if status == "dismissed":
        return "dismissed_at IS NOT NULL"
    return "TRUE"


@router.get("/incidents", summary="List incidents")
async def list_incidents(
    request: Request,
    status: str = Query(default="active"),
    zone: str | None = Query(default=None),
    since: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=10000),
) -> dict:
    pool = get_db_pool(request)

    if zone:
        async with pool.acquire() as conn:
            exists = await conn.fetchval("SELECT zone_id FROM zones WHERE zone_id = $1", zone)
        if not exists:
            raise HTTPException(status_code=404, detail="Zone not found")

    clause = _status_clause(status)
    params: list = []
    extra = ""

    if zone:
        params.append(zone)
        extra += f" AND zone_id = ${len(params)}"
    if since:
        params.append(since)
        extra += f" AND ts >= ${len(params)}"

    params.append(limit)
    q = f"SELECT * FROM incidents WHERE {clause}{extra} ORDER BY ts DESC LIMIT ${len(params)}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(q, *params)

    return {"incidents": [_row_to_incident(r).model_dump() for r in rows]}


@router.get("/incidents/search", summary="Full-text search incidents")
async def search_incidents(
    request: Request,
    q: str = Query(..., min_length=1),
    zone: str | None = Query(default=None),
    since: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict:
    pool = get_db_pool(request)
    params: list = [q]
    extra = ""

    if zone:
        params.append(zone)
        extra += f" AND i.zone_id = ${len(params)}"
    if since:
        params.append(since)
        extra += f" AND i.ts >= ${len(params)}"

    params.append(limit)
    sql = f"""
        SELECT DISTINCT i.*
        FROM incidents i
        LEFT JOIN incident_notes n ON n.incident_id = i.incident_id
        WHERE (
            to_tsvector('english', COALESCE(i.summary_text, '')) @@ plainto_tsquery('english', $1)
            OR to_tsvector('english', COALESCE(n.body, '')) @@ plainto_tsquery('english', $1)
        ){extra}
        ORDER BY i.ts DESC
        LIMIT ${len(params)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return {"incidents": [_row_to_incident(r).model_dump() for r in rows]}


@router.get("/incidents/export", summary="Export incidents")
async def export_incidents(
    request: Request,
    format: str = Query(default="json"),
    since: str | None = Query(default=None),
    zone: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> StreamingResponse:
    pool = get_db_pool(request)
    params: list = []
    extra = ""

    if zone:
        params.append(zone)
        extra += f" AND zone_id = ${len(params)}"
    if since:
        params.append(since)
        extra += f" AND ts >= ${len(params)}"

    params.append(limit)
    sql = f"SELECT * FROM incidents WHERE TRUE{extra} ORDER BY ts DESC LIMIT ${len(params)}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    incidents = [_row_to_incident(r) for r in rows]

    if format == "csv":

        def _csv_gen():
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                [
                    "incident_id",
                    "ts",
                    "zone_id",
                    "modalities",
                    "confidence",
                    "risk_level",
                    "summary_text",
                    "rule_id",
                    "acknowledged_at",
                    "dismissed_at",
                    "dismissal_reason",
                    "created_at",
                ]
            )
            yield buf.getvalue()
            for inc in incidents:
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(
                    [
                        inc.incident_id,
                        inc.ts,
                        inc.zone_id,
                        ";".join(inc.modalities),
                        inc.confidence,
                        inc.risk_level,
                        inc.summary_text or "",
                        inc.rule_id or "",
                        inc.acknowledged_at or "",
                        inc.dismissed_at or "",
                        inc.dismissal_reason or "",
                        inc.created_at,
                    ]
                )
                yield buf.getvalue()

        return StreamingResponse(
            _csv_gen(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=incidents.csv"},
        )

    def _json_gen():
        yield '{"incidents":['
        for i, inc in enumerate(incidents):
            if i > 0:
                yield ","
            yield inc.model_dump_json()
        yield "]}"

    return StreamingResponse(_json_gen(), media_type="application/json")


@router.get("/incidents/{incident_id}", summary="Get incident")
async def get_incident(request: Request, incident_id: str) -> IncidentResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM incidents WHERE incident_id = $1", incident_id)
    if not row:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _row_to_incident(row)


@router.post("/incidents/{incident_id}/acknowledge", summary="Acknowledge incident")
async def acknowledge_incident(request: Request, incident_id: str) -> IncidentResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE incidents SET acknowledged_at = NOW()
            WHERE incident_id = $1
            RETURNING *
            """,
            incident_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _row_to_incident(row)


@router.post("/incidents/{incident_id}/dismiss", summary="Dismiss incident")
async def dismiss_incident(
    request: Request, incident_id: str, body: DismissBody
) -> IncidentResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE incidents
            SET dismissed_at = NOW(), dismissal_reason = $2
            WHERE incident_id = $1
            RETURNING *
            """,
            incident_id,
            body.reason,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _row_to_incident(row)


@router.post("/incidents/{incident_id}/notes", status_code=201, summary="Add note")
async def add_note(request: Request, incident_id: str, body: NoteCreate) -> NoteResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT incident_id FROM incidents WHERE incident_id = $1", incident_id
        )
        if not exists:
            raise HTTPException(status_code=404, detail="Incident not found")
        row = await conn.fetchrow(
            """
            INSERT INTO incident_notes (incident_id, body, operator_id)
            VALUES ($1, $2, $3)
            RETURNING *
            """,
            incident_id,
            body.body,
            body.operator_id,
        )
    return NoteResponse(
        note_id=str(row["note_id"]),
        incident_id=str(row["incident_id"]),
        body=row["body"],
        operator_id=row["operator_id"],
        created_at=row["created_at"].isoformat(),
    )


@router.get("/incidents/{incident_id}/notes", summary="List notes")
async def list_notes(request: Request, incident_id: str) -> dict:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT incident_id FROM incidents WHERE incident_id = $1", incident_id
        )
        if not exists:
            raise HTTPException(status_code=404, detail="Incident not found")
        rows = await conn.fetch(
            "SELECT * FROM incident_notes WHERE incident_id = $1 ORDER BY created_at ASC",
            incident_id,
        )
    notes = [
        NoteResponse(
            note_id=str(r["note_id"]),
            incident_id=str(r["incident_id"]),
            body=r["body"],
            operator_id=r["operator_id"],
            created_at=r["created_at"].isoformat(),
        ).model_dump()
        for r in rows
    ]
    return {"notes": notes}
