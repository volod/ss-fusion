"""POST /api/v1/events/{modality} — sensor event ingest."""

from fastapi import APIRouter, Depends, HTTPException, Path, Request

from selfsuvis.fusion_rt.db import get_db_pool
from selfsuvis.fusion_rt.deps import require_sensor_key, sensor_rate_limit
from selfsuvis.fusion_rt.routers.v1.schemas import EventEnvelope, SiteEventResponse
from selfsuvis.pipeline.core import get_logger, resolve_allowed_path
from selfsuvis.pipeline.storage.common import decoded_json, jsonb

logger = get_logger(__name__)

router = APIRouter()

VALID_MODALITIES = {"camera", "audio", "rf", "thermal", "vibration", "custom"}


@router.post(
    "/events/{modality}",
    response_model=SiteEventResponse,
    dependencies=[Depends(require_sensor_key)],
)
async def ingest_event(
    request: Request,
    modality: str = Path(...),
    body: EventEnvelope = ...,
) -> SiteEventResponse:
    if modality not in VALID_MODALITIES:
        raise HTTPException(status_code=422, detail=f"Unknown modality: {modality!r}")

    if body.artifact_uri:
        try:
            resolve_allowed_path(body.artifact_uri)
        except (ValueError, PermissionError):
            raise HTTPException(
                status_code=422,
                detail="artifact_uri not in allowed paths",
            )

    sensor_rate_limit(body.sensor_id)

    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO site_events
                (ts, zone_id, sensor_id, modality, confidence, payload, artifact_uri)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
            RETURNING event_id, ts, zone_id, sensor_id, modality, confidence,
                      payload, artifact_uri, created_at
            """,
            body.ts,
            body.zone_id,
            body.sensor_id,
            modality,
            body.confidence,
            jsonb(body.payload, default={}),
            body.artifact_uri,
        )

    return SiteEventResponse(
        event_id=str(row["event_id"]),
        ts=row["ts"].isoformat(),
        zone_id=row["zone_id"],
        sensor_id=row["sensor_id"],
        modality=row["modality"],
        confidence=row["confidence"],
        payload=decoded_json(row["payload"], default={}) or {},
        artifact_uri=row["artifact_uri"],
        created_at=row["created_at"].isoformat(),
    )
