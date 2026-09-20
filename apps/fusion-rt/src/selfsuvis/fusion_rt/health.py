"""GET /health for fusion-rt — correlator heartbeat, SSE, DLQ, postgres, redis, adapters."""

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from selfsuvis.fusion_rt.config import fusion_settings
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    """Health check. Reports correlator heartbeat, SSE subscribers, DLQ depth, and DB."""
    status = "ok"
    details: dict = {}

    pool = getattr(request.app.state, "db_pool", None)
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            details["postgres"] = "ok"
        except Exception as exc:
            logger.warning("Health: postgres error: %s", exc)
            details["postgres"] = "error"
            status = "down"
    else:
        details["postgres"] = "unconfigured"

    details["redis"] = "unconfigured"
    details["correlator_heartbeat_age_s"] = None
    details["dlq_depth"] = 0

    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(fusion_settings.HEALTH_REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        details["redis"] = "ok"

        hb = await r.get("fusion:correlator:heartbeat")
        if hb:
            try:
                hb_ts = datetime.fromisoformat(hb.decode())
                age = (datetime.now(UTC) - hb_ts).total_seconds()
                details["correlator_heartbeat_age_s"] = int(age)
                if age > 30:
                    status = max_status(status, "degraded")
            except Exception:
                details["correlator_heartbeat_age_s"] = None
        else:
            if fusion_settings.CORRELATOR_ENABLED:
                details["correlator_heartbeat_age_s"] = ">60"
                status = max_status(status, "degraded")

        dlq = await r.llen("fusion:alert:dlq")
        details["dlq_depth"] = dlq
        if dlq > 0:
            status = max_status(status, "degraded")

        await r.aclose()
    except Exception as exc:
        logger.warning("Health: redis error: %s", exc)
        details["redis"] = "error"
        status = "down"

    details["sse_subscribers"] = len(getattr(request.app.state, "sse_subscribers", {}))
    details["adapters"] = _get_adapter_status()

    response_body = {"status": status, **details}
    http_status = 200 if status != "down" else 503
    return JSONResponse(response_body, status_code=http_status)


def max_status(current: str, candidate: str) -> str:
    order = {"ok": 0, "degraded": 1, "down": 2}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current


def _get_adapter_status() -> dict:
    try:
        from selfsuvis.fusion_rt.adapters.registry import registry

        out = {}
        for name, adapter in registry.all().items():
            out[name] = {
                "status": "ok" if adapter.enabled else "disabled",
                "last_event_ts": getattr(adapter, "last_event_ts", None),
            }
        return out
    except Exception:
        drone_audio_enabled = bool(
            fusion_settings.DRONE_AUDIO_MODEL_PATH and fusion_settings.DRONE_AUDIO_WATCH_DIR
        )
        return {
            "drone_audio": {
                "status": "ok" if drone_audio_enabled else "disabled",
                "last_event_ts": None,
            }
        }
