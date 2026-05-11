"""Webhook retry worker — async task that delivers incident alerts with exponential backoff.

Backoff: [0, 5, 30][min(attempt, 2)] seconds.
First attempt (attempt=0) is immediate (sleep=0).
After 3 failures: LPUSH to fusion:alert:dlq + log ERROR.
HMAC-SHA256 signature sent as X-SelfSuvis-Signature when WEBHOOK_SECRET is set.
"""

import asyncio
import hashlib
import hmac
import json

import httpx

from selfsuvis.pipeline.core import get_logger, settings

logger = get_logger(__name__)

_BACKOFF = [0, 5, 30]
_MAX_ATTEMPTS = 3
_BRPOP_TIMEOUT = 5


async def run_webhook_retry() -> None:
    """Main retry worker loop. Called as asyncio.create_task from app lifespan."""
    try:
        import redis.asyncio as aioredis  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "Webhook retry requires the 'redis' package. Install with: pip install redis"
        ) from exc

    redis_client = aioredis.from_url(settings.WEBHOOK_REDIS_URL)

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                result = await redis_client.brpop("fusion:alert:retry", timeout=_BRPOP_TIMEOUT)
                if result is None:
                    continue

                _, raw = result
                payload: dict = json.loads(raw)
                attempt: int = payload.get("attempt", 0)

                sleep_s = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                url = settings.WEBHOOK_ALERT_URL
                if not url:
                    continue

                body_dict = {k: v for k, v in payload.items() if k != "attempt"}
                body_bytes = json.dumps(body_dict).encode()

                headers = {
                    "Content-Type": "application/json",
                    "X-SelfSuvis-Version": "1",
                }
                if settings.WEBHOOK_SECRET:
                    sig = hmac.new(
                        settings.WEBHOOK_SECRET.encode(),
                        msg=body_bytes,
                        digestmod=hashlib.sha256,
                    ).hexdigest()
                    headers["X-SelfSuvis-Signature"] = f"sha256={sig}"

                try:
                    resp = await client.post(url, content=body_bytes, headers=headers)
                    resp.raise_for_status()
                    logger.debug(
                        "Webhook: delivered incident %s (attempt %d)",
                        payload.get("incident_id"),
                        attempt,
                    )
                except Exception as exc:
                    next_attempt = attempt + 1
                    if next_attempt >= _MAX_ATTEMPTS:
                        payload["attempt"] = next_attempt
                        await redis_client.lpush("fusion:alert:dlq", json.dumps(payload))
                        logger.error(
                            "Webhook: incident %s moved to DLQ after %d attempts: %s",
                            payload.get("incident_id"),
                            next_attempt,
                            exc,
                        )
                    else:
                        payload["attempt"] = next_attempt
                        await redis_client.lpush("fusion:alert:retry", json.dumps(payload))
                        logger.warning(
                            "Webhook: delivery failed (attempt %d), retrying: %s",
                            attempt,
                            exc,
                        )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Webhook retry worker error: %s", exc)
                await asyncio.sleep(1)
