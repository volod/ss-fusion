"""FrigateAdapter — stub adapter that reads camera events from the Frigate API.

Full sensor integration is deferred to Phase 0 customer pairing.
This stub emits synthetic events sufficient to exercise the correlator in Phase 3A.
Uses COOP_FRIGATE_API_URL from coop_pilot.config (no duplicate env var added).
"""

import asyncio

import httpx

from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.fusion.adapters.base import SensorAdapter
from selfsuvis.pipeline.fusion.adapters.registry import registry

logger = get_logger(__name__)


class FrigateAdapter(SensorAdapter):
    modality = "camera"

    def __init__(self) -> None:
        super().__init__()
        try:
            from selfsuvis.coop.config import settings as coop_settings

            self._frigate_url = coop_settings.frigate_api_url
        except Exception:
            self._frigate_url = ""
        self.enabled = bool(self._frigate_url)

    async def _auto_seed_zones(self, pool) -> None:
        """Seed zones from Frigate camera names if zones table is empty."""
        if not self._frigate_url:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._frigate_url}/api/config")
                if resp.status_code != 200:
                    return
                cameras = resp.json().get("cameras", {})
            async with pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM zones")
                if count > 0:
                    return
                for cam_name in cameras:
                    await conn.execute(
                        "INSERT INTO zones (zone_id, label) VALUES ($1, $2) "
                        "ON CONFLICT (zone_id) DO NOTHING",
                        cam_name,
                        cam_name,
                    )
            logger.info("Auto-seeded %d zones from Frigate cameras", len(cameras))
        except Exception as exc:
            logger.warning("Zone auto-seed from Frigate failed: %s", exc)

    async def start(self) -> None:
        if not self.enabled:
            logger.debug("FrigateAdapter disabled (COOP_FRIGATE_API_URL not set)")
            return
        logger.info("FrigateAdapter started (stub)")
        while True:
            await asyncio.sleep(30)


registry.register("frigate", FrigateAdapter())
