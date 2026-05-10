"""RFAnalyzerAdapter — stub for RF signal analysis events.

Full integration deferred to customer pairing (Phase 0 deliverable).
"""

import asyncio

from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.fusion.adapters.base import SensorAdapter
from selfsuvis.pipeline.fusion.adapters.registry import registry

logger = get_logger(__name__)


class RFAnalyzerAdapter(SensorAdapter):
    modality = "rf"

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False

    async def start(self) -> None:
        if not self.enabled:
            logger.debug("RFAnalyzerAdapter disabled (stub)")
            return
        while True:
            await asyncio.sleep(30)


registry.register("rf_analyzer", RFAnalyzerAdapter())
