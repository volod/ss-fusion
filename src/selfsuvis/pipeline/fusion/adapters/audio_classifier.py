"""AudioClassifierAdapter — stub for acoustic sensor event detection.

Full integration deferred to customer pairing (Phase 0 deliverable).
Emits synthetic audio events for correlator testing.
"""

import asyncio

from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.fusion.adapters.base import SensorAdapter
from selfsuvis.pipeline.fusion.adapters.registry import registry

logger = get_logger(__name__)


class AudioClassifierAdapter(SensorAdapter):
    modality = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False

    async def start(self) -> None:
        if not self.enabled:
            logger.debug("AudioClassifierAdapter disabled (stub)")
            return
        while True:
            await asyncio.sleep(30)


registry.register("audio_classifier", AudioClassifierAdapter())
