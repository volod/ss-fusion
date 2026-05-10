"""SensorAdapter ABC — base class for all sensor adapters."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class SensorAdapter(ABC):
    """Base class for sensor adapters that emit EventEnvelope events to the ingest API.

    Subclasses implement `start()` as an infinite async loop.
    The adapter posts events to POST /api/v1/events/{modality} using an internal
    httpx client (URL from API_URL env or http://localhost:8000).

    Adapters are disabled by returning early from `start()` when their required
    config (model path, watch dir, etc.) is not set.
    """

    modality: str = "custom"

    def __init__(self) -> None:
        self.last_event_ts: str | None = None
        self.enabled: bool = True

    def _record_event(self) -> None:
        self.last_event_ts = datetime.now(timezone.utc).isoformat()

    @abstractmethod
    async def start(self) -> None:
        """Run the adapter event loop. Must be cancellation-safe."""
        ...
