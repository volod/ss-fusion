"""Adapter registry — register and retrieve sensor adapters by name."""

from selfsuvis.pipeline.fusion.adapters.base import SensorAdapter


class _AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SensorAdapter] = {}

    def register(self, name: str, adapter: SensorAdapter) -> None:
        self._adapters[name] = adapter

    def get(self, name: str) -> SensorAdapter | None:
        return self._adapters.get(name)

    def all(self) -> dict[str, SensorAdapter]:
        return dict(self._adapters)


registry = _AdapterRegistry()
