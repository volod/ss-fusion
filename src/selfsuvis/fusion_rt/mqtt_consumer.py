"""Consume ss-common contract MQTT messages for fusion-rt."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from ss_contracts.models import CONTRACTS
from ss_kit.logging import get_logger
from ss_kit.mqtt import MqttSettings, TopicBuilder

from .config import fusion_settings

logger = get_logger(__name__)

OnMessage = Callable[[Any], Awaitable[None]]

_FUSION_CONTRACTS = (
    "sensor-event",
    "sensor-state",
    "camera-event",
    "scene-caption",
)


class FusionContractConsumer:
    """Subscribe to the topic-map contracts fusion-rt owns as a consumer."""

    def __init__(
        self,
        *,
        on_sensor_event: OnMessage | None = None,
        on_sensor_state: OnMessage | None = None,
        on_camera_event: OnMessage | None = None,
        on_scene_caption: OnMessage | None = None,
        mqtt: MqttSettings | None = None,
    ) -> None:
        self._handlers: dict[str, OnMessage | None] = {
            "sensor-event": on_sensor_event,
            "sensor-state": on_sensor_state,
            "camera-event": on_camera_event,
            "scene-caption": on_scene_caption,
        }
        self._mqtt = mqtt if mqtt is not None else fusion_settings.mqtt
        self._topics = TopicBuilder()

    def describe(self) -> str:
        return self._mqtt.describe()

    async def run(self, reconnect_interval: float = 5.0) -> None:
        try:
            import aiomqtt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "aiomqtt is required for the fusion MQTT consumer. "
                "Install it with: pip install aiomqtt"
            ) from exc

        import aiomqtt

        while True:
            try:
                async with aiomqtt.Client(**self._mqtt.client_kwargs()) as client:
                    logger.info("fusion MQTT connected to %s", self._mqtt.describe())
                    await self.consume(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fusion MQTT connection lost (%s), reconnecting in %ss",
                    exc,
                    reconnect_interval,
                )
                await asyncio.sleep(reconnect_interval)

    async def consume(self, client: Any) -> None:
        for contract_id in _FUSION_CONTRACTS:
            await client.subscribe(self._topics.subscription(contract_id))
        async for message in client.messages:
            await self.handle_message(str(message.topic), message.payload)

    async def handle_message(self, topic: str, payload: bytes | str) -> None:
        resolved = self._topics.resolve(topic)
        if resolved is None:
            return
        contract_id, _params = resolved
        await self._dispatch_contract(contract_id, payload)

    async def _dispatch_contract(self, contract_id: str, payload: bytes | str) -> None:
        handler = self._handlers.get(contract_id)
        if handler is None:
            return
        try:
            raw = json.loads(payload)
            model = CONTRACTS[contract_id].model_validate(raw)
        except Exception:
            logger.exception("Invalid %s payload", contract_id)
            return
        try:
            await handler(model)
        except Exception:
            logger.exception("Error dispatching %s", contract_id)
