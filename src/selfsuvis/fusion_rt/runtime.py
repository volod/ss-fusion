"""Start fusion-rt MQTT consumer, correlator, and webhook retry tasks."""

import asyncio
from typing import Any

from fastapi import FastAPI

from selfsuvis.fusion_rt.config import fusion_settings
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)


def _multi_callback(*cbs):
    """Return an async callback that fans out to all provided callbacks."""

    async def _cb(event) -> None:
        for cb in cbs:
            try:
                await cb(event)
            except Exception:
                logger.debug("fusion callback failed", exc_info=True)

    return _cb


def start_site_runtime(app: FastAPI) -> asyncio.Task | None:
    """Start the contract MQTT consumer, combined snapshot, and threat pipeline."""
    try:
        from selfsuvis.fusion_rt.aggregator import RealtimeThreatAggregator
        from selfsuvis.fusion_rt.event_persist import SiteEventPersister
        from selfsuvis.fusion_rt.mqtt_consumer import FusionContractConsumer
        from selfsuvis.fusion_rt.scene_synthesis import SceneSynthesizer
        from selfsuvis.fusion_rt.sensor_ingest import SensorEventIngestor
        from selfsuvis.fusion_rt.site_snapshot import CombinedSiteSnapshot

        snapshot = CombinedSiteSnapshot(camera_window_sec=fusion_settings.CAMERA_EVENT_WINDOW_SEC)
        threat_agg = RealtimeThreatAggregator()
        ingestor = SensorEventIngestor(threat_agg)
        synthesizer = SceneSynthesizer(aggregator=snapshot)
        persister = SiteEventPersister(getattr(app.state, "db_pool", None))

        app.state.site_snapshot = snapshot
        app.state.site_state_aggregator = snapshot
        app.state.site_threat_aggregator = threat_agg
        app.state.scene_synthesizer = synthesizer

        consumer = FusionContractConsumer(
            on_sensor_event=_multi_callback(
                snapshot.ingest_sensor_event,
                ingestor.on_sensor_event,
                persister.on_sensor_event,
            ),
            on_sensor_state=snapshot.ingest_sensor_state,
            on_camera_event=_multi_callback(
                snapshot.ingest_camera_event,
                ingestor.on_camera_event,
                persister.on_camera_event,
            ),
            on_scene_caption=synthesizer.ingest_caption,
        )
        task = asyncio.create_task(consumer.run(), name="fusion_mqtt")
        logger.info("fusion site runtime started (%s)", consumer.describe())
        return task
    except Exception as exc:
        logger.warning("fusion site runtime not started: %s", exc)
        return None


def start_correlator_tasks(app: FastAPI) -> tuple[asyncio.Task | None, asyncio.Task | None]:
    """Start correlator and webhook retry when CORRELATOR_ENABLED."""
    if not fusion_settings.CORRELATOR_ENABLED:
        return None, None
    try:
        from selfsuvis.fusion_rt.correlator import run_correlator
        from selfsuvis.fusion_rt.webhook_retry import run_webhook_retry

        correlator_task = asyncio.create_task(run_correlator(app), name="correlator")
        webhook_task = asyncio.create_task(run_webhook_retry(), name="webhook_retry")
        logger.info("Correlator and webhook retry tasks started")
        return correlator_task, webhook_task
    except Exception as exc:
        logger.warning("Correlator not started: %s", exc)
        return None, None


async def cancel_tasks(tasks: list[asyncio.Task | None]) -> None:
    running = [task for task in tasks if task is not None and not task.done()]
    for task in running:
        task.cancel()
    if running:
        await asyncio.gather(*running, return_exceptions=True)


def fanout(*cbs: Any):
    """Public alias used by tests."""
    return _multi_callback(*cbs)
