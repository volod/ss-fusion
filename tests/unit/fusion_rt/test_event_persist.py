"""Map contract events onto site_events rows."""

from datetime import UTC, datetime

from selfsuvis.fusion_rt.event_persist import camera_event_to_site_row, sensor_event_to_site_row
from ss_contracts.models import CameraEvent, SensorEvent


def test_sensor_lorawan_maps_to_custom() -> None:
    event = SensorEvent(
        event_kind="sensor",
        event_time=datetime(2026, 9, 19, 8, 0, tzinfo=UTC),
        ingest_time=datetime(2026, 9, 19, 8, 0, 1, tzinfo=UTC),
        node_id="70b3d57ed0060001",
        sensor_type="lorawan",
        sector_id="grid:1:2",
        payload={"motion": True},
        freshness_sec=0.0,
    )
    row = sensor_event_to_site_row(event, zone_id="local")
    assert row["modality"] == "custom"
    assert row["zone_id"] == "local"
    assert row["sensor_id"] == "70b3d57ed0060001"
    assert row["confidence"] >= 0.7


def test_camera_event_maps_to_camera() -> None:
    event = CameraEvent(
        event_id="e1",
        camera="entrance",
        label="person",
        score=0.81,
        top_score=0.84,
        event_type="new",
        started_at=datetime(2026, 9, 19, 8, 0, tzinfo=UTC),
        has_snapshot=True,
        has_clip=False,
        region={},
        raw={},
    )
    row = camera_event_to_site_row(event, zone_id="local")
    assert row["modality"] == "camera"
    assert row["sensor_id"] == "entrance"
    assert row["confidence"] == 0.81
