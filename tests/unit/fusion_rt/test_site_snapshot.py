"""Combined camera + sensor snapshot from contract messages."""

from datetime import UTC, datetime
from types import SimpleNamespace

from selfsuvis.fusion_rt.site_snapshot import CombinedSiteSnapshot
from ss_contracts.models import SensorEvent, SensorState


async def test_ingest_sensor_state_and_event_and_camera() -> None:
    snap = CombinedSiteSnapshot(camera_window_sec=120)
    await snap.ingest_sensor_state(
        SensorState(
            dev_eui="70b3d57ed0060001",
            last_seen=datetime(2026, 9, 19, 8, 0, tzinfo=UTC),
            reading_count=3,
            temperature_c=21.4,
            motion=True,
        )
    )
    await snap.ingest_sensor_event(
        SensorEvent(
            event_kind="sensor",
            event_time=datetime(2026, 9, 19, 8, 1, tzinfo=UTC),
            ingest_time=datetime(2026, 9, 19, 8, 1, 1, tzinfo=UTC),
            node_id="70b3d57ed0060002",
            sensor_type="lorawan",
            sector_id="unknown",
            payload={"humidity_pct": 40.0},
            freshness_sec=0.0,
        )
    )
    await snap.ingest_camera_event(
        SimpleNamespace(
            event_id="e1",
            camera="entrance",
            label="person",
            score=0.8,
            started_at=datetime.now(UTC),
            has_snapshot=True,
        )
    )
    state = await snap.get_state()
    assert state.sensor_count == 2
    assert state.camera_count == 1
    assert state.active_motion is True
    assert {row.dev_eui for row in state.sensors} == {"70b3d57ed0060001", "70b3d57ed0060002"}
    assert state.cameras[0].camera == "entrance"
    assert "person" in state.cameras[0].active_labels
