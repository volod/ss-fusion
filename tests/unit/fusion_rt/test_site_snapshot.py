"""Combined camera + sensor snapshot from contract messages."""

from datetime import datetime, timezone

from selfsuvis.fusion_rt.site_snapshot import CombinedSiteSnapshot
from selfsuvis.pipeline.realtime.camera_events import CameraEvent
from ss_contracts.models import SensorEvent, SensorState


async def test_ingest_sensor_state_and_event_and_camera() -> None:
    snap = CombinedSiteSnapshot(camera_window_sec=120)
    await snap.ingest_sensor_state(
        SensorState(
            dev_eui="70b3d57ed0060001",
            last_seen=datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc),
            reading_count=3,
            temperature_c=21.4,
            motion=True,
        )
    )
    await snap.ingest_sensor_event(
        SensorEvent(
            event_kind="sensor",
            event_time=datetime(2026, 9, 19, 8, 1, tzinfo=timezone.utc),
            ingest_time=datetime(2026, 9, 19, 8, 1, 1, tzinfo=timezone.utc),
            node_id="70b3d57ed0060002",
            sensor_type="lorawan",
            sector_id="unknown",
            payload={"humidity_pct": 40.0},
            freshness_sec=0.0,
        )
    )
    await snap.ingest_camera_event(
        CameraEvent(
            event_id="e1",
            camera="entrance",
            label="person",
            score=0.8,
            top_score=0.8,
            event_type="new",
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            has_snapshot=True,
            has_clip=False,
            region={},
            raw={},
        )
    )
    state = await snap.get_state()
    assert state.sensor_count == 2
    assert state.camera_count == 1
    assert state.active_motion is True
    assert {row.dev_eui for row in state.sensors} == {"70b3d57ed0060001", "70b3d57ed0060002"}
    assert state.cameras[0].camera == "entrance"
    assert "person" in state.cameras[0].active_labels
