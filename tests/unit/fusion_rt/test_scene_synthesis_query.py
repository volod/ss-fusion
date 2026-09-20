import asyncio

from selfsuvis.fusion_rt.scene_synthesis import SceneSynthesizer
from selfsuvis.fusion_rt.site_snapshot import CombinedSiteSnapshot
from ss_contracts.models import SceneCaption


def test_ingest_caption_feeds_synthesize_prompt() -> None:
    synthesizer = SceneSynthesizer(CombinedSiteSnapshot(), reasoning_url="")
    caption = SceneCaption(
        mission_id="mission-0917",
        frame_id="frame-000042",
        t_sec=12.5,
        caption="A red pickup truck parked beside the north gate.",
        created_at="2026-09-19T08:00:00.500000Z",
    )
    asyncio.run(synthesizer.ingest_caption(caption))
    result = asyncio.run(synthesizer.synthesize(force=True))
    assert "live captions" in " ".join(result.sources_used)
    assert "No LLM backend" in result.narrative
