from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

from ssv_vdp.agentic import processor  # noqa: E402


def test_unreadable_frame_returns_no_record(tmp_path):
    rec = SimpleNamespace(path=str(tmp_path / "missing.jpg"))
    ontology = {"nodes": []}

    record, returned_ontology, prev_segments, prev_tracks, next_id = (
        processor._process_frame_to_record(
            rec,
            video_name="clip",
            tagger=None,
            segmenter=None,
            ontology=ontology,
            prev_segments=None,
            prev_tracks={},
            next_track_id=3,
            base_metadata={},
        )
    )

    assert record is None
    assert returned_ontology is ontology
    assert (prev_segments, prev_tracks, next_id) == (None, {}, 3)


def test_tagger_description_receives_an_rgb_image():
    frame_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_bgr[..., 0] = 255  # pure blue in BGR order
    seen = {}

    class Tagger:
        def describe_image(self, image):
            seen["pixel"] = image.getpixel((0, 0))
            return {"labels": [{"label": "sky"}]}

    description, _segments = processor.image_to_text_agent(frame_bgr, tagger=Tagger())

    assert seen["pixel"] == (0, 0, 255)
    assert "Likely: sky." in description
