"""Unit tests for Gemma 4 multi-frame analysis features:
- step_gemma_segment_captions (boundary detection + diff descriptions)
- _gemma_diff_two_frames_via_api (two-image sidecar call)
- _step_qwen_captioning_gemma_fallback (Qwen fallback via Gemma)
- write_gemma_segment_captions_md (report writing)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Stubs needed before loading steps_caption ────────────────────────────────


def _stub_settings(**overrides):
    s = MagicMock()
    s.GEMMA_API_URL = overrides.get("GEMMA_API_URL", "")
    s.GEMMA_API_MODEL = overrides.get("GEMMA_API_MODEL", "gemma4:e4b")
    s.GEMMA_API_TIMEOUT_SEC = overrides.get("GEMMA_API_TIMEOUT_SEC", 30.0)
    s.GEMMA_CACHE_RESPONSES = False
    s.GEMMA_SLOW_CALL_SEC = 5.0
    s.GEMMA_ANALYSIS_MAX_SAMPLE_FRAMES = 32
    s.QWEN_API_URL = overrides.get("QWEN_API_URL", "")
    s.QWEN_MAX_FRAMES = 24
    s.QWEN_MODEL = "qwen2.5vl:7b"
    return s


def _make_caption_results(n_frames: int, segment_switches: list[int] = None) -> list[dict]:
    """Build fake caption_results. segment_switches = frame indices where scene changes."""
    captions = []
    for i in range(n_frames):
        if segment_switches and i in segment_switches:
            cap = "completely different scene forest trees sky"
        else:
            cap = "urban road vehicles traffic asphalt"
        captions.append({"t_sec": float(i), "frame_path": f"/frames/f{i:04d}.jpg", "caption": cap})
    return captions


def _write_frame(path: Path, color=(100, 200, 100)) -> None:
    Image.new("RGB", (32, 24), color).save(path)


# ── Tests for _analyze_caption_sequence ──────────────────────────────────────


def test_analyze_caption_sequence_detects_boundary():
    from selfsuvis.pipeline.workflows.local._common import _analyze_caption_sequence

    # Frames 0-2 "road", frames 3-5 "forest" — one boundary at index 3
    results = [
        {
            "t_sec": float(i),
            "frame_path": f"/f{i}.jpg",
            "caption": "urban road vehicles traffic asphalt"
            if i < 3
            else "completely different scene forest trees sky",
        }
        for i in range(6)
    ]
    enriched = _analyze_caption_sequence(results, new_segment_threshold=0.45)

    assert enriched[0]["is_new_segment"] is True
    assert enriched[0]["segment_id"] == 0
    # Frame 3 triggers a new segment
    assert enriched[3]["is_new_segment"] is True
    assert enriched[3]["segment_id"] == 1
    # Frames 4-5 stay in segment 1 (same forest caption)
    assert enriched[4]["segment_id"] == 1
    assert enriched[4]["is_new_segment"] is False


def test_analyze_caption_sequence_no_change():
    from selfsuvis.pipeline.workflows.local._common import _analyze_caption_sequence

    results = _make_caption_results(4)  # all same caption → one segment
    enriched = _analyze_caption_sequence(results, new_segment_threshold=0.45)

    assert all(r["segment_id"] == 0 for r in enriched)
    assert enriched[0]["is_new_segment"] is True
    assert all(not r["is_new_segment"] for r in enriched[1:])


def test_select_segment_boundary_pairs_prefers_strongest_boundaries():
    from selfsuvis.pipeline.workflows.local.steps_caption import _select_segment_boundary_pairs

    enriched = [
        {"t_sec": 0.0, "is_new_segment": True, "similarity": None, "segment_id": 0},
        {"t_sec": 1.0, "is_new_segment": True, "similarity": 0.40, "segment_id": 1},
        {"t_sec": 2.0, "is_new_segment": True, "similarity": 0.05, "segment_id": 2},
        {"t_sec": 3.0, "is_new_segment": True, "similarity": 0.30, "segment_id": 3},
    ]

    selected = _select_segment_boundary_pairs(enriched, max_boundaries=2)

    assert len(selected) == 2
    # strongest boundaries are indices 2 (0.95 delta) and 3 (0.70 delta)
    assert selected[0][1]["t_sec"] == 2.0
    assert selected[1][1]["t_sec"] == 3.0


def test_select_ocr_candidate_frames_prefers_text_like_images(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import _select_ocr_candidate_frames

    text_img = tmp_path / "text_like.jpg"
    scene_img = tmp_path / "scene_like.jpg"

    img = Image.new("L", (160, 96), 245)
    draw = ImageDraw.Draw(img)
    for y in range(18, 78, 12):
        draw.rectangle((20, y, 140, y + 4), fill=20)
    img.convert("RGB").save(text_img)

    Image.new("RGB", (160, 96), (120, 160, 210)).save(scene_img)

    frame_list = [(str(scene_img), 0.0), (str(text_img), 1.0)]
    caption_results = [
        {
            "frame_path": str(scene_img),
            "t_sec": 0.0,
            "caption": "aerial view of open terrain and road",
            "caption_confidence": 0.3,
        },
        {
            "frame_path": str(text_img),
            "t_sec": 1.0,
            "caption": "road sign with visible text and lane markings",
            "caption_confidence": 0.3,
        },
    ]

    selected, skipped, ranking = _select_ocr_candidate_frames(
        frame_list=frame_list,
        caption_results=caption_results,
        ocr_model_id="ocr-test",
        threshold=0.55,
        max_ocr=1,
    )

    assert selected == [(str(text_img), 1.0)]
    assert str(scene_img) in skipped
    assert ranking[0]["frame_path"] == str(text_img)


# ── Tests for write_gemma_segment_captions_md ─────────────────────────────────


def test_write_gemma_segment_captions_md(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_report import write_gemma_segment_captions_md

    boundary_diffs = [
        {
            "boundary_idx": 0,
            "prev_segment_id": 0,
            "next_segment_id": 1,
            "prev_t_sec": 5.0,
            "next_t_sec": 6.0,
            "fp_before": "/frames/f0005.jpg",
            "fp_after": "/frames/f0006.jpg",
            "diff_description": "Scene shifted from urban road to forest path.",
        }
    ]
    out = tmp_path / "gemma_segment_captions.md"
    write_gemma_segment_captions_md(out, "test_video", "gemma4:e4b", boundary_diffs)

    assert out.exists()
    content = out.read_text()
    assert "Gemma Segment Boundary Diffs" in content
    assert "0→1" in content
    assert "5.0s" in content
    assert "forest path" in content


def test_write_gemma_segment_captions_md_empty(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_report import write_gemma_segment_captions_md

    out = tmp_path / "gemma_segment_captions.md"
    write_gemma_segment_captions_md(out, "vid", "gemma4:e4b", [])
    assert out.exists()
    assert "Boundaries: 0" in out.read_text()


# ── Tests for _gemma_diff_two_frames_via_api ──────────────────────────────────


def test_gemma_diff_returns_content_from_openai_compat(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import _gemma_diff_two_frames_via_api

    fp_a = tmp_path / "before.jpg"
    fp_b = tmp_path / "after.jpg"
    _write_frame(fp_a, (200, 100, 100))
    _write_frame(fp_b, (100, 100, 200))

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "The scene changed from red to blue tones."}}]
    }

    with patch("httpx.post", return_value=mock_resp):
        result = _gemma_diff_two_frames_via_api(
            str(fp_a), str(fp_b), "http://localhost:11434/v1", "gemma4:e4b", 30.0
        )

    assert result == "The scene changed from red to blue tones."


def test_gemma_diff_falls_back_to_ollama_when_content_empty(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import _gemma_diff_two_frames_via_api

    fp_a = tmp_path / "before.jpg"
    fp_b = tmp_path / "after.jpg"
    _write_frame(fp_a)
    _write_frame(fp_b, (50, 50, 50))

    # First call (OpenAI compat) returns empty content; second (Ollama native) returns text
    openai_resp = MagicMock()
    openai_resp.raise_for_status = MagicMock()
    openai_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}

    ollama_resp = MagicMock()
    ollama_resp.raise_for_status = MagicMock()
    ollama_resp.json.return_value = {"message": {"content": "New objects appeared on the left."}}

    with patch("httpx.post", side_effect=[openai_resp, ollama_resp]):
        result = _gemma_diff_two_frames_via_api(
            str(fp_a), str(fp_b), "http://localhost:11434/v1", "gemma4:e4b", 30.0
        )

    assert "New objects" in result


def test_gemma_diff_returns_empty_on_missing_frames(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import _gemma_diff_two_frames_via_api

    result = _gemma_diff_two_frames_via_api(
        "/missing/before.jpg", "/missing/after.jpg", "http://localhost:11434/v1", "gemma4:e4b", 30.0
    )
    assert result == ""


# ── Tests for step_gemma_segment_captions ────────────────────────────────────


@patch("selfsuvis.pipeline.workflows.local.steps_caption.settings")
def test_step_gemma_segment_captions_skips_when_no_api_url(mock_settings, tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import step_gemma_segment_captions

    mock_settings.GEMMA_API_URL = ""
    mock_settings.GEMMA_API_MODEL = "gemma4:e4b"
    mock_settings.GEMMA_API_TIMEOUT_SEC = 30.0

    result = step_gemma_segment_captions(
        [], [], "vid", tmp_path, gemma_api_url="", gemma_api_model=""
    )
    assert result["skipped"] is True
    assert "GEMMA_API_URL" in result["reason"]


@patch("selfsuvis.pipeline.workflows.local.steps_caption.settings")
def test_step_gemma_segment_captions_skips_when_no_captions(mock_settings, tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import step_gemma_segment_captions

    mock_settings.GEMMA_API_URL = ""
    mock_settings.GEMMA_API_MODEL = "gemma4:e4b"
    mock_settings.GEMMA_API_TIMEOUT_SEC = 30.0

    result = step_gemma_segment_captions(
        [], [], "vid", tmp_path, gemma_api_url="http://localhost:11434/v1"
    )
    assert result["skipped"] is True
    assert "no caption" in result["reason"]


@patch("selfsuvis.pipeline.workflows.local.steps_caption._gemma_diff_two_frames_via_api")
@patch("selfsuvis.pipeline.workflows.local.steps_caption.settings")
def test_step_gemma_segment_captions_writes_md(mock_settings, mock_diff, tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import step_gemma_segment_captions

    mock_settings.GEMMA_API_URL = ""
    mock_settings.GEMMA_API_MODEL = "gemma4:e4b"
    mock_settings.GEMMA_API_TIMEOUT_SEC = 30.0
    mock_diff.return_value = "The vehicle count dropped significantly."

    caption_results = _make_caption_results(6, segment_switches=[3])
    frame_list = [(r["frame_path"], r["t_sec"]) for r in caption_results]

    result = step_gemma_segment_captions(
        frame_list,
        caption_results,
        "test_video",
        tmp_path,
        gemma_api_url="http://localhost:11434/v1",
        gemma_api_model="gemma4:e4b",
    )

    assert result["skipped"] is False
    assert result["boundary_count"] >= 1
    assert result["described_count"] >= 1
    assert (tmp_path / "gemma_segment_captions.md").exists()
    assert mock_diff.called


@patch("selfsuvis.pipeline.workflows.local.steps_caption._gemma_diff_two_frames_via_api")
@patch("selfsuvis.pipeline.workflows.local.steps_caption.settings")
def test_step_gemma_segment_captions_no_boundaries(mock_settings, mock_diff, tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import step_gemma_segment_captions

    mock_settings.GEMMA_API_URL = ""
    mock_settings.GEMMA_API_MODEL = "gemma4:e4b"
    mock_settings.GEMMA_API_TIMEOUT_SEC = 30.0

    caption_results = _make_caption_results(4)  # all same → no boundaries
    frame_list = [(r["frame_path"], r["t_sec"]) for r in caption_results]

    result = step_gemma_segment_captions(
        frame_list,
        caption_results,
        "vid",
        tmp_path,
        gemma_api_url="http://localhost:11434/v1",
    )
    assert result["skipped"] is True
    assert "no segment boundaries" in result["reason"]
    mock_diff.assert_not_called()


# ── Tests for _gemma_extract_frame_structured (Qwen fallback) ─────────────────


def test_gemma_extract_frame_structured_parses_json(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import _gemma_extract_frame_structured

    fp = tmp_path / "frame.jpg"
    _write_frame(fp)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "vehicle_groups": [
                                {"type": "car", "count": 2, "color": "red", "position": "centre"}
                            ],
                            "road_surface": "asphalt",
                            "road_condition": "clear",
                            "scene_summary": "Two red cars on a dry asphalt road.",
                        }
                    )
                }
            }
        ]
    }

    with patch("httpx.post", return_value=mock_resp):
        result = _gemma_extract_frame_structured(
            str(fp), "http://localhost:11434/v1", "gemma4:e4b", 30.0, t_sec=2.5
        )

    assert result["road_surface"] == "asphalt"
    assert result["road_condition"] == "clear"
    assert len(result["vehicle_groups"]) == 1
    assert result["vehicle_groups"][0]["type"] == "car"
    assert result["t_sec"] == 2.5


def test_gemma_extract_frame_structured_returns_parse_error_on_bad_json(tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import _gemma_extract_frame_structured

    fp = tmp_path / "frame.jpg"
    _write_frame(fp)

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "not json at all"}}]}

    # Second call (Ollama native fallback) also returns bad content
    ollama_resp = MagicMock()
    ollama_resp.raise_for_status = MagicMock()
    ollama_resp.json.return_value = {"message": {"content": "still not json"}}

    with patch("httpx.post", side_effect=[mock_resp, ollama_resp]):
        result = _gemma_extract_frame_structured(
            str(fp), "http://localhost:11434/v1", "gemma4:e4b", 30.0, t_sec=1.0
        )

    assert result.get("parse_error") is True


# ── Tests for _step_qwen_captioning_gemma_fallback ───────────────────────────


@patch("selfsuvis.pipeline.workflows.local.steps_caption._gemma_extract_frame_structured")
@patch("selfsuvis.pipeline.workflows.local.steps_caption.settings")
def test_qwen_gemma_fallback_returns_structured_results(mock_settings, mock_extract, tmp_path):
    from selfsuvis.pipeline.workflows.local.steps_caption import (
        _step_qwen_captioning_gemma_fallback,
    )

    mock_settings.QWEN_MAX_FRAMES = 4
    mock_settings.GEMMA_API_TIMEOUT_SEC = 30.0
    mock_extract.return_value = {
        "t_sec": 0.0,
        "frame_path": "/frames/f0.jpg",
        "vehicle_groups": [],
        "road_surface": "asphalt",
        "road_condition": "clear",
        "scene_summary": "Empty road.",
    }

    frame_list = [("/frames/f0.jpg", 0.0), ("/frames/f1.jpg", 1.0), ("/frames/f2.jpg", 2.0)]
    result = _step_qwen_captioning_gemma_fallback(
        frame_list, "test_video", tmp_path, "http://localhost:11434/v1", "gemma4:e4b"
    )

    assert result["skipped"] is False
    assert result["backend"] == "gemma_fallback"
    assert result["ok_count"] >= 0
    assert len(result["results"]) > 0
    assert (tmp_path / "detailed_captions.md").exists()


@patch("selfsuvis.pipeline.workflows.local.steps_caption.settings")
def test_step_qwen_captioning_uses_gemma_fallback_when_qwen_disabled(mock_settings, tmp_path):
    """step_qwen_captioning falls back to Gemma when Qwen disabled + GEMMA_API_URL set."""
    from selfsuvis.pipeline.workflows.local.steps_caption import step_qwen_captioning

    mock_settings.QWEN_API_URL = ""
    mock_settings.GEMMA_API_URL = "http://localhost:11434/v1"
    mock_settings.GEMMA_API_MODEL = "gemma4:e4b"
    mock_settings.GEMMA_API_TIMEOUT_SEC = 30.0
    mock_settings.QWEN_MAX_FRAMES = 4
    mock_settings.QWEN_MODEL = "qwen2.5vl:7b"

    fallback_result = {"skipped": False, "results": [], "ok_count": 0, "backend": "gemma_fallback"}
    mock_qwen_instance = MagicMock()
    mock_qwen_instance.is_enabled.return_value = False

    frame_list = [("/frames/f0.jpg", 0.0)]
    with (
        patch(
            "selfsuvis.pipeline.workflows.local.steps_caption._step_qwen_captioning_gemma_fallback",
            return_value=fallback_result,
        ) as mock_fallback,
        patch.dict(
            "sys.modules",
            {
                "selfsuvis.pipeline.vision.qwen": MagicMock(
                    QwenModel=MagicMock(return_value=mock_qwen_instance)
                )
            },
        ),
    ):
        result = step_qwen_captioning(frame_list, "vid", tmp_path, {}, [])

    mock_fallback.assert_called_once()
    assert result["backend"] == "gemma_fallback"
