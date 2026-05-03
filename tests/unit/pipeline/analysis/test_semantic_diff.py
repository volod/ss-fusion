"""Unit tests for Phase 4 semantic change detection additions.

Tests compute_semantic_diff, generate_change_explanation, and the
integration of semantic_diff_json into detect_changes output.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from selfsuvis.pipeline.analysis.change_detection import (
    compute_semantic_diff,
    detect_changes,
    generate_change_explanation,
)

# ── compute_semantic_diff ─────────────────────────────────────────────────────

def test_semantic_diff_empty_when_identical():
    facts = {
        "vehicle_groups": [{"type": "truck", "count": 2}],
        "road_condition": "clear",
        "road_surface": "asphalt",
    }
    diff = compute_semantic_diff(facts, facts)
    assert diff == {}


def test_semantic_diff_vehicle_count_increased():
    ref = {"vehicle_groups": [{"type": "truck", "count": 2}], "road_condition": "clear"}
    new = {"vehicle_groups": [{"type": "truck", "count": 5}], "road_condition": "clear"}
    diff = compute_semantic_diff(ref, new)
    assert "vehicle_count" in diff
    assert diff["vehicle_count"]["before"] == 2
    assert diff["vehicle_count"]["after"] == 5
    assert diff["vehicle_count"]["delta"] == 3


def test_semantic_diff_vehicle_count_decreased():
    ref = {"vehicle_groups": [{"count": 4}]}
    new = {"vehicle_groups": [{"count": 1}]}
    diff = compute_semantic_diff(ref, new)
    assert diff["vehicle_count"]["delta"] == -3


def test_semantic_diff_multiple_groups_summed():
    ref = {"vehicle_groups": [{"count": 2}, {"count": 3}]}
    new = {"vehicle_groups": [{"count": 10}]}
    diff = compute_semantic_diff(ref, new)
    assert diff["vehicle_count"]["before"] == 5
    assert diff["vehicle_count"]["after"] == 10


def test_semantic_diff_road_condition_changed():
    ref = {"road_condition": "clear", "road_surface": "asphalt"}
    new = {"road_condition": "wet", "road_surface": "asphalt"}
    diff = compute_semantic_diff(ref, new)
    assert "road_condition" in diff
    assert diff["road_condition"] == {"before": "clear", "after": "wet"}
    assert "road_surface" not in diff


def test_semantic_diff_road_surface_changed():
    ref = {"road_surface": "asphalt"}
    new = {"road_surface": "gravel"}
    diff = compute_semantic_diff(ref, new)
    assert "road_surface" in diff
    assert diff["road_surface"] == {"before": "asphalt", "after": "gravel"}


def test_semantic_diff_no_vehicle_groups_in_facts():
    """Missing vehicle_groups → no vehicle_count entry in diff."""
    ref = {"road_condition": "clear"}
    new = {"road_condition": "wet"}
    diff = compute_semantic_diff(ref, new)
    assert "vehicle_count" not in diff
    assert "road_condition" in diff


def test_semantic_diff_empty_vehicle_groups():
    ref = {"vehicle_groups": []}
    new = {"vehicle_groups": [{"count": 3}]}
    diff = compute_semantic_diff(ref, new)
    assert diff["vehicle_count"]["before"] == 0
    assert diff["vehicle_count"]["after"] == 3


def test_semantic_diff_invalid_vehicle_groups_type():
    """Non-list vehicle_groups → vehicle_count treated as None → no entry."""
    ref = {"vehicle_groups": "invalid"}
    new = {"vehicle_groups": [{"count": 2}]}
    diff = compute_semantic_diff(ref, new)
    assert "vehicle_count" not in diff


# ── generate_change_explanation ───────────────────────────────────────────────

def test_generate_change_explanation_returns_none_when_no_gemma_url(monkeypatch):
    import selfsuvis.pipeline.analysis.change_detection as cd
    monkeypatch.setattr(cd.settings, "GEMMA_API_URL", "")
    result = generate_change_explanation({"vehicle_count": {"before": 2, "after": 5, "delta": 3}}, 0.4)
    assert result is None


def test_generate_change_explanation_returns_none_for_empty_diff(monkeypatch):
    import selfsuvis.pipeline.analysis.change_detection as cd
    monkeypatch.setattr(cd.settings, "GEMMA_API_URL", "http://gemma:11434/v1")
    result = generate_change_explanation({}, 0.3)
    assert result is None


def test_generate_change_explanation_calls_gemma_api(monkeypatch):
    import selfsuvis.pipeline.analysis.change_detection as cd
    monkeypatch.setattr(cd.settings, "GEMMA_API_URL", "http://gemma:11434/v1")
    monkeypatch.setattr(cd.settings, "GEMMA_API_MODEL", "gemma4:e4b")

    choice = MagicMock()
    choice.message.content = "Vehicle traffic increased significantly at this waypoint."
    resp = MagicMock()
    resp.choices = [choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = resp

    with patch("openai.OpenAI", return_value=fake_client):
        result = generate_change_explanation(
            {"vehicle_count": {"before": 1, "after": 4, "delta": 3}},
            change_score=0.45,
        )

    assert result is not None
    assert "vehicle" in result.lower() or "traffic" in result.lower() or len(result) > 0


def test_generate_change_explanation_truncates_at_first_sentence(monkeypatch):
    import selfsuvis.pipeline.analysis.change_detection as cd
    monkeypatch.setattr(cd.settings, "GEMMA_API_URL", "http://gemma:11434/v1")
    monkeypatch.setattr(cd.settings, "GEMMA_API_MODEL", "gemma4:e4b")

    choice = MagicMock()
    choice.message.content = "First sentence. Second sentence. Third."
    resp = MagicMock()
    resp.choices = [choice]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = resp

    with patch("openai.OpenAI", return_value=fake_client):
        result = generate_change_explanation(
            {"road_condition": {"before": "clear", "after": "wet"}},
            change_score=0.3,
        )

    # Should be truncated at first period
    assert result == "First sentence."


def test_generate_change_explanation_returns_none_on_api_error(monkeypatch):
    import selfsuvis.pipeline.analysis.change_detection as cd
    monkeypatch.setattr(cd.settings, "GEMMA_API_URL", "http://gemma:11434/v1")
    monkeypatch.setattr(cd.settings, "GEMMA_API_MODEL", "gemma4:e4b")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = ConnectionError("refused")

    with patch("openai.OpenAI", return_value=fake_client):
        result = generate_change_explanation(
            {"vehicle_count": {"before": 2, "after": 5, "delta": 3}},
            change_score=0.4,
        )

    assert result is None


# ── detect_changes integration with frame_facts_json ─────────────────────────

def _make_embedding(val: float = 1.0, dim: int = 4) -> list:
    v = np.zeros(dim)
    v[0] = val
    return v.tolist()


def _make_candidate(frame_id: str, mission_id: str, emb_val: float, facts=None):
    return {
        "frame_id": frame_id,
        "mission_id": mission_id,
        "embedding": _make_embedding(emb_val),
        "frame_facts_json": facts,
    }


def test_detect_changes_includes_semantic_diff_when_facts_present():
    """detect_changes sets semantic_diff_json when both frames have facts."""
    new_facts = {"vehicle_groups": [{"count": 5}], "road_condition": "wet"}
    ref_facts  = {"vehicle_groups": [{"count": 2}], "road_condition": "clear"}

    new_frames = [
        {
            "frame_id": "f_new",
            "mission_id": "m_new",
            "embedding": _make_embedding(0.0),  # orthogonal to ref → dist=1.0
            "gps": {"lat": 47.0, "lon": 8.0},
            "frame_facts_json": new_facts,
        }
    ]

    ref_candidate = _make_candidate("f_ref", "m_old", 1.0, ref_facts)

    def _query_fn(emb, bbox):
        return [ref_candidate]

    changes = detect_changes(new_frames, _query_fn, threshold=0.5)
    assert len(changes) == 1
    diff = changes[0]["semantic_diff_json"]
    assert diff is not None
    assert "vehicle_count" in diff
    assert diff["vehicle_count"]["before"] == 2
    assert diff["vehicle_count"]["after"] == 5


def test_detect_changes_semantic_diff_none_when_no_facts():
    """When frames lack facts, semantic_diff_json is None."""
    new_frames = [
        {
            "frame_id": "f_new",
            "mission_id": "m_new",
            "embedding": _make_embedding(0.0),
            "gps": {"lat": 47.0, "lon": 8.0},
            # no frame_facts_json
        }
    ]
    ref_candidate = {"frame_id": "f_ref", "mission_id": "m_old", "embedding": _make_embedding(1.0)}

    def _query_fn(emb, bbox):
        return [ref_candidate]

    changes = detect_changes(new_frames, _query_fn, threshold=0.5)
    assert len(changes) == 1
    assert changes[0]["semantic_diff_json"] is None


def test_detect_changes_empty_diff_when_facts_identical():
    """Identical facts → semantic_diff_json is {}."""
    facts = {"vehicle_groups": [{"count": 3}], "road_condition": "clear"}
    new_frames = [
        {
            "frame_id": "f_new",
            "mission_id": "m_new",
            "embedding": _make_embedding(0.0),
            "gps": {"lat": 47.0, "lon": 8.0},
            "frame_facts_json": facts,
        }
    ]
    ref = _make_candidate("f_ref", "m_old", 1.0, facts)

    def _query_fn(emb, bbox):
        return [ref]

    changes = detect_changes(new_frames, _query_fn, threshold=0.5)
    assert len(changes) == 1
    assert changes[0]["semantic_diff_json"] == {}
