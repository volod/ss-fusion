"""Unit tests for pipeline.supervised_finetune._normalize_labels."""
import logging

from selfsuvis.pipeline.training.supervised import _normalize_labels


class TestNormalizeLabels:
    def test_empty_input(self):
        assert _normalize_labels([], {}) == []

    def test_passthrough_when_no_mappings(self):
        rows = [("a.jpg", "car"), ("b.jpg", "truck")]
        result = _normalize_labels(rows, {})
        assert set(result) == {("a.jpg", "car"), ("b.jpg", "truck")}

    def test_applies_mapping(self):
        rows = [("a.jpg", "vehicle"), ("b.jpg", "car")]
        mappings = {"vehicle": "car"}
        result = _normalize_labels(rows, mappings)
        assert set(result) == {("a.jpg", "car"), ("b.jpg", "car")}

    def test_unknown_labels_pass_through(self):
        rows = [("a.jpg", "unknown_class")]
        mappings = {"car": "vehicle"}
        result = _normalize_labels(rows, mappings)
        assert result == [("a.jpg", "unknown_class")]

    def test_deduplicates_identical_frame_same_label(self):
        """Same frame + same label after normalization → one entry."""
        rows = [("a.jpg", "car"), ("a.jpg", "car")]
        result = _normalize_labels(rows, {})
        assert result == [("a.jpg", "car")]

    def test_conflict_keeps_alphabetically_first(self):
        """Same frame, different canonical labels → alphabetically first wins."""
        rows = [("a.jpg", "truck"), ("a.jpg", "car")]
        result = _normalize_labels(rows, {})
        assert result == [("a.jpg", "car")]

    def test_conflict_after_mapping_logs_warning(self, caplog):
        """Conflict triggered by normalization mapping emits a warning."""
        rows = [("a.jpg", "automobile"), ("a.jpg", "truck")]
        mappings = {"automobile": "car", "truck": "car"}
        with caplog.at_level(logging.WARNING, logger="selfsuvis.pipeline.supervised_finetune"):
            result = _normalize_labels(rows, mappings)
        # After mapping both become "car" — same label, no conflict
        assert result == [("a.jpg", "car")]

    def test_conflict_different_canonical_logs_warning(self, caplog):
        """Frame appearing under two different canonical labels after mapping → warning."""
        rows = [("x.jpg", "auto"), ("x.jpg", "van")]
        mappings = {"auto": "car", "van": "truck"}
        with caplog.at_level(logging.WARNING, logger="selfsuvis.pipeline.supervised_finetune"):
            result = _normalize_labels(rows, mappings)
        assert len(result) == 1
        # alphabetically first: "car" < "truck"
        assert result[0] == ("x.jpg", "car")
        assert any("conflict" in r.message.lower() for r in caplog.records)

    def test_multiple_frames_independent(self):
        """Conflicts on one frame don't affect other frames."""
        rows = [
            ("a.jpg", "van"),
            ("a.jpg", "truck"),
            ("b.jpg", "car"),
        ]
        mappings = {}
        result = _normalize_labels(rows, mappings)
        result_dict = dict(result)
        assert result_dict["b.jpg"] == "car"
        assert result_dict["a.jpg"] in ("truck", "van")

    def test_many_frames_no_conflicts(self):
        rows = [(f"{i}.jpg", "car" if i % 2 == 0 else "truck") for i in range(20)]
        result = _normalize_labels(rows, {})
        assert len(result) == 20
