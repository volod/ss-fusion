from selfsuvis.pipeline.vision import rfdetr


def test_rfdetr_weights_path_uses_data_dir(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    monkeypatch.setattr(rfdetr.settings, "DATA_DIR", str(data_dir))

    resolved = rfdetr.Path(rfdetr._rfdetr_weights_path("base"))

    assert resolved == data_dir.resolve() / "models" / "rfdetr" / "rf-detr-base.pth"


def test_expand_target_labels_maps_abstract_vehicle_category_to_detector_labels():
    expanded = rfdetr._expand_target_labels(["vehicle"])

    assert "vehicle" in expanded
    assert "car" in expanded
    assert "truck" in expanded


def test_label_matches_any_handles_alias_expansion_inputs():
    expanded = rfdetr._expand_target_labels(["person"])

    assert rfdetr._label_matches_any("pedestrian", expanded) is True
    assert rfdetr._label_matches_any("truck", expanded) is False


def test_track_match_score_preserves_vehicle_track_on_small_center_shift():
    track = {
        "label": "car",
        "bbox_norm": [0.10, 0.10, 0.20, 0.20],
    }
    det = {
        "label": "truck",
        "bbox_norm": [0.15, 0.10, 0.25, 0.20],
    }

    score = rfdetr._track_match_score(track, det)

    assert score >= 0.10


def test_track_match_score_rejects_cross_category_match():
    track = {
        "label": "person",
        "bbox_norm": [0.10, 0.10, 0.20, 0.20],
    }
    det = {
        "label": "car",
        "bbox_norm": [0.11, 0.10, 0.21, 0.20],
    }

    assert rfdetr._track_match_score(track, det) == 0.0
