"""Unit tests for Gemma-directed tracking helpers and step orchestration."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_frame(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 24), color).save(path)


_STUB_MODULE_NAMES = [
    "ssv_vdp.steps.gemma_tracking",
    "ssv_vdp.steps.perception",
    "pipeline",
    "selfsuvis.pipeline.core",
    "selfsuvis.pipeline.vision",
    "selfsuvis.pipeline.vision.rfdetr",
    "selfsuvis.pipeline.workflows",
    "selfsuvis.pipeline.workflows.local",
    "ssv_vdp.steps.common",
]


def _load_steps_module():
    module_name = "ssv_vdp.steps.gemma_tracking"
    module_path = ROOT / "src/ssv_vdp/steps/gemma_tracking.py"

    # Save originals so we can restore them after loading (prevent contamination
    # of later tests that import real pipeline.core).
    saved = {k: sys.modules.get(k) for k in _STUB_MODULE_NAMES}

    for name in _STUB_MODULE_NAMES:
        sys.modules.pop(name, None)

    pipeline_pkg = types.ModuleType("pipeline")
    pipeline_pkg.__path__ = [str(ROOT / "src/selfsuvis/pipeline")]
    sys.modules["pipeline"] = pipeline_pkg

    settings = types.SimpleNamespace(
        RFDETR_ENABLED=True,
        RFDETR_MODEL="base",
        RFDETR_CONFIDENCE=0.35,
        SAM_ENABLED=True,
        GEMMA_API_TIMEOUT_SEC=60,
        GEMMA_TRACKING_MAX_SAMPLE_FRAMES=12,
        GEMMA_CACHE_RESPONSES=False,
        GEMMA_SLOW_CALL_SEC=30,
    )
    core_mod = types.ModuleType("selfsuvis.pipeline.core")
    core_mod.settings = settings
    sys.modules["selfsuvis.pipeline.core"] = core_mod

    vision_pkg = types.ModuleType("selfsuvis.pipeline.vision")
    vision_pkg.__path__ = [str(ROOT / "src/selfsuvis/pipeline/vision")]
    sys.modules["selfsuvis.pipeline.vision"] = vision_pkg

    rfdetr_mod = types.ModuleType("selfsuvis.pipeline.vision.rfdetr")

    class StubRFDETRTracker:
        def __init__(self) -> None:
            self.model_id = "rfdetr_base"

        def is_enabled(self) -> bool:
            return True

        def track_sequence(self, frame_items, target_labels=None):
            return [
                {"frame_path": fp, "t_sec": t_sec, "detections": []} for fp, t_sec in frame_items
            ]

        def release(self) -> None:
            pass

    def _label_matches_any(label, target_labels):
        label = label.lower()
        return any(t.lower() in label or label in t.lower() for t in target_labels)

    def _classify_priority(label):
        label = label.lower()
        if label == "person":
            return 1
        if label in {"vehicle", "car", "truck"}:
            return 2
        return 3

    rfdetr_mod.RFDETRTracker = StubRFDETRTracker
    rfdetr_mod._label_matches_any = _label_matches_any
    rfdetr_mod._classify_priority = _classify_priority
    sys.modules["selfsuvis.pipeline.vision.rfdetr"] = rfdetr_mod

    workflows_pkg = types.ModuleType("selfsuvis.pipeline.workflows")
    workflows_pkg.__path__ = [str(ROOT / "src/selfsuvis/pipeline/workflows")]
    sys.modules["selfsuvis.pipeline.workflows"] = workflows_pkg

    local_pkg = types.ModuleType("selfsuvis.pipeline.workflows.local")
    local_pkg.__path__ = [str(ROOT / "src/selfsuvis/pipeline/workflows/local")]
    sys.modules["selfsuvis.pipeline.workflows.local"] = local_pkg

    # Stub the perception package so its __init__.py (which imports embed.py →
    # OpenCLIPEmbedder → heavy pipeline deps) is never executed.
    perception_pkg = types.ModuleType("ssv_vdp.steps.perception")
    perception_pkg.__path__ = [str(ROOT / "src/ssv_vdp/steps/perception")]
    sys.modules["ssv_vdp.steps.perception"] = perception_pkg

    common_mod = types.ModuleType("ssv_vdp.steps.common")
    common_mod._log = types.SimpleNamespace(
        info=lambda *a, **k: None, debug=lambda *a, **k: None, warning=lambda *a, **k: None
    )
    common_mod._open_frame_image = lambda frame_path: Image.open(frame_path).convert("RGB")
    sys.modules["ssv_vdp.steps.common"] = common_mod

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Restore all stubbed entries except the loaded module itself so that
    # subsequent tests can import real pipeline.core / pipeline etc.
    for k, v in saved.items():
        if k == module_name:
            continue  # keep the freshly loaded module
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v

    return module


def test_aggregate_scene_responses_prefers_specific_bbox_and_ranked_priority():
    tracking = _load_steps_module()

    responses = [
        {
            "scene_type": "urban_street",
            "dominant_objects": [
                {
                    "category": "Vehicle",
                    "count_estimate": 2,
                    "spatial_hint": "center",
                    "rough_bbox": [0.1, 0.1, 0.9, 0.9],
                },
                {
                    "category": "Person",
                    "count_estimate": 1,
                    "spatial_hint": "left",
                    "rough_bbox": [0.2, 0.2, 0.4, 0.7],
                },
            ],
            "areas_of_interest": ["intersection"],
            "motion_present": False,
            "tracking_priority": ["vehicle", "person"],
        },
        {
            "scene_type": "urban_street",
            "dominant_objects": [
                {
                    "category": "vehicle",
                    "count_estimate": 3,
                    "spatial_hint": "foreground",
                    "rough_bbox": [0.25, 0.2, 0.55, 0.6],
                }
            ],
            "areas_of_interest": ["intersection", "crosswalk"],
            "motion_present": True,
            "tracking_priority": ["vehicle", "truck"],
        },
    ]

    aggregated = tracking._aggregate_scene_responses(responses)

    assert aggregated["scene_type"] == "urban_street"
    assert aggregated["motion_present"] is True
    assert aggregated["areas_of_interest"] == ["intersection", "crosswalk"]
    assert aggregated["tracking_priority"] == ["vehicle", "person", "truck"]
    assert [obj["category"] for obj in aggregated["dominant_objects"]] == ["vehicle", "person"]
    assert aggregated["dominant_objects"][0]["rough_bbox"] == [0.25, 0.2, 0.55, 0.6]


def test_sam_directed_by_gemma_path_b_is_pure_fallback_not_supplement():
    """Path B (auto-mask) only runs when Path A finds nothing.

    When Path A succeeds (vehicle with a small bbox), Path B is NOT used as a
    supplement — only the single Path A result is returned. This is intentional:
    running auto-mask generation alongside Path A caused 35+ min freezes.
    """
    tracking = _load_steps_module()

    image = Image.new("RGB", (20, 20), (0, 0, 0))

    class FakePredictor:
        def predict_boxes(self, _image, boxes):
            results = []
            for idx, _box in enumerate(boxes):
                mask = np.zeros((20, 20), dtype=bool)
                mask[2 + idx : 8 + idx, 2:8] = True
                results.append({"mask": mask, "score": 0.9 - idx * 0.1})
            return results

    class FakeClip:
        def encode_texts(self, labels):
            return np.stack([np.array([1.0, 0.0], dtype=np.float32)] * len(labels))

        def encode_images(self, crops):
            return np.array([[0.95, 0.05]], dtype=np.float32)

    auto_mask = np.zeros((20, 20), dtype=bool)
    auto_mask[10:19, 10:19] = True

    original = tracking._get_sam_auto_masks
    tracking._get_sam_auto_masks = lambda *_args, **_kwargs: [
        {"mask": auto_mask, "bbox": [10, 10, 9, 9], "area": 81, "score": 0.7}
    ]
    try:
        results = tracking._sam_directed_by_gemma(
            image,
            gemma_objects=[
                # small bbox → Path A; large bbox → skipped by Path A (near-fallback)
                {"category": "vehicle", "rough_bbox": [0.1, 0.1, 0.4, 0.4]},
                {"category": "person", "rough_bbox": [0.05, 0.05, 0.95, 0.95]},
            ],
            sam_predictor=FakePredictor(),
            clip_model=FakeClip(),
        )
    finally:
        tracking._get_sam_auto_masks = original

    # Path A succeeded → Path B does NOT supplement; only the gemma_bbox result.
    assert len(results) == 1
    assert results[0]["source"] == "gemma_bbox"
    assert results[0]["category"] == "vehicle"


def test_step_gemma_directed_tracking_writes_outputs_and_tracks_priority(monkeypatch, tmp_path):
    tracking = _load_steps_module()
    # Patches must target the perception module (where step_gemma_directed_tracking
    # looks up its globals), not the shim module.
    _pm = sys.modules["ssv_vdp.steps.perception.gemma_tracking"]

    frame_a = tmp_path / "frame_a.jpg"
    frame_b = tmp_path / "frame_b.jpg"
    _write_frame(frame_a, (255, 0, 0))
    _write_frame(frame_b, (0, 255, 0))
    frame_list = [(str(frame_a), 0.0), (str(frame_b), 1.0)]

    gemma_scene = {
        "scene_type": "urban_street",
        "dominant_objects": [
            {
                "category": "vehicle",
                "count_estimate": 2,
                "spatial_hint": "center",
                "rough_bbox": [0.2, 0.2, 0.5, 0.5],
            }
        ],
        "areas_of_interest": ["road center"],
        "motion_present": True,
        "tracking_priority": ["vehicle"],
    }
    monkeypatch.setattr(
        _pm, "_gemma_structured_scene_analysis", lambda *args, **kwargs: gemma_scene
    )

    class FakeTracker:
        def __init__(self) -> None:
            self.model_id = "rfdetr_base"
            self.calls = []

        def is_enabled(self) -> bool:
            return True

        def track_sequence(self, frames, target_labels=None):
            self.calls.append({"frames": frames, "target_labels": target_labels})
            return [
                {
                    "frame_path": str(frame_a),
                    "t_sec": 0.0,
                    "detections": [
                        {
                            "label": "vehicle",
                            "confidence": 0.91,
                            "bbox_norm": [0.1, 0.2, 0.4, 0.6],
                            "track_id": 7,
                            "priority": 2,
                            "priority_label": "vehicle",
                        }
                    ],
                },
                {"frame_path": str(frame_b), "t_sec": 1.0, "detections": []},
            ]

        def release(self) -> None:
            pass

    fake_tracker = FakeTracker()
    monkeypatch.setattr(_pm, "RFDETRTracker", lambda: fake_tracker)

    class FakeSAMPredictor:
        def is_available(self) -> bool:
            return True

        def release(self) -> None:
            pass

    fake_sam_module = types.SimpleNamespace(SAMPredictor=lambda: FakeSAMPredictor())
    monkeypatch.setitem(sys.modules, "selfsuvis.pipeline.vision.sam", fake_sam_module)
    monkeypatch.setattr(tracking.settings, "RFDETR_ENABLED", True)
    monkeypatch.setattr(tracking.settings, "SAM_ENABLED", True)
    monkeypatch.setattr(tracking.settings, "GEMMA_API_TIMEOUT_SEC", 3.0)
    monkeypatch.setattr(
        _pm,
        "_sam_directed_by_gemma",
        lambda *_args, **_kwargs: [
            {
                "category": "vehicle",
                "area_norm": 0.08,
                "source": "gemma_bbox",
                "score": 0.82,
                "clip_score": None,
            }
        ],
    )

    result = tracking.step_gemma_directed_tracking(
        frame_list=frame_list,
        video_name="demo-video",
        video_dir=tmp_path,
        device="cpu",
        models={"clip": object()},
        gemma_api_url="http://gemma.local/v1",
        gemma_api_model="gemma4:e4b",
    )

    assert result["skipped"] is False
    assert result["scene_type"] == "urban_street"
    assert result["tracking_priority"] == ["vehicle"]
    assert result["n_tracked_objects"] == 1
    assert result["sam_masks_total"] == 2
    assert fake_tracker.calls[0]["target_labels"] == ["vehicle"]

    results_path = tmp_path / "gemma_tracking_results.json"
    summary_path = tmp_path / "gemma_tracking_summary.md"
    annotated_path = tmp_path / "gemma_tracking" / "frame_0.000_tracked.jpg"

    assert results_path.is_file()
    assert summary_path.is_file()
    assert annotated_path.is_file()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["gemma_scene_type"] == "urban_street"
    assert payload["tracking_priority"] == ["vehicle"]
    assert payload["sam_enabled"] is True
    assert payload["frames"][0]["sam_masks"][0]["source"] == "gemma_bbox"
    assert "SAM directed segmentation produced **2 masks**." in summary_path.read_text(
        encoding="utf-8"
    )


def test_step_gemma_directed_tracking_retries_without_label_filter_when_first_pass_is_empty(
    monkeypatch, tmp_path
):
    tracking = _load_steps_module()
    _pm = sys.modules["ssv_vdp.steps.perception.gemma_tracking"]

    frame_a = tmp_path / "frame_a.jpg"
    _write_frame(frame_a, (255, 0, 0))
    frame_list = [(str(frame_a), 0.0)]

    gemma_scene = {
        "scene_type": "urban_street",
        "dominant_objects": [{"category": "vehicle", "rough_bbox": [0.2, 0.2, 0.5, 0.5]}],
        "areas_of_interest": [],
        "motion_present": False,
        "tracking_priority": ["vehicle"],
    }
    monkeypatch.setattr(
        _pm, "_gemma_structured_scene_analysis", lambda *args, **kwargs: gemma_scene
    )

    class FakeTracker:
        def __init__(self) -> None:
            self.model_id = "rfdetr_base"
            self.calls = []

        def is_enabled(self) -> bool:
            return True

        def track_sequence(self, frames, target_labels=None):
            self.calls.append(target_labels)
            if target_labels:
                return [{"frame_path": str(frame_a), "t_sec": 0.0, "detections": []}]
            return [
                {
                    "frame_path": str(frame_a),
                    "t_sec": 0.0,
                    "detections": [
                        {
                            "label": "truck",
                            "confidence": 0.9,
                            "bbox_norm": [0.1, 0.1, 0.4, 0.4],
                            "track_id": 1,
                            "priority": 2,
                            "priority_label": "vehicle",
                        }
                    ],
                }
            ]

        def release(self) -> None:
            pass

    monkeypatch.setattr(_pm, "RFDETRTracker", FakeTracker)
    monkeypatch.setattr(tracking.settings, "RFDETR_ENABLED", True)
    monkeypatch.setattr(tracking.settings, "SAM_ENABLED", False)
    monkeypatch.setattr(tracking.settings, "GEMMA_API_TIMEOUT_SEC", 3.0)

    result = tracking.step_gemma_directed_tracking(
        frame_list=frame_list,
        video_name="demo-video",
        video_dir=tmp_path,
        device="cpu",
        models={"clip": object()},
        gemma_api_url="http://gemma.local/v1",
        gemma_api_model="gemma4:e4b",
    )

    assert result["skipped"] is False
    assert result["total_objects"] == 1


def test_normalise_tracking_targets_drops_scene_nouns():
    tracking = _load_steps_module()

    targets = tracking._normalise_tracking_targets(
        ["vehicle", "intersection", "road", "traffic"],
        [{"category": "truck"}],
    )

    assert targets == ["vehicle"]


def test_scene_is_actionable_requires_detector_aligned_targets():
    tracking = _load_steps_module()

    assert (
        tracking._scene_is_actionable(
            {
                "scene_type": "urban_street",
                "tracking_priority": ["intersection", "roadway"],
                "dominant_objects": [],
            }
        )
        is False
    )
    assert (
        tracking._scene_is_actionable(
            {
                "scene_type": "urban_street",
                "tracking_priority": ["vehicle", "roadway"],
                "dominant_objects": [
                    {
                        "category": "vehicle",
                        "rough_bbox": [0.1, 0.1, 0.9, 0.9],
                    }
                ],
            }
        )
        is True
    )
    assert (
        tracking._scene_is_actionable(
            {
                "scene_type": "aerial",
                "tracking_priority": ["vehicle"],
                "dominant_objects": [],
                "areas_of_interest": ["road corridor"],
                "motion_present": True,
            }
        )
        is True
    )
    assert (
        tracking._scene_is_actionable(
            {
                "scene_type": "aerial",
                "tracking_priority": ["vehicle"],
                "dominant_objects": [
                    {
                        "category": "vehicle",
                        "rough_bbox": [0.2, 0.2, 0.8, 0.8],
                        "spatial_hint": "scene-context fallback fallback-bbox",
                    }
                ],
                "areas_of_interest": ["road corridor"],
                "motion_present": True,
            }
        )
        is False
    )
    assert (
        tracking._scene_is_actionable(
            {
                "scene_type": "urban_street|rural_terrain|indoor|aerial|waterway|construction|industrial|other",
                "tracking_priority": ["vehicle"],
                "dominant_objects": [
                    {
                        "category": "vehicle",
                        "rough_bbox": [0.1, 0.1, 0.9, 0.9],
                    }
                ],
            }
        )
        is False
    )
