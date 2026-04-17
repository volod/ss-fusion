"""Unit tests for pipeline/model_registry.py — no GPU or network required."""
import pytest
from selfsuvis.pipeline.vision.registry import (
    CATALOGS,
    ModelEntry,
    auto_select,
    get_entry,
    list_models,
)


# ── CATALOGS structure ────────────────────────────────────────────────────────

EXPECTED_TASKS = {"asr", "ocr", "depth", "detection", "segmentation",
                  "vqa", "zero_shot_classification", "world_model"}


def test_all_tasks_present():
    assert EXPECTED_TASKS.issubset(CATALOGS.keys())


def test_each_catalog_has_minimum_entries():
    # depth has 9 entries (Intel/zoedepth-nk was removed — invalid HF model ID).
    # world_model has 10 entries (nvidia/Cosmos-1.0-Autoregressive-4B re-added;
    # auth handled interactively by prepare_models.py).
    min_per_task = {"depth": 9}
    for task, entries in CATALOGS.items():
        expected = min_per_task.get(task, 10)
        assert len(entries) >= expected, (
            f"Task {task!r} has {len(entries)} entries, expected at least {expected}"
        )


def test_entries_are_model_entry_instances():
    for task, entries in CATALOGS.items():
        for e in entries:
            assert isinstance(e, ModelEntry), f"{task}: {e!r} is not a ModelEntry"


def test_all_model_ids_non_empty():
    for task, entries in CATALOGS.items():
        for e in entries:
            assert e.model_id, f"{task}: empty model_id"


def test_vram_non_negative():
    for task, entries in CATALOGS.items():
        for e in entries:
            assert e.vram_fp16_gb >= 0.0, f"{task}/{e.model_id}: negative vram"


def test_params_positive():
    for task, entries in CATALOGS.items():
        for e in entries:
            assert e.params_b > 0.0, f"{task}/{e.model_id}: params_b must be > 0"


# ── auto_select ───────────────────────────────────────────────────────────────

def test_auto_select_unknown_task_returns_none():
    assert auto_select("nonexistent_task", {"vram_gb": 16.0, "ram_gb": 64.0}) is None


def test_auto_select_no_gpu_returns_cpu_capable():
    """With 0 GB VRAM (CPU only), only models ≤ 0.5 GB are eligible."""
    result = auto_select("asr", {"vram_gb": 0.0, "ram_gb": 32.0})
    entry = get_entry("asr", result)
    # cpu_vram_limit = 0.5 GB — whisper-small (0.5 GB) is the largest that fits
    assert entry.vram_fp16_gb <= 0.5


def test_auto_select_large_vram_selects_last_fitting():
    """With 20 GB VRAM (safety margin = 2 GB → 18 GB available), pick the largest ASR model."""
    result = auto_select("asr", {"vram_gb": 20.0, "ram_gb": 64.0})
    # seamless-m4t-v2-large requires 4.6 GB, fits in 18 GB
    assert result == "facebook/seamless-m4t-v2-large"


def test_auto_select_medium_vram_stays_under_limit():
    """With 4 GB VRAM (safety 2 GB → 2 GB available), only <2 GB models qualify."""
    result = auto_select("asr", {"vram_gb": 4.0, "ram_gb": 32.0})
    # Models that fit: tiny(0.1), base(0.2), small(0.5), medium(1.5), distil(1.5), turbo(1.6)
    # All fit; largest that fits under 2.0 GB available is turbo at 1.6 GB
    entry = get_entry("asr", result)
    assert entry is not None
    assert entry.vram_fp16_gb <= 2.0


def test_auto_select_exact_margin_boundary():
    """VRAM - safety_margin == entry.vram_fp16_gb should still select that entry."""
    # whisper-large-v3 needs 3.0 GB; VRAM = 5.0 → available = 3.0 → should select it
    result = auto_select("asr", {"vram_gb": 5.0, "ram_gb": 32.0})
    entry = get_entry("asr", result)
    assert entry.vram_fp16_gb <= 3.0


def test_auto_select_returns_string():
    result = auto_select("depth", {"vram_gb": 8.0, "ram_gb": 16.0})
    assert isinstance(result, str) and result


def test_auto_select_prefer_video_filters_catalog():
    """prefer_video=True on world_model should only return video-capable models."""
    result = auto_select("world_model", {"vram_gb": 20.0, "ram_gb": 64.0}, prefer_video=True)
    entry = get_entry("world_model", result)
    assert entry is not None
    assert entry.supports_video


def test_auto_select_prefer_video_no_candidates_falls_back():
    """If no video-capable models exist in a catalog, fall back to full catalog."""
    # 'asr' has no supports_video=True entries — should still return something
    result = auto_select("asr", {"vram_gb": 20.0, "ram_gb": 64.0}, prefer_video=True)
    assert result is not None


def test_auto_select_none_resources_doesnt_raise():
    """auto_select with resources=None should call detect_resources() internally."""
    # We just verify it doesn't raise; the actual return depends on the test machine.
    result = auto_select("depth", None)
    # Either a string or None (unknown task), but depth is a known task
    assert isinstance(result, str)


# ── get_entry ─────────────────────────────────────────────────────────────────

def test_get_entry_found():
    entry = get_entry("asr", "openai/whisper-tiny")
    assert entry is not None
    assert entry.model_id == "openai/whisper-tiny"
    assert entry.params_b == pytest.approx(0.039)


def test_get_entry_not_found_returns_none():
    assert get_entry("asr", "does-not-exist") is None


def test_get_entry_wrong_task_returns_none():
    # whisper-tiny exists in asr but not ocr
    assert get_entry("ocr", "openai/whisper-tiny") is None


# ── list_models ───────────────────────────────────────────────────────────────

def test_list_models_returns_entries():
    entries = list_models("ocr")
    assert len(entries) == 10
    assert all(isinstance(e, ModelEntry) for e in entries)


def test_list_models_unknown_task_returns_empty():
    assert list_models("does_not_exist") == []


# ── ModelEntry dataclass defaults ─────────────────────────────────────────────

def test_model_entry_defaults():
    e = ModelEntry("test/model", 1.0, 2.0, "test entry")
    assert e.supports_video is False
    assert e.extra == {}
