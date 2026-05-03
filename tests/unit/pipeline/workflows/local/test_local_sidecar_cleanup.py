
from types import SimpleNamespace

import torch

from selfsuvis.pipeline.vision import unidrive
from selfsuvis.pipeline.workflows.local import steps_caption as sc


class _FakeBackbone:
    def __init__(self) -> None:
        self._device = "cpu"

    def to(self, device):
        if str(device).startswith("cuda"):
            raise RuntimeError("CUDA out of memory")
        self._device = str(device)
        return self

    def cpu(self):
        self._device = "cpu"
        return self

    def parameters(self):
        yield SimpleNamespace(device=torch.device(self._device))


def test_restore_models_to_gpu_returns_false_on_oom(monkeypatch):
    monkeypatch.setattr(sc, "_flush_cuda_allocator", lambda: None)

    models = {"clip": SimpleNamespace(model=_FakeBackbone()), "dino": None}

    restored = sc._restore_models_to_gpu(models, "cuda")

    assert restored is False
    assert sc._models_on_device(models, "cpu") is True


def test_unload_known_sidecars_counts_successful_unique_unloads(monkeypatch):
    calls = []

    def _fake_unload(url: str, model: str) -> bool:
        calls.append((url, model))
        return model == "qwen2.5vl:7b"

    monkeypatch.setattr(sc, "_unload_ollama_model", _fake_unload)

    count = sc._unload_known_sidecars(
        [
            ("http://localhost:11434/v1", "qwen2.5vl:7b"),
            ("http://localhost:11434/v1", "qwen2.5vl:7b"),
            ("http://localhost:8010/v1", "owl10/UniDriveVLA_Nusc_Base_Stage3"),
        ]
    )

    assert count == 1
    assert calls == [
        ("http://localhost:11434/v1", "qwen2.5vl:7b"),
        ("http://localhost:8010/v1", "owl10/UniDriveVLA_Nusc_Base_Stage3"),
    ]


def test_unidrive_backend_auto_detects_ollama_port(monkeypatch):
    monkeypatch.setattr(unidrive.settings, "UNIDRIVE_BACKEND", "vllm")
    monkeypatch.setattr(unidrive.settings, "UNIDRIVE_API_URL", "http://localhost:11434/v1")

    assert unidrive._effective_backend() == "ollama"


def test_guard_min_free_vram_raises_when_headroom_too_low(monkeypatch):
    from selfsuvis.pipeline.vision import registry

    monkeypatch.setattr(registry, "detect_resources", lambda: {"vram_gb": 16.0, "free_vram_gb": 3.5, "ram_gb": 64.0})
    monkeypatch.setattr(sc.settings, "LOCAL_CUDA_STAGE_MIN_FREE_VRAM_GB", 6.0)

    try:
        sc._guard_min_free_vram("SSL fine-tuning")
    except RuntimeError as exc:
        assert "refusing to start CUDA stage" in str(exc)
        return

    raise AssertionError("expected RuntimeError when free VRAM is too low")


def test_restore_models_to_gpu_skips_when_vram_is_too_low(monkeypatch):
    monkeypatch.setattr(sc, "_flush_cuda_allocator", lambda: None)
    monkeypatch.setattr(sc, "_detect_free_vram_gb", lambda: 1.5)

    models = {"clip": SimpleNamespace(model=_FakeBackbone()), "dino": None}

    restored = sc._restore_models_to_gpu(models, "cuda")

    assert restored is False
    assert sc._models_on_device(models, "cpu") is True


def test_select_qwen_frames_keeps_boundaries_and_ocr(monkeypatch):
    frame_list = [(f"frame_{i:02d}.jpg", float(i)) for i in range(10)]
    ocr_map = {2.0: "speed limit", 7.0: "exit"}

    selected = sc._select_qwen_frames(
        frame_list,
        max_frames=5,
        knowledge=None,
        ocr_map=ocr_map,
    )

    selected_ts = [t for _fp, t in selected]
    assert 0.0 in selected_ts
    assert 9.0 in selected_ts
    assert 2.0 in selected_ts
    assert 7.0 in selected_ts
    assert len(selected) == 5


def test_select_qwen_frames_uses_segment_boundaries():
    frame_list = [(f"frame_{i:02d}.jpg", float(i)) for i in range(12)]
    knowledge = SimpleNamespace(
        _segments=[
            {"start_t": 0.0, "end_t": 3.0, "segment_id": 0},
            {"start_t": 4.0, "end_t": 7.0, "segment_id": 1},
            {"start_t": 8.0, "end_t": 11.0, "segment_id": 2},
        ]
    )

    selected = sc._select_qwen_frames(
        frame_list,
        max_frames=6,
        knowledge=knowledge,
        ocr_map={},
    )

    selected_ts = [t for _fp, t in selected]
    assert 0.0 in selected_ts
    assert 4.0 in selected_ts
    assert 8.0 in selected_ts
