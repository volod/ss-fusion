
import selfsuvis.scripts.prepare_models as pm

def test_resolve_unidrive_backend_fails_without_runtimes(monkeypatch):
    monkeypatch.setattr(pm, "_has_ollama_installed", lambda: False)
    monkeypatch.setattr(pm, "_has_vllm_installed", lambda: False)

    try:
        pm._resolve_unidrive_backend("auto", "owl10/UniDriveVLA_Nusc_Base_Stage3")
    except RuntimeError as exc:
        assert "Install vllm" in str(exc) or "vllm" in str(exc)
        return

    raise AssertionError("expected RuntimeError when neither Ollama nor vllm is installed")


def test_resolve_unidrive_backend_prefers_vllm_for_hf_repo(monkeypatch):
    monkeypatch.setattr(pm, "_has_ollama_installed", lambda: True)
    monkeypatch.setattr(pm, "_has_vllm_installed", lambda: True)

    backend = pm._resolve_unidrive_backend("auto", "owl10/UniDriveVLA_Nusc_Base_Stage3")

    assert backend == "vllm"


def test_resolve_unidrive_prepare_model_maps_hf_repo_to_ollama_fallback():
    model = pm._resolve_unidrive_prepare_model("owl10/UniDriveVLA_Nusc_Base_Stage3", "ollama")

    assert model == "qwen2.5vl:7b"
