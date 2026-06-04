from argparse import Namespace

import selfsuvis.scripts.prepare_models as pm


def test_default_all_if_no_selection_enables_all():
    args = Namespace(
        clip=False,
        dino=False,
        gemma=False,
        flash_attn=False,
        whisper=False,
        florence=False,
        ocr=False,
        depth=False,
        detection=False,
        world_model=False,
        unidrive=False,
        reasoning=False,
        yolo=False,
        sam=False,
        scenetok=False,
        all=False,
    )

    resolved = pm._default_all_if_no_selection(args)

    assert resolved.all is True


def test_resolve_unidrive_backend_fails_without_runtimes(monkeypatch):
    monkeypatch.setattr(pm._ollama, "_has_ollama_installed", lambda: False)
    monkeypatch.setattr(pm._ollama, "_has_vllm_installed", lambda: False)

    try:
        pm._resolve_unidrive_backend("auto", "owl10/UniDriveVLA_Nusc_Base_Stage3")
    except RuntimeError as exc:
        assert "Install vllm" in str(exc) or "vllm" in str(exc)
        return

    raise AssertionError("expected RuntimeError when neither Ollama nor vllm is installed")


def test_resolve_unidrive_backend_prefers_vllm_for_hf_repo(monkeypatch):
    monkeypatch.setattr(pm._ollama, "_has_ollama_installed", lambda: True)
    monkeypatch.setattr(pm._ollama, "_has_vllm_installed", lambda: True)

    backend = pm._resolve_unidrive_backend("auto", "owl10/UniDriveVLA_Nusc_Base_Stage3")

    assert backend == "vllm"


def test_resolve_unidrive_prepare_model_maps_hf_repo_to_ollama_fallback():
    model = pm._resolve_unidrive_prepare_model("owl10/UniDriveVLA_Nusc_Base_Stage3", "ollama")

    assert model == "qwen2.5vl:7b"


def test_normalize_scenetok_checkpoint_accepts_suffix():
    checkpoint = pm._normalize_scenetok_checkpoint_name("va-videodc_dl3dv.ckpt")

    assert checkpoint == "va-videodc_dl3dv.ckpt"


def test_normalize_scenetok_checkpoint_rejects_unknown_variant():
    try:
        pm._normalize_scenetok_checkpoint_name("unknown-checkpoint")
    except ValueError as exc:
        assert "Known checkpoints" in str(exc)
        return

    raise AssertionError("expected ValueError for unknown SceneTok checkpoint")
