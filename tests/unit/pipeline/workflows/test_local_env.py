import os
from argparse import Namespace

from ssv_vdp.local_env import apply_local_env


def _make_args(**overrides):
    base = dict(
        mode="local",
        output_dir=".data/test-local-env",
        device="cuda",
        fps=2.0,
        asr=None,
        asr_model="openai/whisper-large-v3-turbo",
        asr_language="en",
        ocr=None,
        ocr_model="auto",
        depth=None,
        depth_model="auto",
        detection=None,
        detection_model="auto",
        detection_labels="",
        world_model=None,
        world_model_id="auto",
        qwen=None,
        qwen_api_url="",
        qwen_model="",
        qwen_backend="",
        unidrive=None,
        unidrive_api_url="",
        unidrive_model="",
        unidrive_backend="",
        scenetok=False,
        scenetok_api_url="",
        scenetok_checkpoint="",
        gemma_api_url="",
        gemma_api_model="",
        gemma_api_backend="",
        reasoning_api_url="",
        reasoning_model="",
        reasoning_backend="",
        florence_api_url="",
        florence_model="",
        no_yolo=False,
        yolo_model="yolo11l",
        no_sam=False,
        sam_model="auto",
        no_rfdetr=False,
        rfdetr_model="base",
    )
    base.update(overrides)
    return Namespace(**base)


def test_apply_local_env_sets_both_pytorch_allocator_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    args = _make_args(output_dir=str(tmp_path))
    apply_local_env(args)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_apply_local_env_preserves_existing_pytorch_allocator_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")

    args = _make_args(output_dir=str(tmp_path))
    apply_local_env(args)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"
    assert os.environ["PYTORCH_ALLOC_CONF"] == "max_split_size_mb:64"


def test_apply_local_env_preserves_scenetok_env_when_cli_omits_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENETOK_ENABLED", "true")

    args = _make_args(output_dir=str(tmp_path), scenetok=None)
    apply_local_env(args)

    assert os.environ["SCENETOK_ENABLED"] == "true"


def test_apply_local_env_overrides_scenetok_env_when_cli_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENETOK_ENABLED", "true")

    args = _make_args(output_dir=str(tmp_path), scenetok=False)
    apply_local_env(args)

    assert os.environ["SCENETOK_ENABLED"] == "false"


def test_apply_local_env_sets_scenetok_arg_from_env_when_unspecified(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENETOK_ENABLED", "true")

    args = _make_args(output_dir=str(tmp_path), scenetok=None)
    apply_local_env(args)

    assert args.scenetok is True
