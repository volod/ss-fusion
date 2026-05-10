from argparse import Namespace


def _make_args(tmp_path, **overrides):
    base = dict(
        output_dir=str(tmp_path),
        asr=False,
        asr_model="openai/whisper-large-v3-turbo",
        ocr=False,
        ocr_model="auto",
        depth=False,
        depth_model="auto",
        detection=False,
        detection_model="auto",
        world_model=False,
        world_model_id="auto",
        qwen=False,
        qwen_api_url="",
        qwen_model="",
        qwen_backend="",
        unidrive=False,
        unidrive_api_url="",
        unidrive_model="",
        unidrive_backend="",
        scenetok=False,
        drone_detection=False,
        gemma_api_url="",
        gemma_api_model="",
        gemma_api_backend="",
        reasoning_api_url="",
        reasoning_model="",
        reasoning_backend="",
    )
    base.update(overrides)
    return Namespace(**base)


def test_run_local_preflight_reports_missing_cached_models(monkeypatch, tmp_path):
    from selfsuvis.pipeline.core.preflight import run_local_preflight

    monkeypatch.setattr("selfsuvis.pipeline.core.preflight._has_module", lambda _name: True)
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._is_openclip_cached",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._is_dino_hub_cached",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight._check_tcp_service",
        lambda report, *_args, **_kwargs: report.add_warning("service check skipped"),
    )

    report = run_local_preflight(_make_args(tmp_path))

    assert any("OpenCLIP" in item for item in report.errors)
    assert any("DINOv2/v3" in item for item in report.errors)


def test_run_local_preflight_checks_drone_detection_runtime_bits(monkeypatch, tmp_path):
    from selfsuvis.pipeline.core.preflight import run_local_preflight

    monkeypatch.setattr("selfsuvis.pipeline.core.preflight._has_module", lambda _name: True)
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._is_openclip_cached",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._is_dino_hub_cached",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._is_yolo_cached",
        lambda model: model == "yolov8n",
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight._check_tcp_service",
        lambda report, *_args, **_kwargs: report.add_check("service check skipped"),
    )

    report = run_local_preflight(_make_args(tmp_path, detection=True, drone_detection=True))

    assert not any("YOLOv8n training weights" in item for item in report.errors)
    assert any("dataset cache is empty" in item for item in report.warnings)


def test_resolve_auto_model_expands_auto_for_hf_tasks(monkeypatch):
    from selfsuvis.pipeline.core.preflight import _resolve_auto_model

    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._resolve_hf_model",
        lambda task, override: f"{task}-resolved:{override or 'empty'}",
    )

    assert _resolve_auto_model("ocr", "auto") == "ocr-resolved:empty"


def test_run_local_preflight_downgrades_scenetok_to_warning_on_small_gpu(monkeypatch, tmp_path):
    from selfsuvis.pipeline.core.preflight import run_local_preflight

    monkeypatch.setattr("selfsuvis.pipeline.core.preflight._has_module", lambda _name: True)
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._is_openclip_cached",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.model_prep._is_dino_hub_cached",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight._check_tcp_service",
        lambda report, *_args, **_kwargs: report.add_check("service check skipped"),
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.core.preflight.settings",
        type(
            "S",
            (),
            {
                **{
                    k: getattr(
                        __import__(
                            "selfsuvis.pipeline.core.preflight", fromlist=["settings"]
                        ).settings,
                        k,
                    )
                    for k in dir(
                        __import__(
                            "selfsuvis.pipeline.core.preflight", fromlist=["settings"]
                        ).settings
                    )
                    if k.isupper()
                },
                "SCENETOK_ENABLED": True,
                "SCENETOK_API_URL": "",
            },
        )(),
    )
    monkeypatch.setattr(
        "selfsuvis.pipeline.vision.registry.detect_vram_gb",
        lambda: 16.0,
    )

    report = run_local_preflight(_make_args(tmp_path, scenetok=True))

    assert not any("scenetok" in item.lower() for item in report.errors)
    assert any(
        "SceneTok is enabled but local execution is not possible" in item
        for item in report.warnings
    )
