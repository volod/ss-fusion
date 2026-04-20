from PIL import Image

from selfsuvis.pipeline.vision import depth


def test_resolve_model_id_prefers_fast_profile_for_auto(monkeypatch):
    monkeypatch.setattr(depth.settings, "DEPTH_MODEL", "auto")
    monkeypatch.setattr(depth.settings, "DEPTH_AUTO_PROFILE", "fast")

    assert depth._resolve_model_id() == "depth-anything/Depth-Anything-V2-Base-hf"


def test_resolve_model_id_honors_explicit_model(monkeypatch):
    monkeypatch.setattr(depth.settings, "DEPTH_MODEL", "apple/DepthPro-hf")
    monkeypatch.setattr(depth.settings, "DEPTH_AUTO_PROFILE", "fast")

    assert depth._resolve_model_id() == "apple/DepthPro-hf"


def test_prepare_depth_image_resizes_oversized_input(monkeypatch):
    monkeypatch.setattr(depth.settings, "DEPTH_IMAGE_MAX_SIDE", 768)

    image = Image.new("RGB", (1920, 1080))
    resized = depth._prepare_depth_image(image)

    assert max(resized.size) == 768


def test_prepare_depth_image_keeps_small_input(monkeypatch):
    monkeypatch.setattr(depth.settings, "DEPTH_IMAGE_MAX_SIDE", 768)

    image = Image.new("RGB", (640, 360))
    resized = depth._prepare_depth_image(image)

    assert resized.size == image.size
