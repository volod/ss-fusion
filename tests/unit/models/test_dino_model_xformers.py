import sys
from types import ModuleType


def test_set_dino_xformers_enabled_keeps_missing_unbind_disabled(monkeypatch):
    from selfsuvis.models.dino_model import _set_dino_xformers_enabled

    fake = ModuleType("dinov2.fake_attention")
    fake.XFORMERS_AVAILABLE = False
    monkeypatch.setitem(sys.modules, "dinov2.fake_attention", fake)

    _set_dino_xformers_enabled(True)

    assert fake.XFORMERS_AVAILABLE is False
    assert not hasattr(fake, "unbind")


def test_set_dino_xformers_enabled_uses_existing_symbols(monkeypatch):
    from selfsuvis.models.dino_model import _set_dino_xformers_enabled

    fake = ModuleType("dinov2.fake_attention_ready")
    fake.XFORMERS_AVAILABLE = False
    fake.memory_efficient_attention = object()
    fake.unbind = object()
    monkeypatch.setitem(sys.modules, "dinov2.fake_attention_ready", fake)

    _set_dino_xformers_enabled(True)

    assert fake.XFORMERS_AVAILABLE is True
