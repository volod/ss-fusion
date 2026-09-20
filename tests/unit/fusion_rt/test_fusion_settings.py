"""Unit tests for FusionSettings on KitSettings."""

from selfsuvis.fusion_rt.config import FusionSettings, derive_fusion_database_url, fusion_settings
from ss_kit.settings import KitSettings


def test_fusion_settings_is_kit_settings() -> None:
    assert issubclass(FusionSettings, KitSettings)
    fields = fusion_settings.fields()
    assert "FUSION_DATABASE_URL" in fields
    assert "CORRELATOR_ENABLED" in fields
    assert "STATE_FUSION_ENABLED" in fields
    assert "DATABASE_URL" not in fields
    root = fusion_settings.PROJECT_ROOT
    assert (root / "pyproject.toml").is_file()
    assert (root / "AGENTS.md").is_file()


def test_derive_fusion_database_url_from_video() -> None:
    url = derive_fusion_database_url(
        database_url="postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis",
        fusion_database_url="",
    )
    assert url.endswith("/selfsuvis_fusion")


def test_derive_fusion_database_url_explicit_wins() -> None:
    url = derive_fusion_database_url(
        database_url="postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis",
        fusion_database_url="postgresql://selfsuvis:selfsuvis@localhost:5432/custom_fusion",
    )
    assert url.endswith("/custom_fusion")
