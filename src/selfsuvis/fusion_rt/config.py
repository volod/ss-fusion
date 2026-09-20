"""Fusion-rt and state-estimation settings on KitSettings.

Env var names are unchanged. Perception modules must keep reading video Settings
for RF and training paths; they cannot import this module.
"""

import os
from pathlib import Path

from selfsuvis.pipeline.core.config._helpers import _env, _env_float, _env_int
from selfsuvis.pipeline.core.db_urls import sibling_database_url
from selfsuvis.pipeline.core.env import kit_project_root
from selfsuvis.pipeline.core.logging import get_logger
from ss_kit.settings import KitSettings
from ss_kit.settings import load_layered_env as load_kit_layered_env

_PKG = Path(__file__).resolve().parent.parent
_ROOT = kit_project_root(Path(__file__).resolve().parent)
load_kit_layered_env(
    _ROOT,
    app_env=os.getenv("APP_ENV", "dev"),
    package_env_dir=_PKG / "env",
)

logger = get_logger(__name__)

_DEFAULT_VIDEO_DB = "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis"


def derive_fusion_database_url(database_url: str = "", fusion_database_url: str = "") -> str:
    """Return FUSION_DATABASE_URL, deriving ``.../selfsuvis_fusion`` when unset."""
    explicit = (fusion_database_url or os.environ.get("FUSION_DATABASE_URL", "")).strip()
    if explicit:
        return explicit
    video_url = (database_url or os.environ.get("DATABASE_URL", "")).strip() or _DEFAULT_VIDEO_DB
    return sibling_database_url(video_url, "selfsuvis_fusion")


class FusionSettings(KitSettings):
    """Site-operations and probabilistic fusion knobs. Env names are unchanged."""

    PROJECT_ROOT = _ROOT

    FUSION_DATABASE_URL = derive_fusion_database_url()

    CORRELATOR_REDIS_URL = _env("CORRELATOR_REDIS_URL", "redis://localhost:6379/1")
    WEBHOOK_REDIS_URL = _env("WEBHOOK_REDIS_URL", "redis://localhost:6379/2")
    WEBHOOK_ALERT_URL = _env("WEBHOOK_ALERT_URL", "")
    WEBHOOK_SECRET = _env("WEBHOOK_SECRET", "")
    CORRELATOR_ENABLED = _env("CORRELATOR_ENABLED", "true").lower() == "true"

    DRONE_AUDIO_MODEL_PATH = _env("DRONE_AUDIO_MODEL_PATH", "")
    DRONE_AUDIO_WATCH_DIR = _env("DRONE_AUDIO_WATCH_DIR", "")

    SENSOR_FUSION_ENABLED = _env("SENSOR_FUSION_ENABLED", "true").lower() == "true"
    SENSOR_FUSION_MAX_LAG_MS = _env_int("SENSOR_FUSION_MAX_LAG_MS", 100)

    STATE_FUSION_ENABLED = _env("STATE_FUSION_ENABLED", "true").lower() == "true"
    STATE_FUSION_GPS_POS_STD_M = _env_float("STATE_FUSION_GPS_POS_STD_M", 5.0)
    STATE_FUSION_BARO_ALT_STD_M = _env_float("STATE_FUSION_BARO_ALT_STD_M", 2.5)
    STATE_FUSION_IMU_ACCEL_STD_MPS2 = _env_float("STATE_FUSION_IMU_ACCEL_STD_MPS2", 1.5)
    STATE_FUSION_PROCESS_POS_STD_M = _env_float("STATE_FUSION_PROCESS_POS_STD_M", 0.75)
    STATE_FUSION_PROCESS_VEL_STD_MPS = _env_float("STATE_FUSION_PROCESS_VEL_STD_MPS", 1.5)
    STATE_FUSION_INIT_VEL_STD_MPS = _env_float("STATE_FUSION_INIT_VEL_STD_MPS", 3.0)
    STATE_FUSION_CONTEXT_GAP_SEC = _env_float("STATE_FUSION_CONTEXT_GAP_SEC", 1.0)
    STATE_FUSION_SFM_POS_STD_M = _env_float("STATE_FUSION_SFM_POS_STD_M", 2.0)
    STATE_FUSION_SFM_MIN_FRAMES = _env_int("STATE_FUSION_SFM_MIN_FRAMES", 6)
    OBJECT_FUSION_ENABLED = _env("OBJECT_FUSION_ENABLED", "true").lower() == "true"
    OBJECT_FUSION_OBS_NOISE = _env_float("OBJECT_FUSION_OBS_NOISE", 0.005)
    OBJECT_FUSION_CONFIRM_HITS = _env_int("OBJECT_FUSION_CONFIRM_HITS", 3)
    OBJECT_FUSION_MAX_MISS = _env_int("OBJECT_FUSION_MAX_MISS", 5)
    MAP_FUSION_SMOOTH = _env("MAP_FUSION_SMOOTH", "true").lower() == "true"

    THERMAL_ENABLED = _env("THERMAL_ENABLED", "true").lower() == "true"
    THERMAL_MODEL = _env("THERMAL_MODEL", "")
    LIDAR_ENABLED = _env("LIDAR_ENABLED", "true").lower() == "true"
    GAS_ENABLED = _env("GAS_ENABLED", "true").lower() == "true"
    ACOUSTIC_ENABLED = _env("ACOUSTIC_ENABLED", "true").lower() == "true"


fusion_settings = FusionSettings()


def validate_fusion_settings() -> None:
    """Validate fusion-rt settings. Raises ValueError on invalid config."""
    if fusion_settings.CORRELATOR_ENABLED and not fusion_settings.FUSION_DATABASE_URL:
        raise ValueError("FUSION_DATABASE_URL must be set when CORRELATOR_ENABLED is true")
    logger.info("Fusion settings validated successfully")
