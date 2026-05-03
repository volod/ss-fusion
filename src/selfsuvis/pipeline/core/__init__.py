"""Core runtime primitives shared across the pipeline."""

from .config import get_dino_model_name, settings, validate_settings
from .docker import REALTIME_ENGINE_IMAGES, DockerImageRef
from .env import (
    env_bool,
    env_csv,
    env_float,
    env_int,
    env_json_dict,
    env_str,
    load_layered_env,
    set_env_if_present,
)
from .gpu_utils import is_cuda_oom, pipeline_device_arg, resolve_device
from .log_analytics import get_log_analytics
from .logging import configure_logging, get_logger
from .preflight import log_preflight, run_local_preflight, run_production_preflight
from .sidecars import HttpSidecarClient, load_jsonl_sidecar, load_video_jsonl_sidecar, sidecar_path
from .utils import (
    RateTimer,
    clamp,
    datetime_to_ts,
    ensure_dir,
    file_sha256,
    now_iso,
    resolve_allowed_path,
    resolve_allowed_paths_for_walk,
    stable_point_id,
    to_utc_datetime,
    utcnow,
)

__all__ = [
    "RateTimer",
    "clamp",
    "configure_logging",
    "datetime_to_ts",
    "DockerImageRef",
    "ensure_dir",
    "env_bool",
    "env_csv",
    "env_float",
    "env_int",
    "env_json_dict",
    "env_str",
    "file_sha256",
    "get_log_analytics",
    "get_dino_model_name",
    "get_logger",
    "HttpSidecarClient",
    "is_cuda_oom",
    "load_jsonl_sidecar",
    "log_preflight",
    "load_layered_env",
    "load_video_jsonl_sidecar",
    "now_iso",
    "pipeline_device_arg",
    "REALTIME_ENGINE_IMAGES",
    "resolve_allowed_path",
    "resolve_allowed_paths_for_walk",
    "resolve_device",
    "run_local_preflight",
    "run_production_preflight",
    "set_env_if_present",
    "sidecar_path",
    "settings",
    "stable_point_id",
    "to_utc_datetime",
    "utcnow",
    "validate_settings",
]
