"""External service settings mixin: Qdrant, DB, Realtime, MediaMTX, Redis, CVAT."""

import os

from ._helpers import _env, _env_float, _env_int, _env_json_dict


class _ServiceSettings:
    # -- Qdrant vector store ---------------------------------------------------
    QDRANT_HOST = _env("QDRANT_HOST", "qdrant")
    QDRANT_PORT = int(_env("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "video_semantic")

    # -- PostgreSQL ------------------------------------------------------------
    DATABASE_URL = _env("DATABASE_URL", "")

    # -- Realtime ingest / autonomy scaffolding --------------------------------
    REALTIME_ENABLED = _env("REALTIME_ENABLED", "false").lower() == "true"
    REALTIME_BACKEND = _env("REALTIME_BACKEND", "stub")
    REALTIME_POSE_BACKEND = _env("REALTIME_POSE_BACKEND", "stub")
    REALTIME_POSE_API_URL = _env("REALTIME_POSE_API_URL", "")
    REALTIME_OCCUPANCY_BACKEND = _env("REALTIME_OCCUPANCY_BACKEND", "stub")
    REALTIME_OCCUPANCY_API_URL = _env("REALTIME_OCCUPANCY_API_URL", "")
    REALTIME_PACKET_BATCH_SIZE = _env_int("REALTIME_PACKET_BATCH_SIZE", 128)
    REALTIME_MAX_SENSOR_LAG_MS = _env_int("REALTIME_MAX_SENSOR_LAG_MS", 120)
    REALTIME_SESSION_TIMEOUT_SEC = _env_int("REALTIME_SESSION_TIMEOUT_SEC", 30)
    REALTIME_OCCUPANCY_RESOLUTION_M = _env_float("REALTIME_OCCUPANCY_RESOLUTION_M", 0.2)
    REALTIME_BRIDGE_ENABLED = _env("REALTIME_BRIDGE_ENABLED", "false").lower() == "true"
    REALTIME_BRIDGE_BACKEND = _env("REALTIME_BRIDGE_BACKEND", "mavsdk")
    REALTIME_BRIDGE_SESSION_ID = _env("REALTIME_BRIDGE_SESSION_ID", "realtime_bridge")
    REALTIME_BRIDGE_ROBOT_ID = _env("REALTIME_BRIDGE_ROBOT_ID", "drone_bridge")
    REALTIME_BRIDGE_MISSION_ID = _env("REALTIME_BRIDGE_MISSION_ID", "")
    REALTIME_BRIDGE_SENSORS = _env("REALTIME_BRIDGE_SENSORS", "gps,imu,barometer,magnetometer")
    REALTIME_BRIDGE_AUTO_CREATE_SESSION = (
        _env("REALTIME_BRIDGE_AUTO_CREATE_SESSION", "true").lower() == "true"
    )
    REALTIME_BRIDGE_FLUSH_INTERVAL_SEC = _env_float("REALTIME_BRIDGE_FLUSH_INTERVAL_SEC", 0.5)
    REALTIME_BRIDGE_RECONNECT_SEC = _env_float("REALTIME_BRIDGE_RECONNECT_SEC", 5.0)
    REALTIME_BRIDGE_LOG_EVERY_N_PACKETS = _env_int("REALTIME_BRIDGE_LOG_EVERY_N_PACKETS", 100)
    REALTIME_MAVSDK_SYSTEM_ADDRESS = _env("REALTIME_MAVSDK_SYSTEM_ADDRESS", "udp://:14540")
    REALTIME_MAVSDK_SERVER_ADDRESS = _env("REALTIME_MAVSDK_SERVER_ADDRESS", "")
    REALTIME_MAVSDK_SERVER_PORT = _env_int("REALTIME_MAVSDK_SERVER_PORT", 50051)
    REALTIME_MAVSDK_CONNECT_TIMEOUT_SEC = _env_float("REALTIME_MAVSDK_CONNECT_TIMEOUT_SEC", 20.0)
    REALTIME_ROS_DOMAIN_ID = _env("REALTIME_ROS_DOMAIN_ID", "")
    REALTIME_ROS_IMU_TOPIC = _env("REALTIME_ROS_IMU_TOPIC", "/imu")
    REALTIME_ROS_GPS_TOPIC = _env("REALTIME_ROS_GPS_TOPIC", "/gps/fix")
    REALTIME_ROS_BAROMETER_TOPIC = _env("REALTIME_ROS_BAROMETER_TOPIC", "/barometer")
    REALTIME_ROS_MAG_TOPIC = _env("REALTIME_ROS_MAG_TOPIC", "/mag")
    REALTIME_ROS_CAMERA_TOPIC = _env("REALTIME_ROS_CAMERA_TOPIC", "")

    # -- MediaMTX (RTSP/WebRTC proxy) ------------------------------------------
    MEDIAMTX_API_URL = _env("MEDIAMTX_API_URL", "http://mediamtx:9997")
    MEDIAMTX_API_USER = _env("MEDIAMTX_API_USER", "")
    MEDIAMTX_API_PASS = _env("MEDIAMTX_API_PASS", "")
    MEDIAMTX_RTSP_BASE_URL = _env("MEDIAMTX_RTSP_BASE_URL", "rtsp://mediamtx:8554")
    MEDIAMTX_PUBLIC_RTSP_BASE_URL = _env("MEDIAMTX_PUBLIC_RTSP_BASE_URL", "rtsp://localhost:8554")

    # -- Reports and maps output directories -----------------------------------
    _data_dir = _env("DATA_DIR", "./data")
    REPORTS_DIR = _env("REPORTS_DIR", os.path.join(_data_dir, "reports"))
    MAPS_DIR = _env("MAPS_DIR", os.path.join(_data_dir, "maps"))

    # -- External service URLs -------------------------------------------------
    STATIC_SERVER_URL = _env("STATIC_SERVER_URL", "http://localhost:8080")
    SUPERSPLAT_SERVER_URL = _env("SUPERSPLAT_SERVER_URL", "http://localhost:8090")
    NERFSTUDIO_API_URL = _env("NERFSTUDIO_API_URL", "http://nerfstudio:8000")
    # ICP fusion mapper service (docker-compose.override.yml, port 8100 on host)
    MAPPER_API_URL = _env("MAPPER_API_URL", "http://mapper:8000")

    # -- CVAT annotation service -----------------------------------------------
    # Runs at http://localhost:8091 via make cvat-up
    CVAT_URL = _env("CVAT_URL", "http://localhost:8091")
    # HMAC-SHA256 secret for X-Hook-Secret header verification.
    # Must match the secret in CVAT webhook settings.
    # When empty, /webhook/cvat rejects all requests (fail-closed).
    CVAT_WEBHOOK_SECRET = _env("CVAT_WEBHOOK_SECRET", "")
    # JSON dict mapping CVAT label names to canonical vocabulary.
    # Applied in AnnotatedFrameDataset.from_xml() and from_db() to normalize
    # label names across CVAT tasks before building the class vocabulary.
    CVAT_LABEL_MAPPINGS: dict = _env_json_dict("CVAT_LABEL_MAPPINGS", {})

    # -- Redis -----------------------------------------------------------------
    # Shared base URL -- kept for any code that hasn't migrated to per-consumer URLs.
    REDIS_URL = _env("REDIS_URL", "redis://localhost:6379/0")
    # Per-consumer endpoints default to separate DBs on the same instance for key isolation.
    CORRELATOR_REDIS_URL = _env("CORRELATOR_REDIS_URL", "redis://localhost:6379/1")
    WEBHOOK_REDIS_URL = _env("WEBHOOK_REDIS_URL", "redis://localhost:6379/2")
    HEALTH_REDIS_URL = _env("HEALTH_REDIS_URL", "redis://localhost:6379/3")
    WEBHOOK_ALERT_URL = _env("WEBHOOK_ALERT_URL", "")
    WEBHOOK_SECRET = _env("WEBHOOK_SECRET", "")
    CORRELATOR_ENABLED = _env("CORRELATOR_ENABLED", "true").lower() == "true"

    # -- DroneAudioAdapter runtime paths ---------------------------------------
    # DRONE_AUDIO_MODEL_PATH: path to ONNX from SV-21 training. Empty = disabled.
    # DRONE_AUDIO_WATCH_DIR: directory to poll for .wav files. Empty = disabled.
    DRONE_AUDIO_MODEL_PATH = _env("DRONE_AUDIO_MODEL_PATH", "")
    DRONE_AUDIO_WATCH_DIR = _env("DRONE_AUDIO_WATCH_DIR", "")

    # -- Worker and ffmpeg -----------------------------------------------------
    WORKER_POLL_INTERVAL = _env_float("WORKER_POLL_INTERVAL", 2.0)
    FFMPEG_TIMEOUT_SEC = _env_int("FFMPEG_TIMEOUT_SEC", 3600)
    # Maximum RTSP recording duration in seconds.
    RTSP_MAX_DURATION_SEC = _env_int("RTSP_MAX_DURATION_SEC", 3600)
    # RTSP live captioner: frames per second to sample from the stream.
    RTSP_CAPTION_FPS = _env_float("RTSP_CAPTION_FPS", 0.5)
