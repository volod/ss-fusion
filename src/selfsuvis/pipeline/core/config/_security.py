"""Security, API auth, and input-limit settings mixin."""

import os

from ._helpers import _env, _env_int, _parse_allowed_paths


class _SecuritySettings:
    # -- Path-based indexing restrictions -------------------------------------
    ALLOWED_INDEX_PATHS = _parse_allowed_paths(os.getenv("ALLOWED_INDEX_PATHS"))
    MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 2 * 1024 * 1024 * 1024)
    MAX_DOWNLOAD_BYTES = _env_int("MAX_DOWNLOAD_BYTES", 2 * 1024 * 1024 * 1024)
    PRECHECK_URL_TIMEOUT = _env_int("PRECHECK_URL_TIMEOUT", 20)
    MAX_REDIRECTS = _env_int("MAX_REDIRECTS", 5)
    ALLOW_PRIVATE_URLS = _env("ALLOW_PRIVATE_URLS", "false").lower() == "true"

    # -- API authentication and rate limiting ----------------------------------
    API_KEY = _env("API_KEY", "")
    _app_env = _env("APP_ENV", "dev").strip().lower()
    API_AUTH_REQUIRED = (
        _env("API_AUTH_REQUIRED", "true" if _app_env == "prod" else "false").lower() == "true"
    )
    RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 120)
    RATE_LIMIT_BURST = _env_int("RATE_LIMIT_BURST", 60)
    TRUST_PROXY_HEADERS = _env("TRUST_PROXY_HEADERS", "false").lower() == "true"

    # -- Input limits ---------------------------------------------------------
    MAX_IMAGE_PIXELS = _env_int("MAX_IMAGE_PIXELS", 80_000_000)
    MAX_DIR_FILES = _env_int("MAX_DIR_FILES", 5000)
    MAX_DIR_BYTES = _env_int("MAX_DIR_BYTES", 50 * 1024 * 1024 * 1024)
    MAX_DIR_DEPTH = _env_int("MAX_DIR_DEPTH", 10)
