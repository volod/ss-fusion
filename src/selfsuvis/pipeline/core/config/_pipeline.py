"""Pipeline signal-processing settings mixin: sampling, tiles, dedup, SfM, AL."""

from ._helpers import _env, _env_float, _env_int


class _PipelineSettings:
    SAMPLE_FPS_BASE = float(_env("SAMPLE_FPS_BASE", "2"))
    SAMPLE_FPS_MIN = float(_env("SAMPLE_FPS_MIN", "0.5"))
    SAMPLE_FPS_MAX = float(_env("SAMPLE_FPS_MAX", "5"))

    HIST_THRESH = float(_env("HIST_THRESH", "0.25"))
    EMBED_DRIFT_THRESH = float(_env("EMBED_DRIFT_THRESH", "0.15"))
    MAX_GAP_SEC = float(_env("MAX_GAP_SEC", "10"))

    MOTION_LOW = float(_env("MOTION_LOW", "0.02"))
    MOTION_HIGH = float(_env("MOTION_HIGH", "0.08"))

    STAB_ENABLE = _env("STAB_ENABLE", "true").lower() == "true"
    STAB_SIZE = int(_env("STAB_SIZE", "64"))
    PHASECORR_MIN_RESPONSE = float(_env("PHASECORR_MIN_RESPONSE", "0.15"))
    STAB_MAX_SHIFT = float(_env("STAB_MAX_SHIFT", "12"))

    TILE_SIZE = int(_env("TILE_SIZE", "384"))
    STRIDE = int(_env("STRIDE", "256"))

    BLUR_LAPL_VAR_MIN_FRAME = float(_env("BLUR_LAPL_VAR_MIN_FRAME", "80"))
    BLUR_LAPL_VAR_MIN_TILE = float(_env("BLUR_LAPL_VAR_MIN_TILE", "60"))
    MEAN_INTENSITY_MIN = float(_env("MEAN_INTENSITY_MIN", "20"))
    MEAN_INTENSITY_MAX = float(_env("MEAN_INTENSITY_MAX", "235"))

    SKY_BLUE_RATIO_MAX = float(_env("SKY_BLUE_RATIO_MAX", "0.35"))
    EDGE_DENSITY_MIN = float(_env("EDGE_DENSITY_MIN", "0.02"))

    TILE_STD_MIN = float(_env("TILE_STD_MIN", "12"))
    TILE_ENTROPY_MIN = float(_env("TILE_ENTROPY_MIN", "3.5"))

    CELL_SIZE = int(_env("CELL_SIZE", str(STRIDE)))  # default mirrors STRIDE
    CELL_WINDOW_SEC = float(_env("CELL_WINDOW_SEC", "5"))

    PHASH_LRU_SIZE = int(_env("PHASH_LRU_SIZE", "50000"))
    PHASH_HAMMING_MAX = int(_env("PHASH_HAMMING_MAX", "6"))

    DEDUP_RECENT_TILES = int(_env("DEDUP_RECENT_TILES", "200000"))
    DEDUP_TTL_SEC = float(_env("DEDUP_TTL_SEC", "120"))
    DEDUP_COS_SIM_THRESH = float(_env("DEDUP_COS_SIM_THRESH", "0.95"))

    TILE_INDEX_IF_EMBED_DRIFT_GT = float(_env("TILE_INDEX_IF_EMBED_DRIFT_GT", "0.10"))
    MAX_TILES_PER_SEGMENT = int(_env("MAX_TILES_PER_SEGMENT", "200"))

    K_RETRIEVE = int(_env("K_RETRIEVE", "100"))
    K_RETURN = int(_env("K_RETURN", "20"))

    # -- SfM and 3DGS ----------------------------------------------------------
    SFM_FPS = _env_float("SFM_FPS", 2.0)
    # Clips shorter than SFM_MIN_DURATION_SEC skip SfM and go straight to the
    # PCA/semantic-pseudo3d fallback -- SfM on a 10s clip recovers too few poses.
    SFM_MIN_DURATION_SEC = _env_float("SFM_MIN_DURATION_SEC", 30.0)
    PYCOLMAP_CAMERA_MODEL = _env("PYCOLMAP_CAMERA_MODEL", "SIMPLE_RADIAL")
    PYCOLMAP_SINGLE_CAMERA = _env("PYCOLMAP_SINGLE_CAMERA", "true").lower() == "true"
    PYCOLMAP_MAX_IMAGE_SIZE = _env_int("PYCOLMAP_MAX_IMAGE_SIZE", 1920)
    PYCOLMAP_NUM_THREADS = _env_int("PYCOLMAP_NUM_THREADS", 8)
    PYCOLMAP_MATCHING = _env("PYCOLMAP_MATCHING", "sequential")
    PYCOLMAP_SEQUENTIAL_OVERLAP = _env_int("PYCOLMAP_SEQUENTIAL_OVERLAP", 8)
    PYCOLMAP_MIN_LOG_LEVEL = _env_int("PYCOLMAP_MIN_LOG_LEVEL", 2)
    PYCOLMAP_INIT_MIN_NUM_INLIERS = _env_int("PYCOLMAP_INIT_MIN_NUM_INLIERS", 50)
    PYCOLMAP_INIT_MIN_TRI_ANGLE = _env_float("PYCOLMAP_INIT_MIN_TRI_ANGLE", 4.0)
    PYCOLMAP_INIT_MAX_FORWARD_MOTION = _env_float("PYCOLMAP_INIT_MAX_FORWARD_MOTION", 0.99)
    PYCOLMAP_ABS_POSE_MIN_INLIER_RATIO = _env_float("PYCOLMAP_ABS_POSE_MIN_INLIER_RATIO", 0.15)

    # -- Active learning -------------------------------------------------------
    AL_TAG_K = _env_int("AL_TAG_K", 50)
    # Switch from KMeans to MiniBatchKMeans when total embedded frames exceed this.
    KMEANS_BATCH_THRESHOLD = _env_int("KMEANS_BATCH_THRESHOLD", 25_000)

    # -- Change detection ------------------------------------------------------
    CHANGE_DETECTION_THRESHOLD_CLIP = _env_float("CHANGE_DETECTION_THRESHOLD_CLIP", 0.35)
    CHANGE_DETECTION_THRESHOLD_DINO = _env_float("CHANGE_DETECTION_THRESHOLD_DINO", 0.25)
