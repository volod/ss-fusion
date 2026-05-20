"""Training and active-learning settings mixin: SSL, supervised, edge models."""

import os

from ._helpers import _env, _env_float, _env_int


class _TrainingSettings:
    _data_dir = _env("DATA_DIR", "./.data")

    # -- Self-supervised DINOv3 domain adaptation ------------------------------
    SSL_CHECKPOINT_DIR = _env("SSL_CHECKPOINT_DIR", os.path.join(_data_dir, "checkpoints"))
    SSL_FINETUNE_EPOCHS = _env_int("SSL_FINETUNE_EPOCHS", 10)
    SSL_FINETUNE_LR = _env_float("SSL_FINETUNE_LR", 1e-5)
    SSL_FINETUNE_BATCH_SIZE = _env_int("SSL_FINETUNE_BATCH_SIZE", 32)
    SSL_FINETUNE_FREEZE_BLOCKS = _env_int("SSL_FINETUNE_FREEZE_BLOCKS", 10)
    SSL_FINETUNE_TEMPERATURE = _env_float("SSL_FINETUNE_TEMPERATURE", 0.07)
    # "temporal": consecutive frames from same video dir; "augment": two augmented views
    SSL_FINETUNE_APPROACH = _env("SSL_FINETUNE_APPROACH", "temporal")
    # Path to fine-tuned DINOv3 weights. When set, DINOEmbedder loads this instead
    # of the pretrained hub weights.
    DINO_CHECKPOINT = _env("DINO_CHECKPOINT", "")

    # -- Supervised CVAT fine-tuning -------------------------------------------
    SUP_CHECKPOINT_DIR = _env(
        "SUP_CHECKPOINT_DIR", os.path.join(_data_dir, "checkpoints", "supervised")
    )
    SUP_FINETUNE_EPOCHS = _env_int("SUP_FINETUNE_EPOCHS", 10)
    SUP_FINETUNE_LR = _env_float("SUP_FINETUNE_LR", 1e-5)
    SUP_FINETUNE_BATCH_SIZE = _env_int("SUP_FINETUNE_BATCH_SIZE", 16)
    SUP_FINETUNE_FREEZE_BLOCKS = _env_int("SUP_FINETUNE_FREEZE_BLOCKS", 8)
    SUP_FINETUNE_TEMPERATURE = _env_float("SUP_FINETUNE_TEMPERATURE", 0.07)

    # -- Active learning loop closure ------------------------------------------
    # Whether the CVAT webhook auto-triggers supervised fine-tuning.
    SUP_AUTO_TRIGGER = _env("SUP_AUTO_TRIGGER", "true").lower() == "true"
    MIN_ANNOTATED_FRAMES = _env_int("MIN_ANNOTATED_FRAMES", 50)
    MIN_NEW_ANNOTATED_SINCE_RETRAIN = _env_int("MIN_NEW_ANNOTATED_SINCE_RETRAIN", 100)
    REEMBED_BATCH_SIZE = _env_int("REEMBED_BATCH_SIZE", 256)
    CVAT_API_TOKEN = _env("CVAT_API_TOKEN", "")
    SUP_EVAL_FRACTION = _env_float("SUP_EVAL_FRACTION", 0.1)
    SUP_MIN_PER_CLASS_EVAL = _env_int("SUP_MIN_PER_CLASS_EVAL", 2)
    SUP_MIN_EVAL_GATE_FRAMES = _env_int("SUP_MIN_EVAL_GATE_FRAMES", 20)
    SUP_EVAL_GATE_THRESHOLD = _env_float("SUP_EVAL_GATE_THRESHOLD", 0.6)
    # Intra-vs-inter cosine gap above this value triggers an overfitting warning (not a gate).
    SUP_OVERFITTING_SHIFT_THRESHOLD = _env_float("SUP_OVERFITTING_SHIFT_THRESHOLD", 0.9)

    # -- Drone audio detection dataset and training ----------------------------
    DRONE_AUDIO_DATA_DIR = _env("DRONE_AUDIO_DATA_DIR", os.path.join(_data_dir, "drone-audio-data"))
    DRONE_AUDIO_EPOCHS = _env_int("DRONE_AUDIO_EPOCHS", 10)

    # -- Edge model hydration --------------------------------------------------
    # scripts/export_onnx.py, scripts/build_gallery.py, pipeline/edge_inference.py
    EDGE_MODELS_DIR = _env("EDGE_MODELS_DIR", os.path.join(_data_dir, "models"))
    EDGE_GALLERY_DIR = _env("EDGE_GALLERY_DIR", os.path.join(_data_dir, "gallery"))
    EDGE_ONNX_PATH = _env("EDGE_ONNX_PATH", "")
    EDGE_GALLERY_PATH = _env("EDGE_GALLERY_PATH", "")
    EDGE_TOP_K = _env_int("EDGE_TOP_K", 3)
