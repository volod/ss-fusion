"""Argument parser for the agentic video processing pipeline CLI."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument(
        "--mode",
        choices=["local", "file", "stream", "analyse"],
        default="local",
        help=(
            "Execution mode: local=full local analysis/train orchestration, "
            "file=lightweight indexing CLI, stream=live stream CLI, "
            "analyse=post-run analytics/report generation"
        ),
    )
    parser.add_argument("--input", help="Video file path")
    parser.add_argument("--dir", help="Directory containing videos")
    parser.add_argument("--output-dir", default=".data/local_runs", help="Output directory")

    parser.add_argument("--interval", type=float, default=1.0, help="Fixed interval in seconds")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive frame sampling",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=1.0,
        help="Minimum interval for adaptive sampling",
    )
    parser.add_argument("--max-gap", type=float, default=10.0, help="Max gap between kept frames")
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.12,
        help="Frame diff threshold",
    )
    parser.add_argument(
        "--probe-fps", type=float, default=5.0, help="Probe fps for adaptive sampling"
    )

    parser.add_argument("--source", help="Stream source (URL or device index)")
    parser.add_argument("--stream-name", help="Name for stream output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames for stream mode")
    parser.add_argument(
        "--steps",
        default="extract,describe,index",
        help="Comma-separated steps: extract,describe,index",
    )
    parser.add_argument(
        "--run-dir",
        help="[analyse] Path to a local run output directory (e.g. .data/local_runs/drone_mission)",
    )
    parser.add_argument(
        "--charts-dir",
        default=None,
        help="[analyse] Directory for individual PNG charts (defaults to --run-dir)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="[analyse] Skip generating the HTML report",
    )
    parser.add_argument(
        "--report-filename",
        default="analysis_report.html",
        help="[analyse] Filename for the HTML report (default: analysis_report.html)",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="[analyse] Optional path to write a compact machine-readable summary JSON",
    )

    parser.add_argument(
        "--model-type",
        choices=["openclip_sam", "openclip_only"],
        default="openclip_sam",
        help="Vision model stack",
    )
    parser.add_argument("--sam-checkpoint", help="Path to SAM checkpoint file")
    parser.add_argument(
        "--sam-model-type",
        default=None,
        help="SAM model type (vit_h/vit_l/vit_b)",
    )
    parser.add_argument("--labels-file", help="Path to label vocabulary file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging per frame",
    )

    parser.add_argument(
        "--es-url",
        help="Elasticsearch base URL, e.g. http://localhost:9200",
    )
    parser.add_argument("--es-index", help="Elasticsearch index name")

    # -- Local orchestration args (used when --mode local) ---------------------
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="[local] Directory containing input .mp4/.mov/.mkv files (default: .data/videos, or DATA_DIR/videos when DATA_DIR is set)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="[local] Torch device (auto selects CUDA when available)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="[local] SSL fine-tuning epochs per video",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="[local] SSL fine-tuning batch size",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="[local] Nearest neighbours to show in search tests",
    )
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="[local] Skip Qdrant; use in-memory cosine search",
    )
    parser.add_argument(
        "--no-sfm",
        action="store_true",
        help="[local] Skip pycolmap SfM; use PCA point-cloud fallback",
    )
    parser.add_argument(
        "--no-gsplat",
        action="store_true",
        help="[local] Skip 3D Gaussian Splatting (step I); keep sparse point-cloud only",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="[local] Skip Florence-2 scene captioning (step L)",
    )
    parser.add_argument(
        "--florence-api-url",
        default="",
        help="[local] vLLM endpoint serving Florence-2 (e.g. http://localhost:8020/v1). "
        "When set, Florence is called via API instead of loading locally — "
        "no local VRAM consumed. See README for vLLM setup instructions.",
    )
    parser.add_argument(
        "--florence-model",
        default="",
        help="[local] Florence-2 model ID for vLLM API (default: microsoft/Florence-2-large)",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=5,
        help="[local] Knowledge distillation epochs (student ViT-S/14)",
    )
    parser.add_argument(
        "--no-distill",
        action="store_true",
        help="[local] Skip knowledge distillation; export teacher to ONNX instead",
    )
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="[local] Skip ONNX export",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="[local] Frame-extraction rate (fps)",
    )
    parser.add_argument(
        "--view-npz",
        metavar="PATH",
        nargs="?",
        const="",
        help="[local] Visualize existing sparse_map.npz without running pipeline",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="[local] Skip the interactive 3D map viewer",
    )
    # Optional multimodal steps
    parser.add_argument(
        "--asr",
        dest="asr",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable Whisper ASR speech-to-text (step M)",
    )
    parser.add_argument(
        "--no-asr",
        dest="asr",
        action="store_const",
        const=False,
        help="[local] Disable Whisper ASR speech-to-text (step M)",
    )
    parser.add_argument("--asr-model", default="auto", help="[local] Whisper model ID or 'auto'")
    parser.add_argument(
        "--asr-language",
        default="",
        help="[local] Force ASR language code (e.g. 'en'). Empty = auto-detect",
    )
    parser.add_argument(
        "--ocr",
        dest="ocr",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable OCR text extraction per frame (step N)",
    )
    parser.add_argument(
        "--no-ocr",
        dest="ocr",
        action="store_const",
        const=False,
        help="[local] Disable OCR text extraction per frame (step N)",
    )
    parser.add_argument("--ocr-model", default="auto", help="[local] OCR model ID or 'auto'")
    parser.add_argument(
        "--depth",
        dest="depth",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable depth estimation per frame (step O)",
    )
    parser.add_argument(
        "--no-depth",
        dest="depth",
        action="store_const",
        const=False,
        help="[local] Disable depth estimation per frame (step O)",
    )
    parser.add_argument("--depth-model", default="auto", help="[local] Depth model ID or 'auto'")
    parser.add_argument(
        "--detection",
        dest="detection",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable object detection per frame (step P)",
    )
    parser.add_argument(
        "--no-detection",
        dest="detection",
        action="store_const",
        const=False,
        help="[local] Disable object detection per frame (step P)",
    )
    parser.add_argument(
        "--detection-model", default="auto", help="[local] Detection model ID or 'auto'"
    )
    parser.add_argument(
        "--detection-labels",
        default="",
        help="[local] Comma-separated labels for open-vocabulary detection",
    )
    # YOLO11 + SAM2/3 detection and segmentation (step P2) — ON by default
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="[local] Disable YOLO11 detection + priority scoring (step P2)",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11l",
        help="[local] YOLO model: 'yolo11l' (default) or yolo11n/yolo11s/yolo11m/yolo11x",
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="[local] Disable SAM2/3 segmentation (detection only; no masks)",
    )
    parser.add_argument(
        "--sam-model",
        default="auto",
        help="[local] SAM backend: 'auto' (tries sam3→sam2→sam1) | 'sam3' | 'sam2' | 'sam1'",
    )
    # Gemma 4 directed tracking: SAM segmentation + RF-DETR tracking (step P3)
    # Enabled automatically when --gemma-api-url is provided; disable with --no-rfdetr.
    parser.add_argument(
        "--no-rfdetr",
        action="store_true",
        help="[local] Disable Gemma directed SAM segmentation + RF-DETR tracking (step P3)",
    )
    parser.add_argument(
        "--rfdetr-model",
        default="base",
        choices=["base", "large"],
        help="[local] RF-DETR model tier: 'base' (faster) or 'large' (higher accuracy)",
    )
    parser.add_argument(
        "--world-model",
        dest="world_model",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable world model video embeddings (step Q)",
    )
    parser.add_argument(
        "--no-world-model",
        dest="world_model",
        action="store_const",
        const=False,
        help="[local] Disable world model video embeddings (step Q)",
    )
    parser.add_argument("--world-model-id", default="auto", help="[local] World model ID or 'auto'")
    parser.add_argument(
        "--qwen",
        dest="qwen",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable Qwen VLM detailed captioning (step R)",
    )
    parser.add_argument(
        "--no-qwen",
        dest="qwen",
        action="store_const",
        const=False,
        help="[local] Disable Qwen VLM detailed captioning (step R)",
    )
    parser.add_argument(
        "--qwen-api-url",
        default="",
        help="[local] Qwen vLLM/ollama endpoint (e.g. http://localhost:8010/v1)",
    )
    parser.add_argument(
        "--qwen-model",
        default="",
        help="[local] Qwen model ID; empty = use QWEN_MODEL env var default",
    )
    parser.add_argument(
        "--qwen-backend",
        default="",
        choices=["", "vllm", "ollama"],
        help="[local] Qwen backend type. Empty = auto-detect",
    )
    parser.add_argument(
        "--unidrive",
        dest="unidrive",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable UniDriveVLA expert analysis (understanding + perception + planning)",
    )
    parser.add_argument(
        "--no-unidrive",
        dest="unidrive",
        action="store_const",
        const=False,
        help="[local] Disable UniDriveVLA expert analysis",
    )
    parser.add_argument(
        "--unidrive-api-url",
        default="",
        help="[local] UniDriveVLA bridge endpoint (OpenAI-compatible /chat/completions)",
    )
    parser.add_argument(
        "--unidrive-model",
        default="",
        help="[local] UniDriveVLA model ID; empty = use UNIDRIVE_MODEL env var default",
    )
    parser.add_argument(
        "--unidrive-backend",
        default="",
        choices=["", "vllm", "ollama"],
        help="[local] UniDrive backend type. Empty = auto-detect",
    )
    parser.add_argument(
        "--scenetok",
        dest="scenetok",
        action="store_const",
        const=True,
        default=None,
        help="[local] Enable SceneTok Step 14 — streaming scene encoder + segmentation decoder (~24 GB VRAM)",
    )
    parser.add_argument(
        "--no-scenetok",
        dest="scenetok",
        action="store_const",
        const=False,
        help="[local] Disable SceneTok",
    )
    parser.add_argument(
        "--scenetok-api-url",
        default="",
        help="[local] SceneTok FastAPI sidecar endpoint; falls back to local torch if unset",
    )
    parser.add_argument(
        "--scenetok-checkpoint",
        default="",
        help="[local] SceneTok checkpoint variant (va-videodc_re10k, va-videodc_dl3dv, va-wan_dl3dv)",
    )
    parser.add_argument(
        "--cosmos3",
        dest="cosmos3",
        action="store_const",
        const=True,
        default=None,
        help=(
            "[local] Enable Cosmos3 Step 15 — omnimodal world-model inference "
            "(nvidia/Cosmos3-Nano, ~32 GB BF16; auto-selects layerwise offload "
            "when free VRAM is 18-40 GB; requires COSMOS3_API_URL for sidecar mode)"
        ),
    )
    parser.add_argument(
        "--no-cosmos3",
        dest="cosmos3",
        action="store_const",
        const=False,
        help="[local] Disable Cosmos3 world-model inference",
    )
    parser.add_argument(
        "--cosmos3-api-url",
        default="",
        help="[local] vLLM-Omni endpoint for Cosmos3 (e.g. http://localhost:8000); skips local load",
    )
    parser.add_argument(
        "--cosmos3-model",
        default="",
        help="[local] Cosmos3 model ID override; empty = auto (nvidia/Cosmos3-Nano based on VRAM)",
    )
    parser.add_argument(
        "--gemma-api-url",
        default="",
        help="[local] Gemma vLLM/ollama endpoint (e.g. http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--gemma-api-model",
        default="",
        help="[local] Gemma model ID; empty = use GEMMA_API_MODEL env var default",
    )
    parser.add_argument(
        "--gemma-api-backend",
        default="",
        choices=["", "vllm", "ollama"],
        help="[local] Gemma backend type. Empty = auto-detect",
    )
    parser.add_argument(
        "--reasoning-api-url",
        default="",
        help="[local] Reasoning endpoint for the final agentic audit step; "
        "empty = reuse Gemma endpoint, then Qwen endpoint",
    )
    parser.add_argument(
        "--reasoning-model",
        default="",
        help="[local] Reasoning model ID for the final agentic audit step; "
        "empty = auto-select from detected hardware",
    )
    parser.add_argument(
        "--reasoning-backend",
        default="",
        choices=["", "vllm", "ollama"],
        help="[local] Reasoning backend type. Empty = auto-detect",
    )
    parser.add_argument(
        "--drone-detection",
        dest="drone_detection",
        action="store_const",
        const=True,
        default=None,
        help="[local] Train a YOLOv8n drone detector and export edge models "
        "(ONNX fp32 for Cortex-A76, int8 for RV1106G3). "
        "Downloads seraphim-drone-detection-dataset subset from HuggingFace.",
    )
    parser.add_argument(
        "--no-drone-detection",
        dest="drone_detection",
        action="store_const",
        const=False,
        help="[local] Skip drone detection training (step 30)",
    )
    parser.add_argument(
        "--drone-audio",
        dest="drone_audio",
        action="store_const",
        const=True,
        default=None,
        help="[local] Train DroneAudioCNN on drone-audio-detection-samples "
        "and export ONNX (step 32). Dataset cached in .data/drone-audio-data/. "
        "Prepare with: scripts/split_drone_audio_data.sh",
    )
    parser.add_argument(
        "--no-drone-audio",
        dest="drone_audio",
        action="store_const",
        const=False,
        help="[local] Skip drone audio detection training (step 32)",
    )
    parser.add_argument(
        "--drone-audio-epochs",
        type=int,
        default=None,
        help="[local] DroneAudioCNN training epochs (default: DRONE_AUDIO_EPOCHS env var, or 10)",
    )
    parser.add_argument(
        "--drau-eval",
        dest="drau_eval",
        action="store_const",
        const=True,
        default=None,
        help="[local] Run drau range-detection evaluation (step 33). Evaluates "
        "DroneAudioCNN ONNX model at simulated distances using the drau physics "
        "model (github.com/volod/drau). Requires --drone-audio to have produced "
        "drone_audio_cnn.onnx. Exports drau_edge_test.py for edge deployment.",
    )
    parser.add_argument(
        "--no-drau-eval",
        dest="drau_eval",
        action="store_const",
        const=False,
        help="[local] Skip drau range-detection evaluation (step 33)",
    )

    return parser
