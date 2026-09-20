# Multimodal Model Catalog — selfsuvis Pipeline

Comprehensive catalog of multimodal models integrated into (or available for)
the selfsuvis pipeline.  Each section lists the **top-10 models ordered by
parameter count (small → large)**, with VRAM requirements, a one-line
description, and the CLI env var to activate each.

GPU-aware auto-selection runs at startup when a model is set to `"auto"` —
it reads `nvidia-smi memory.total` and picks the largest model that fits
within available VRAM minus a 2 GB safety margin.

---

## Pipeline Model Map

```
Video input
   │
   ├─ Pass 0 (per frame, main loop)
   │     CLIP  →  embedding + tile dedup           [always on]
   │     DINOv3 →  embedding (if MODEL_NAME=dinov3) [always on if configured]
   │
   ├─ Pass ASR  (once per video, before captioning)
   │     Whisper → subtitle_text per frame          [ASR_ENABLED=true]
   │
   ├─ Pass Florence (batch, post-loop)
   │     Florence-2 → caption, caption_confidence   [always on]
   │
   ├─ Pass OCR  (batch, post-Florence)
   │     TrOCR / GOT-OCR2 / DeepSeek-OCR-2 → ocr_text  [OCR_ENABLED=true]
   │
   ├─ Pass Qwen (per frame, post-OCR)
   │     Qwen2.5-VL-7B → frame_facts_json          [QWEN_API_URL set]
   │     + injects subtitle_text + ocr_text as context
   │
   ├─ Pass Depth (per frame)
   │     DepthAnything-V2 → depth percentiles       [DEPTH_ENABLED=true]
   │
   ├─ Pass Detection (batch)
   │     RT-DETR / Grounding DINO → detections      [DETECTION_ENABLED=true]
   │
   └─ Pass World Model (clip windows)
         VideoMAE / Cosmos → clip embedding          [WORLD_MODEL_ENABLED=true]
```

---

## 1. ASR — Automatic Speech Recognition

Extracts audio track from video (16 kHz mono WAV via ffmpeg) and transcribes
to time-stamped subtitle segments.  Subtitles are mapped to frame timestamps
(±`ASR_SUBTITLE_WINDOW_SEC`, default 3 s) and stored in `frames.subtitle_text`.

The subtitle text is injected into the Qwen2.5-VL prompt as
`[Audio context at this moment]: ...` for richer structured scene extraction.

> **Note:** Current ASR models top out at ~2.3 B parameters.  No 3B+ ASR
> models exist as of 2026-Q1; the Whisper family dominates the field.

| # | Model ID | Params | VRAM (FP16) | Notes |
|---|---|---|---|---|
| 1 | `openai/whisper-tiny` | 39 M | ~0.1 GB | Fastest; English-focused; ~32× real-time |
| 2 | `openai/whisper-base` | 74 M | ~0.2 GB | Good speed/quality balance; 99 languages |
| 3 | `openai/whisper-small` | 244 M | ~0.5 GB | Strong multilingual; ~6× real-time |
| 4 | `openai/whisper-medium` | 769 M | ~1.5 GB | Near large-v2 quality; lower VRAM |
| 5 | `distil-whisper/distil-large-v3` | 756 M | ~1.5 GB | **6× faster** than large-v3; same WER |
| 6 | `openai/whisper-large-v3-turbo` | 809 M | ~1.6 GB | Pruned decoder; **8× speedup** vs large-v3 |
| 7 | `openai/whisper-large-v2` | 1.55 B | ~3.0 GB | Best pre-v3 accuracy; 99 languages |
| 8 | `openai/whisper-large-v3` | 1.55 B | ~3.0 GB | Best accuracy; handles accented speech |
| 9 | `nvidia/canary-1b` | 1.0 B | ~2.0 GB | CTC+AED; punctuation/capitalisation output |
| 10 | `facebook/seamless-m4t-v2-large` | 2.3 B | ~4.6 GB | Speech-to-speech/text; 100+ languages |

**Auto-selection thresholds** (VRAM available after 2 GB safety margin):

| VRAM | Selected model |
|---|---|
| < 2 GB or CPU | `openai/whisper-tiny` |
| 2–3 GB | `openai/whisper-base` |
| 3–5 GB | `openai/whisper-small` |
| 5–8 GB | `distil-whisper/distil-large-v3` |
| 8–12 GB | `openai/whisper-large-v3-turbo` |
| > 12 GB | `openai/whisper-large-v3` |

**CLI:**
```bash
ASR_ENABLED=true ASR_MODEL=openai/whisper-large-v3-turbo ASR_LANGUAGE=en
ASR_ENABLED=true ASR_MODEL=auto  # GPU auto-select
```

---

## 2. OCR — Optical Character Recognition

Extracts text visible in frame images (signs, labels, displays, documents).
Results stored in `frames.ocr_text` and `frame_facts_json["ocr_text"]`.
Injected into Qwen prompt as `[Text visible in frame]: ...`.

Two modes:
- **Local** (default): TrOCR / GOT-OCR2 / DeepSeek-OCR-2 via transformers
- **Sidecar** (`OCR_API_URL` set): any vision model served via vLLM/ollama

**DeepSeek-OCR-2** (3 B, recommended): uses DeepEncoder V2 for human-like
reading order across columns, tables, and mixed text+structure layouts.
Available at `deepseek-ai/DeepSeek-OCR-2`; Apache 2.0 license.
Architecture: SAM-ViT-B backbone + Qwen2Decoder2Encoder; ~6.79 GB BF16.

| # | Model ID | Params | VRAM (FP16) | Notes |
|---|---|---|---|---|
| 1 | `microsoft/trocr-base-printed` | 334 M | ~0.7 GB | Fast; printed document OCR |
| 2 | `microsoft/trocr-large-printed` | 558 M | ~1.2 GB | Better accuracy on complex layouts |
| 3 | `ucaslcl/GOT-OCR2_0` | 580 M | ~1.2 GB | Scene text, formulas, tables, multi-page |
| 4 | `microsoft/Florence-2-base` | 230 M | ~0.5 GB | Already in pipeline; handles OCR tasks |
| 5 | `microsoft/Florence-2-large` | 770 M | ~1.5 GB | Already in pipeline; caption+OCR |
| 6 | `Qwen/Qwen2.5-VL-3B-Instruct` | 3 B | ~6.0 GB | Strong spatial OCR with reasoning |
| 7 | `deepseek-ai/DeepSeek-OCR-2` | 3 B | ~6.8 GB | **Best layout understanding**; DeepEncoder V2 |
| 8 | `Qwen/Qwen2.5-VL-7B-Instruct` | 7 B | ~14.0 GB | Already in pipeline; top VLM OCR |
| 9 | `microsoft/Phi-3.5-vision-instruct` | 4.2 B | ~8.5 GB | 128 K context; document understanding |
| 10 | `llava-hf/llava-1.5-13b-hf` | 13 B | ~26.0 GB | Strong VLM with OCR capabilities |

**CLI:**
```bash
OCR_ENABLED=true OCR_MODEL=deepseek-ai/DeepSeek-OCR-2
OCR_ENABLED=true OCR_MODEL=auto  # GPU auto-select
OCR_ENABLED=true OCR_API_URL=http://localhost:8010/v1  # vLLM sidecar
```

---

## 3. Depth Estimation

Monocular depth estimation stores 5-bucket depth percentiles
`[p10, p25, p50, p75, p90]` in `frame_facts_json["depth"]` — compact
enough for DB storage while capturing relative scene depth structure.

| # | Model ID | Params | VRAM (FP16) | Notes |
|---|---|---|---|---|
| 1 | `depth-anything/Depth-Anything-V2-Small-hf` | 25 M | ~0.05 GB | Fastest; strong outdoor |
| 2 | `depth-anything/Depth-Anything-V2-Base-hf` | 97 M | ~0.2 GB | Good indoor+outdoor balance |
| 3 | `vinvino02/glpn-kitti` | 85 M | ~0.2 GB | GLPN; strong on KITTI outdoor |
| 4 | `Intel/dpt-large` | 307 M | ~0.6 GB | Dense prediction transformer; solid |
| 5 | `depth-anything/Depth-Anything-V2-Large-hf` | 335 M | ~0.7 GB | Best DepthAnything-V2 quality |
| 6 | `LiheYoung/depth-anything-large-hf` | 335 M | ~0.7 GB | DepthAnything V1 Large; still strong |
| 7 | `Intel/zoedepth-nk` | 345 M | ~0.7 GB | Metric depth; indoor+outdoor jointly |
| 8 | `prs-eth/marigold-lcm-v1-0` | 859 M | ~1.7 GB | Diffusion-based; photorealistic depth |
| 9 | `apple/DepthPro-hf` | 1.1 B | ~2.2 GB | Metric depth + focal length estimation |
| 10 | `geovision-research/DPT-DINOv2-L-384` | 307 M | ~0.6 GB | DPT with DINOv2 backbone |

**CLI:**
```bash
DEPTH_ENABLED=true DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf
DEPTH_ENABLED=true DEPTH_MODEL=auto  # GPU auto-select (smallest that fits)
```

---

## 4. Object Detection

Detects objects and stores normalised bounding boxes in
`frame_facts_json["detections"]`.  Open-vocabulary models (Grounding DINO,
OmDet-Turbo) accept custom label prompts via `DETECTION_LABELS`.

| # | Model ID | Params | VRAM (FP16) | Notes |
|---|---|---|---|---|
| 1 | `facebook/detr-resnet-50` | 41 M | ~0.1 GB | Classic DETR; COCO 42 mAP |
| 2 | `PekingU/rtdetr_r50vd` | 42 M | ~0.1 GB | RT-DETR; 53.1 mAP @ 108 FPS on T4 |
| 3 | `PekingU/rtdetr_r101vd` | 76 M | ~0.2 GB | RT-DETR larger backbone; 54.3 mAP |
| 4 | `omlab/omdet-turbo-swin-tiny-hf` | 108 M | ~0.3 GB | Open-vocabulary zero-shot |
| 5 | `omlab/omdet-turbo-swin-large-hf` | 218 M | ~0.5 GB | Open-vocab; best speed/accuracy |
| 6 | `IDEA-Research/grounding-dino-tiny` | 173 M | ~0.4 GB | Text-guided; zero-shot |
| 7 | `IDEA-Research/grounding-dino-base` | 341 M | ~0.7 GB | Text-guided; stronger accuracy |
| 8 | `microsoft/conditional-detr-resnet-101` | 62 M | ~0.2 GB | Faster convergence than DETR |
| 9 | `jozhang97/deta-swin-large` | 218 M | ~0.5 GB | 63.5 COCO AP |
| 10 | `SenseTime/deformable-detr` | 40 M | ~0.1 GB | Sparse attention; fast convergence |

**CLI:**
```bash
DETECTION_ENABLED=true DETECTION_MODEL=IDEA-Research/grounding-dino-tiny
DETECTION_ENABLED=true DETECTION_LABELS="vehicle,person,weapon,infrastructure"
DETECTION_ENABLED=true DETECTION_MODEL=auto DETECTION_CONFIDENCE=0.4
```

---

## 5. Image Segmentation

SAM-2 supports both image and video segmentation (object tracking across
frames).  Results can be stored in `frame_facts_json["segments"]` (mask
counts and class labels only — full masks are too large for DB storage).

| # | Model ID | Params | VRAM (FP16) | Video? | Notes |
|---|---|---|---|---|---|
| 1 | `facebook/sam2-hiera-tiny` | 38 M | ~0.1 GB | ✓ | Fastest SAM2; interactive |
| 2 | `facebook/sam2-hiera-small` | 46 M | ~0.1 GB | ✓ | Good quality; real-time |
| 3 | `facebook/sam-vit-base` | 93 M | ~0.2 GB | — | SAM1; prompt-based |
| 4 | `facebook/sam-vit-large` | 308 M | ~0.6 GB | — | SAM1 Large; strong edges |
| 5 | `CIDAS/clipseg-rd64-refined` | 71 M | ~0.2 GB | — | Text+click guided |
| 6 | `nvidia/segformer-b5-finetuned-ade-512-512` | 85 M | ~0.2 GB | — | Semantic; ADE20k 84.0 mIoU |
| 7 | `facebook/sam-vit-huge` | 641 M | ~1.3 GB | — | Best SAM1 quality |
| 8 | `facebook/sam2-hiera-large` | 224 M | ~0.5 GB | ✓ | Best SAM2; video tracking |
| 9 | `shi-labs/oneformer_coco_swin_large` | 219 M | ~0.5 GB | — | Panoptic+semantic+instance |
| 10 | `openmmlab/mask2former-swin-large-coco-panoptic` | 216 M | ~0.5 GB | — | SOTA panoptic |

**CLI / env:**
```bash
SEGMENTATION_ENABLED=true SEGMENTATION_MODEL=facebook/sam2-hiera-small
SEGMENTATION_MAX_MASKS=16 SEGMENTATION_MIN_AREA_NORM=0.002
```

---

## 6. Visual Question Answering / Vision-Language Models

The Qwen2.5-VL-7B (already the Phase 2 engine) falls in this category.
Additional VLM options for different scale/quality tradeoffs.

| # | Model ID | Params | VRAM (FP16) | Notes |
|---|---|---|---|---|
| 1 | `microsoft/Florence-2-base` | 230 M | ~0.5 GB | Already in pipeline; fast |
| 2 | `allenai/MolmoE-1B-0924` | 1 B | ~2.0 GB | MoE VLM; strong spatial pointing |
| 3 | `Qwen/Qwen2.5-VL-3B-Instruct` | 3 B | ~6.0 GB | Compact; strong grounding+OCR |
| 4 | `llava-hf/llava-1.5-7b-hf` | 7 B | ~14.0 GB | Instruction-following VLM |
| 5 | `allenai/Molmo-7B-D-0924` | 7 B | ~14.0 GB | Strong spatial reasoning + pointing |
| 6 | `Qwen/Qwen2.5-VL-7B-Instruct` | 7 B | ~14.0 GB | **Phase 2 engine**; already in pipeline |
| 7 | `microsoft/Phi-3.5-vision-instruct` | 4.2 B | ~8.5 GB | 128 K context; strong reasoning |
| 8 | `llava-hf/llava-1.5-13b-hf` | 13 B | ~26.0 GB | Best open LLaVA quality |
| 9 | `Qwen/Qwen2.5-VL-32B-Instruct` | 32 B | ~64.0 GB | Near GPT-4V quality; needs A100 |
| 10 | `Qwen/Qwen2.5-VL-72B-Instruct` | 72 B | ~144.0 GB | Top open-source VLM; needs 2×A100 |

**CLI (replace Phase 2 engine):**
```bash
QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct QWEN_API_URL=http://localhost:8010/v1
QWEN_MODEL=Qwen/Qwen2.5-VL-32B-Instruct QWEN_API_URL=http://localhost:8010/v1
```

---

## 7. Zero-Shot Image Classification (CLIP / SigLIP)

Already used internally for vehicle pre-screening before Qwen.  Can also be
exposed as a standalone endpoint or used to classify frames by custom taxonomy.

| # | Model ID | Params | VRAM (FP16) | Notes |
|---|---|---|---|---|
| 1 | `openai/clip-vit-base-patch32` | 151 M | ~0.3 GB | **Already in pipeline**; fastest CLIP |
| 2 | `openai/clip-vit-base-patch16` | 151 M | ~0.3 GB | Finer patches; better spatial detail |
| 3 | `google/siglip-base-patch16-224` | 200 M | ~0.4 GB | SigLIP; better zero-shot than CLIP |
| 4 | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 151 M | ~0.3 GB | LAION-2B training; wider coverage |
| 5 | `openai/clip-vit-large-patch14` | 428 M | ~0.9 GB | Strong baseline; 23 M+ downloads |
| 6 | `openai/clip-vit-large-patch14-336` | 428 M | ~0.9 GB | 336 px input; sharper features |
| 7 | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 986 M | ~2.0 GB | Largest standard CLIP |
| 8 | `google/siglip-so400m-patch14-384` | 878 M | ~1.8 GB | 400 M patch encoder; top zero-shot |
| 9 | `google/siglip2-so400m-patch14-384` | 878 M | ~1.8 GB | SigLIP2; improved calibration |
| 10 | `laion/CLIP-ViT-g-14-laion2B-s34B-b88K` | 1.37 B | ~2.7 GB | Largest available CLIP variant |

**CLI:**
```bash
OPENCLIP_MODEL=ViT-L-14 OPENCLIP_PRETRAINED=openai   # larger CLIP backbone
```

---

## 8. World Models — Video Understanding & Physical Prediction

World models understand scene dynamics, temporal relationships, and physical
plausibility across video frames.

**Target model:** arxiv.org/abs/2603.19312v1 (March 2026 — details pending
public HuggingFace release).  Set `WORLD_MODEL` to its HF ID once available.

Current fallback models produce 768-dim video clip embeddings stored in
`frame_facts_json["world_model"]` for downstream temporal analysis.

| # | Model ID | Params | VRAM (FP16) | Video? | Notes |
|---|---|---|---|---|---|
| 1 | `facebook/timesformer-base-finetuned-k400` | 122 M | ~0.3 GB | ✓ | Divided space-time attention |
| 2 | `MCG-NJU/videomae-base` | 122 M | ~0.3 GB | ✓ | Masked autoencoder pretraining |
| 3 | `MCG-NJU/videomae-large` | 307 M | ~0.6 GB | ✓ | Stronger video features |
| 4 | `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | 1 B | ~2.0 GB | ✓ | Video-language; strong retrieval |
| 5 | `THUDM/CogVideoX-2b` | 2 B | ~4.0 GB | ✓ | Generative world model |
| 6 | `THUDM/CogVideoX-5b` | 5 B | ~10.0 GB | ✓ | Higher quality video generation |
| 7 | `nvidia/Cosmos-1.0-Autoregressive-4B` | 4 B | ~8.0 GB | ✓ | **Physical world model for robotics** |
| 8 | `nvidia/Cosmos-1.0-Autoregressive-12B` | 12 B | ~24.0 GB | ✓ | Best physical fidelity |
| 9 | `tencent/HunyuanVideo` | 13 B | ~26.0 GB | ✓ | Open-source video world model |
| 10 | `Wan-AI/Wan2.1-T2V-14B` | 14 B | ~28.0 GB | ✓ | Text-to-video; 480P@24fps |

**CLI:**
```bash
WORLD_MODEL_ENABLED=true WORLD_MODEL=MCG-NJU/videomae-large
WORLD_MODEL_ENABLED=true WORLD_MODEL=nvidia/Cosmos-1.0-Autoregressive-4B
WORLD_MODEL_ENABLED=true WORLD_MODEL_CLIP_FRAMES=16 WORLD_MODEL_STORE_EMBED=true
```

---

## 9. VLA — Vision-Language-Action Models

VLA models extend VLMs with an **action head**: they produce structured driving-domain
observations (perception summary, object list, trajectory hint, hazard assessment) in
addition to free-text descriptions.

The selfsuvis integration uses a thin OpenAI-compatible HTTP adapter (`pipeline/vision/unidrive.py`)
that works with **any** capable vision LLM served via vLLM or ollama.  The actual
`owl10/UniDriveVLA_Nusc_Base_Stage3` checkpoint (Qwen3-VL backbone, trained on nuScenes)
can be used if served via an appropriate bridge; alternatively, any Qwen2.5-VL or other
vision model can serve the same structured-output schema.

> **Practical note:** The upstream UniDriveVLA checkpoint requires multi-camera nuScenes
> format for its internal driving stack.  For arbitrary single-camera mission video,
> **point `--unidrive-api-url` at a Qwen2.5-VL-7B sidecar** — it handles the structured
> JSON output schema with equal or better quality for non-road domains (aerial, maritime,
> off-road terrain).

| # | Model ID | Params | VRAM (FP16) | Notes |
|---|---|---|---|---|
| 1 | `owl10/UniDriveVLA_Nusc_Base_Stage3` | ~2 B | ~4 GB | Qwen3-VL-2B backbone; nuScenes stage 3 (final) |
| 2 | `owl10/UniDriveVLA_Nusc_Large_Stage3` | ~8 B | ~16 GB | Qwen3-VL-8B backbone; stronger scene understanding |
| 3 | `Qwen/Qwen2.5-VL-7B-Instruct` | 7 B | ~14 GB | **Recommended general backend**; handles non-road scenes well |
| 4 | `Qwen/Qwen2.5-VL-3B-Instruct` | 3 B | ~6 GB | Lower VRAM; good for edge devices |
| 5 | `Qwen/Qwen2.5-VL-32B-Instruct` | 32 B | ~64 GB | Highest quality; needs A100 |

HuggingFace collection: `https://huggingface.co/collections/owl10/unidrivevla`

**Structured output schema** (returned for every analysed frame):

```json
{
  "understanding": {"scene_summary": "...", "traffic_context": "...", "risk_level": "low|medium|high|unknown", "key_agents": []},
  "perception":    {"objects": [{"label": "...", "count": 1, "salience": "high"}], "drivable_area": "clear|partial|blocked|unknown", "lane_structure": "..."},
  "planning":      {"recommended_action": "...", "trajectory_hint": "...", "hazards": []},
  "mixture_of_experts": {"consensus_summary": "...", "expert_agreement": "high|medium|low|unknown", "disagreement_points": []}
}
```

**CLI:**
```bash
# Point at Qwen2.5-VL sidecar (recommended for non-road missions)
ssv --mode local --unidrive-api-url http://localhost:8010/v1 --unidrive-model Qwen/Qwen2.5-VL-7B-Instruct

# Point at actual UniDriveVLA checkpoint (if served via compatible bridge)
ssv --mode local --unidrive-api-url http://localhost:8030/v1 --unidrive-model owl10/UniDriveVLA_Nusc_Large_Stage3

# Download model weights for local bridge
python -m selfsuvis.scripts.prepare_models --unidrive
python -m selfsuvis.scripts.prepare_models --unidrive --unidrive-model owl10/UniDriveVLA_Nusc_Large_Stage3

# Env vars
UNIDRIVE_ENABLED=true UNIDRIVE_API_URL=http://localhost:8010/v1 UNIDRIVE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct UNIDRIVE_MAX_FRAMES=24
```

---

## 10. Self-Supervised Learning (SSL) — DAE Pretraining

The pipeline includes a lightweight convolutional Denoising Autoencoder (DAE)
trained on mission frames as a self-supervised pretext task.  It runs as step 20b
in Phase 3 (SSL) of the local pipeline and produces two artifacts:

- `{video_dir}/ssl/dae_best.pt`     — full encoder+decoder weights (best MSE epoch)
- `{video_dir}/ssl/dae_encoder.pt`  — encoder weights only (downstream feature use)

At inference the reconstruction MSE of any frame serves as an out-of-distribution
score, enabling unsupervised anomaly detection without labels.

| Component | Params | VRAM training (FP32, bs=32) | VRAM inference | Notes |
|---|---|---|---|---|
| ConvEncoder (3->64->128->256->256) | ~962 K | ~0.5 GB | ~50 MB | 14x14 bottleneck matches DINOv3 patch grid |
| ConvDecoder (256->256->128->64->3) | ~1.7 M | included above | ~50 MB | Symmetric transposed-conv |
| **DenoisingAutoencoder (full)** | **~2.7 M** | **~0.5 GB** | **~100 MB** | FP32; fits on CPU for inference |

Architecture: 4 stride-2 conv blocks (encoder) + 4 transposed-conv blocks (decoder).
Input resolution: 224x224.  Training target: MSE between reconstruction and clean frame.

**Corruption modes (applied during training):**

| Mode | Description |
|---|---|
| `gaussian` | Additive Gaussian noise (std=0.2) |
| `masking` | 15% of 16x16 patches replaced with channel mean |
| `both` | Gaussian noise then patch masking (default) |

**Active-learning integration** — the DAE reconstruction score is an optional
fourth signal in the `assign_al_tags` scoring formula:

```
Two signals   (no DAE):  0.60 * dino_dist + 0.40 * (1 - caption_conf)
Three signals (DAE only): 0.45 * dino_dist + 0.30 * (1 - caption_conf) + 0.25 * recon_score
Four signals  (RSSM+DAE): 0.25 * dino_dist + 0.20 * (1 - caption_conf)
                         + 0.30 * rssm_surprise + 0.25 * recon_score
```

**CLI (local pipeline step 20b):**
```bash
# Step 20b runs automatically in the local pipeline when Phase 3 SSL is active.
# To train standalone:
python scripts/finetune_dae.py \
  --frames-dir /data/missions/my_mission/frames \
  --output-dir /data/missions/my_mission/ssl \
  --epochs 15 --batch-size 32 --device cuda

# Python API:
from selfsuvis.pipeline.training.dae import DAEFinetuneConfig, run_dae_finetune
ckpt = run_dae_finetune(DAEFinetuneConfig(
    frames_dir="...", output_dir="...", epochs=15, device="cuda"
))
```

**Anomaly scoring API:**
```bash
# Python API:
from selfsuvis.pipeline.analysis.anomaly import load_dae_scorer, score_frames_anomaly, tag_anomalous_frames
scorer = load_dae_scorer(checkpoint_path="dae_best.pt", device="cuda")
raw    = score_frames_anomaly(frame_paths, scorer)
norms, tags = tag_anomalous_frames(raw)   # tags: high_anomaly | anomaly | normal
```

See ``examples.md`` for full workflow code.

---

## GPU Resource Guide

Typical selfsuvis worker VRAM budget on a 16 GB GPU (e.g. RTX 4060 Ti / 4080):

| Component | VRAM | Notes |
|---|---|---|
| CLIP ViT-B/16 (FP16) | ~0.3 GB | Always loaded |
| DINOv3 ViT-B/14 (FP16) | ~0.4 GB | Loaded when MODEL_NAME=dinov3 |
| Florence-2-large (FP16) | ~1.5 GB | Loaded per-pass then can be freed |
| Whisper large-v3-turbo (FP16) | ~1.6 GB | Loaded per-video then freed |
| **Qwen2.5-VL-7B sidecar** | **~14 GB** | **Separate container; manages own VRAM** |
| DepthAnything-V2-Small | ~0.05 GB | Tiny overhead |
| RT-DETR-R50 | ~0.1 GB | Tiny overhead |
| VideoMAE-Base | ~0.3 GB | Loaded per-clip window |

**Total worker-side (without Qwen sidecar):** ~4.3 GB on a dinov3 + Florence run.
Leaves ~11 GB for the Qwen sidecar to use from the same GPU, or Qwen can
use a separate GPU if available.

---

## Adding a New Model

1. Add a `ModelEntry` to the appropriate list in `pipeline/model_registry.py`
2. If it requires a new pass: add `_run_<name>_pass()` to `VideoIndexer`
3. Add config vars to `pipeline/config.py` following the `_env()` pattern
4. Wire the pass in `index_video()` between OCR and Qwen passes
5. Add any new DB columns to `scripts/migrate_postgres.py` and `pipeline/mission_db.py`
