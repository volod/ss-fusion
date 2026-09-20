# OCR Text Extraction Runbook

> Covers: enabling OCR, model selection, vLLM sidecar mode,
> and Qwen context injection.

---

## 1. Architecture overview

```
VideoIndexer
  └─ OCR pass (batch, post-Florence)
       ├─ Local:   OCRModel.extract_batch()     ← loaded in worker VRAM
       └─ Sidecar: POST /v1/chat/completions    ← vLLM/ollama endpoint
            OCR_API_URL set → no local weights loaded

       Results: frames.ocr_text column
                frame_facts_json["ocr_text"]
                injected into Qwen prompt as "[Text visible in frame]: ..."
```

OCR is **disabled by default** (`OCR_ENABLED=false`). Enable when missions capture
frames with informative text: road signs, vehicle plates, displays, documents.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `OCR_ENABLED` | `false` | Enable OCR pass |
| `OCR_MODEL` | `auto` | Model ID or `auto` for GPU-aware selection |
| `OCR_API_URL` | `""` | vLLM/ollama sidecar endpoint — when set, no local load |
| `OCR_BATCH_SIZE` | `4` | Frames per batch |
| `OCR_SIDECAR_CONCURRENCY` | env-specific | Parallel OCR sidecar requests during local runs |
| `OCR_IMAGE_MAX_SIDE` | env-specific | Resize bound for OCR frame uploads |
| `OCR_TIMEOUT_SEC` | `30` | Per-request timeout (sidecar mode) |
| `OCR_MIN_CAPTION_CONFIDENCE` | `0.55` | Skip OCR for frames with confidence above this |

---

## 3. Model selection

| Model ID | Params | VRAM | Notes |
|---|---|---|---|
| `microsoft/trocr-base-printed` | 334 M | ~0.7 GB | Fast; printed text only |
| `ucaslcl/GOT-OCR2_0` | 580 M | ~1.2 GB | Scene text + formulas + tables |
| `deepseek-ai/DeepSeek-OCR-2` | 3 B | ~6.8 GB | **Best layout understanding** |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7 B | ~14 GB | Already in pipeline (sidecar); handles OCR well |

**Auto-selection thresholds** (VRAM available):

| VRAM | Selected model |
|---|---|
| < 2 GB | `microsoft/trocr-base-printed` |
| 2–4 GB | `ucaslcl/GOT-OCR2_0` |
| > 4 GB | `deepseek-ai/DeepSeek-OCR-2` |

---

## 4. Quick start

### Local inference

```bash
# Enable with auto model selection
OCR_ENABLED=true ssv --mode local

# Explicit model
OCR_ENABLED=true OCR_MODEL=ucaslcl/GOT-OCR2_0 ssv --mode local

# Download weights
python -m selfsuvis.scripts.prepare_models --ocr
```

### vLLM sidecar (reuse Qwen endpoint)

```bash
# If Qwen sidecar is already running, point OCR at it too
OCR_ENABLED=true OCR_API_URL=http://localhost:8010/v1 OCR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
  ssv --mode local
```

## 5. Local pipeline default behavior

In the current local pipeline, OCR is no longer run on every extracted frame by default.
After Florence captioning, the runner prescreens frames by `caption_confidence` and only
sends lower-confidence frames to the OCR stage:

- frames with `caption_confidence < OCR_MIN_CAPTION_CONFIDENCE` are selected for OCR
- higher-confidence frames are skipped to save latency and sidecar load

This is why a local run may log something like `OCR prescreen: 16/51 frames selected`
followed by OCR results on fewer frames than the video contains.

---

## 6. Health check

```bash
# Verify OCR produces output
python -c "
import os; os.environ['OCR_ENABLED']='true'
from pipeline.vision.ocr import OCRModel
from PIL import Image, ImageDraw
img = Image.new('RGB', (300,100), 'white')
ImageDraw.Draw(img).text((10,30), 'TEST 123', fill='black')
m = OCRModel()
print(m.extract([img]))
"
```

---

## 7. Verifying text injection

```bash
# Check ocr_text column was populated
grep "Text visible" output/<video>/detailed_captions.md | head -3
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OCR pass skipped` | `OCR_ENABLED=false` | Set `OCR_ENABLED=true` |
| Empty `ocr_text` for all frames | No text visible in video | Expected; OCR returns empty string |
| OCR ran on fewer frames than the video contains | Florence prescreen skipped high-confidence frames | Expected; set `OCR_MIN_CAPTION_CONFIDENCE=1.0` to run OCR on all frames |
| `trust_remote_code` error | GOT-OCR2 needs it | `pip install transformers>=4.38` |
| Wrong characters extracted | Low-resolution source frames | Increase `SFM_FPS` to extract sharper frames |
| `CUDA out of memory` | DeepSeek-OCR-2 too large | Switch to `GOT-OCR2_0` or use sidecar |
