# Florence-2 Captioning Runbook

> Covers: local vs vLLM sidecar modes, batch size tuning, OOM fallback,
> caption confidence, and prompt version bumps.

---

## 1. Architecture overview

```
VideoIndexer / step_gemma_caption (local pipeline)
  └─ Florence-2 captioning pass (batch, post-loop)
       ├─ Local:  FlorenceModel.caption_batch()   ← loaded in worker VRAM
       │    FLORENCE_BATCH_SIZE=16 (auto-fallback to 1 on OOM)
       └─ Sidecar: POST /v1/chat/completions      ← vLLM serving Florence-2
            FLORENCE_API_URL set → no local weights loaded
```

Florence-2 is the **always-on fallback captioner** — it runs for all frames that
Gemma skips or when `GEMMA_API_URL` is unset. Every indexed frame gets a caption.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` | HuggingFace model ID |
| `FLORENCE_BATCH_SIZE` | `16` | Images per forward pass (auto-falls to 1 on OOM) |
| `FLORENCE_PROMPT_VERSION` | `v1` | Prompt version tag stored with each caption |
| `FLORENCE_API_URL` | `""` | vLLM sidecar endpoint — when set, no local load |
| `DEVICE` | `auto` | Device for local inference |
| `USE_FP16` | `true` | FP16 inference (halves VRAM vs FP32) |

---

## 3. Model variants

| Model ID | Params | VRAM FP16 | Notes |
|---|---|---|---|
| `microsoft/Florence-2-base` | 230 M | ~0.5 GB | Fast; lower caption quality |
| `microsoft/Florence-2-large` | 770 M | ~1.5 GB | **Default**; best quality |

---

## 4. Quick start

### Local inference (default)

```bash
# Default: Florence-2-large loaded locally
ssv --mode local

# Smaller model for low-VRAM machines
FLORENCE_MODEL=microsoft/Florence-2-base ssv --mode local

# Download weights
python -m selfsuvis.scripts.prepare_models --florence
```

### vLLM sidecar

```bash
# Start Florence-2 via vLLM
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Florence-2-large \
  --trust-remote-code \
  --task generate \
  --port 8020

# Point pipeline at sidecar (no local VRAM used)
FLORENCE_API_URL=http://localhost:8020/v1 ssv --mode local
```

---

## 5. Health check

```bash
# Local
python -c "
from pipeline.florence_model import FlorenceModel
from PIL import Image
import numpy as np
m = FlorenceModel()
img = Image.fromarray(np.zeros((224,224,3), dtype='uint8'))
r = m.caption_batch([img])
print('Caption:', r)
"

# Sidecar
curl -s http://localhost:8020/v1/models | python -m json.tool
```

---

## 6. OOM recovery

Florence auto-detects CUDA OOM during the batch pass:

```
WARNING  Florence OOM on batch_size=16; retrying with batch_size=1
```

This is expected behavior. If batch=1 still OOMs:
1. Switch to `microsoft/Florence-2-base` (~0.5 GB)
2. Or use the vLLM sidecar (offloads VRAM from worker)
3. Or `DEVICE=cpu` (slow but no VRAM constraint)

---

## 7. Prompt version

`FLORENCE_PROMPT_VERSION` is stored in `frames.caption_model` alongside the model
name (e.g. `florence-2-large:v1:fp16`). Bump to `v2` when changing the prompt or
post-processing logic so existing captions can be distinguished from new ones.

```bash
FLORENCE_PROMPT_VERSION=v2 ssv --mode local
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `caption_confidence: 0.5` on many frames | OOM fallback or empty output | Reduce batch size or switch to Florence-base |
| Captions are single words or incomplete | Wrong task token in prompt | Check `FLORENCE_PROMPT_VERSION` and model compatibility |
| `trust_remote_code` error | Florence requires non-standard code | `pip install transformers>=4.38` |
| vLLM returns 400 | Wrong `--task` flag | Add `--task generate` to vLLM startup |
| Slow: >5s per frame | CPU inference | Set `DEVICE=cuda` or use sidecar |
