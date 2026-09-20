# Gemma API Operations Runbook

> Covers: local GemmaEmbedder vs sidecar modes, Ollama/vLLM setup,
> captioning fallback chain, scene analysis for P3 tracking,
> VRAM budget, and exponential backoff guidance.

---

## 1. Architecture overview

```
Gemma runs in two distinct roles in the local pipeline:

Role 1 — Captioning sidecar (step J)
  VideoIndexer / step_gemma_caption
    ├─ POST /v1/chat/completions     ← Ollama / vLLM (GEMMA_API_URL set)
    │   up to GEMMA_MAX_CAPTION_FRAMES per mission
    └─ Florence-2 fallback           ← for remaining frames or on error

Role 2 — Scene analysis for Gemma-directed tracking (step P3)
  step_gemma_directed_tracking
    ├─ POST /v1/chat/completions     ← same Gemma sidecar
    │   12 sampled frames → structured JSON (scene_type, objects, tracking_priority)
    └─ SAM + RF-DETR use Gemma output to guide segmentation and tracking
```

Leave `GEMMA_API_URL` empty to disable both roles (Florence handles all captions;
step P3 is skipped).

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `GEMMA_API_URL` | `""` | Sidecar endpoint: `http://localhost:11434/v1` (Ollama) or `http://localhost:8000/v1` (vLLM) |
| `GEMMA_API_MODEL` | `gemma4:e4b` (Ollama) / `google/gemma-4-4b-it` (vLLM) | Model tag |
| `GEMMA_API_BACKEND` | `ollama` | Backend hint: `ollama` or `vllm` |
| `GEMMA_API_TIMEOUT_SEC` | `60` | Per-request timeout |
| `GEMMA_MAX_CAPTION_FRAMES` | `200` | Max frames captioned via Gemma; rest use Florence |
| `GEMMA_CAPTION_CHUNK_SIZE` | `3` | Frames per async captioning chunk |
| `GEMMA_CAPTION_RETRIES` | `1` | Retry count before Florence fallback |

**Reasoning/audit step (step AA):**

| Variable | Default | Description |
|---|---|---|
| `REASONING_API_URL` | same as `GEMMA_API_URL` | Override for the final audit step |
| `REASONING_MODEL` | `""` | Override model for audit (e.g., larger thinking model) |
| `REASONING_TIMEOUT_SEC` | `240` | Longer timeout for deep reasoning models |

---

## 3. Model options

| Tag (Ollama) | HF ID (vLLM) | VRAM | Notes |
|---|---|---|---|
| `gemma4:e4b` | `google/gemma-4-4b-it` | ~3 GB Q4 | **Default** — fits 16 GB GPU |
| `gemma4:12b` | `google/gemma-4-12b-it` | ~7 GB Q4 | Better reasoning |
| `gemma4:26b` | `google/gemma-4-26b-it` | ~15 GB Q4 | Near GPT-4 quality |
| `gemma4:31b` | `google/gemma-4-31b-it` | ~18 GB Q4 | Best Gemma quality |
| `gemma3:4b` | `google/gemma-3-4b-it` | ~8 GB BF16 | Prior generation; no multimodal |

All Gemma models require HF_TOKEN + license acceptance at `huggingface.co/google/`.

---

## 4. Quick start

### Ollama sidecar (recommended for local runs)

```bash
# Pull and serve
ollama pull gemma4:e4b
ollama serve   # default port 11434

# Run pipeline with Gemma captioning + P3 tracking
ssv --mode local --gemma-api-url http://localhost:11434/v1
```

### vLLM sidecar

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-4b-it \
  --port 8000

ssv --mode local \
  --gemma-api-url http://localhost:8000/v1 \
  --gemma-model google/gemma-4-4b-it
```

### Disable Gemma entirely (Florence-only captioning)

```bash
ssv --mode local   # GEMMA_API_URL unset → Florence handles all captions
```

---

## 5. Health check

```bash
# Ollama: confirm model is loaded
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# vLLM: confirm server health
curl -s http://localhost:8000/health
# → {"status":"ok"}

# Minimal completions ping
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e4b","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
# Expected: 200
```

---

## 6. Florence fallback activation

To disable Gemma captioning and use Florence for all frames:

```bash
# Unset GEMMA_API_URL in .env
GEMMA_API_URL=

# Or omit --gemma-api-url flag
ssv --mode local
```

Pipeline logs confirm which captioner is active:
```
INFO  Gemma captioning pass: 200 frames via Gemma, 47 via Florence fallback
# vs.
INFO  Florence captioning pass: 247 frames (batch_size=16)
```

---

## 7. Exponential backoff guidance

The current retry config (1 retry, 60s timeout) handles < 10% timeout rates.

**If timeout rate exceeds 10%:**

1. Reduce `GEMMA_CAPTION_CHUNK_SIZE` from 3 → 1 to serialize encoding load.
2. Increase `GEMMA_API_TIMEOUT_SEC` to 90 or 120.
3. Scale Ollama parallelism: `OLLAMA_NUM_PARALLEL=4 ollama serve`

---

## 8. VRAM budget (16 GB GPU)

| Component | VRAM |
|---|---|
| CLIP ViT-B/16 | ~0.5 GB |
| DINOv3 ViT-B/14 | ~0.4 GB |
| Florence-2-large (captioning fallback) | ~2.7 GB |
| Gemma4:e4b Q4 (Ollama sidecar) | ~3.0 GB |
| YOLO11l | ~0.5 GB |
| SAM2-hiera-large | ~0.5 GB |
| **Worker total (without Qwen sidecar)** | **~4.6 GB** |

Gemma runs as an HTTP sidecar — no direct VRAM from the worker process.

---

## 9. Log patterns to watch

| Pattern | Meaning |
|---|---|
| `Gemma caption chunk failed after 2 attempt(s)` | Chunk fell back to Florence; check Ollama health |
| `Florence fallback batch failed` | Both Gemma and Florence are unhealthy |
| `caption_model: gemma-api:gemma4:e4b` | Frame captioned by Gemma |
| `caption_model: florence-2-large:v1:fp16` | Frame captioned by Florence |
| `Scene: urban_street \| priority: ['vehicle', ...]` | Gemma structured analysis succeeded (step P3) |
| `Step P3 skipped: gemma_api_url not configured` | GEMMA_API_URL not set; P3 disabled |

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| All captions fall back to Florence | Gemma endpoint unreachable | Check `curl` health (§5); restart Ollama |
| `JSON parse error` in P3 scene analysis | Gemma returned free text, not JSON | Upgrade to `gemma4:12b`; check model vision support |
| Step P3 SAM freeze (30+ min) | Old Path B bug — now fixed | Update to latest code; Path B is now a pure fallback |
| `HF_TOKEN` auth error | Token not set or license not accepted | `huggingface-cli login` and accept license |
| `CUDA out of memory` in Ollama | Gemma model too large | Use `gemma4:e4b` instead of larger variant |
