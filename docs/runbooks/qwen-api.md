# Qwen2.5-VL API Runbook

> Covers: sidecar setup (vLLM / Ollama), vehicle pre-screening,
> structured JSON output, rolling state, and context injection order.

---

## 1. Architecture overview

```
step_caption / step_qwen_caption (step R)
  └─ Qwen pass (per keyframe)
       ├─ Pre-screen: CLIP cosine similarity ≥ QWEN_CLIP_THRESHOLD
       │    → skip non-vehicle frames to save API quota
       ├─ VideoKnowledge.context_for_frame(t_sec)
       │    → full context string: Florence caption, scene segment, ASR,
       │      OCR, depth, detections, prior Qwen state
       ├─ POST /v1/chat/completions     ← vLLM or Ollama sidecar
       │    → structured JSON: vehicle_groups, road_surface,
       │      traffic_conditions, infrastructure, safety_observations
       └─ VideoKnowledge.update_qwen_state()
            → rolling state for next frame's context
            → frames.frame_facts_json["qwen"]
            → detailed_captions.md
```

Qwen is **optional** — disabled when `QWEN_API_URL` is not set. The pipeline runs
normally without it; Florence captions are used instead.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `QWEN_API_URL` | `""` | Set to enable Qwen: `http://localhost:8010/v1` (vLLM) or `http://localhost:11434/v1` (Ollama) |
| `QWEN_BACKEND` | `vllm` | Backend hint: `vllm` or `ollama` |
| `QWEN_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model ID (HF for vLLM, tag for Ollama) |
| `QWEN_TIMEOUT_SEC` | `30` | Per-frame request timeout |
| `QWEN_CLIP_THRESHOLD` | `0.25` | CLIP similarity threshold for pre-screening (0 = disable) |

---

## 3. Model options

| Model ID | Params | VRAM | Notes |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | 3 B | ~6 GB | Compact; good for edge GPUs |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7 B | ~14 GB | **Default** — best quality/cost |
| `Qwen/Qwen2.5-VL-32B-Instruct` | 32 B | ~64 GB | Near GPT-4V quality; needs A100 |
| `Qwen/Qwen2.5-VL-72B-Instruct` | 72 B | ~144 GB | Best available; 2×A100 minimum |

Ollama tag format: `qwen2.5vl:7b`, `qwen2.5vl:3b`.

---

## 4. Quick start

### vLLM sidecar

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8010

ssv --mode local --qwen-api-url http://localhost:8010/v1
```

### Ollama sidecar

```bash
ollama pull qwen2.5vl:7b
ollama serve   # default port 11434

ssv --mode local \
  --qwen-api-url http://localhost:11434/v1 \
  --qwen-model qwen2.5vl:7b
```

### Compact model for low VRAM

```bash
ssv --mode local \
  --qwen-api-url http://localhost:8010/v1 \
  --qwen-model Qwen/Qwen2.5-VL-3B-Instruct
```

---

## 5. Health check

```bash
# vLLM
curl -s http://localhost:8010/v1/models | python -m json.tool

# Ollama
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# Minimal vision ping
curl -s -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-VL-7B-Instruct","messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
# Expected: HTTP 200 with a short response
```

---

## 6. CLIP pre-screening

`QWEN_CLIP_THRESHOLD=0.25` means only frames with cosine similarity ≥ 0.25 to the
label vocabulary (`LABELS_FILE`) are sent to Qwen. This is the vehicle pre-screening
gate — frames without recognizable objects are skipped to save API quota and latency.

Disable pre-screening (send all frames to Qwen):
```bash
QWEN_CLIP_THRESHOLD=0.0 ssv --mode local --qwen-api-url ...
```

---

## 7. Rolling state and context contamination

Qwen receives the previous frame's structured output as `[Prior frame state]` context.
If one frame produces malformed JSON (parse error), `update_qwen_state()` skips
storage — the chain resets to no prior context for the next frame.

Signs of contaminated state in `detailed_captions.md`:
- "Three vehicles present in prior frame" when prior frame had none
- Confidence escalation: each frame reports more objects than the last

Fix: Increase `QWEN_TIMEOUT_SEC` (short timeouts produce truncated JSON), or check
Qwen backend logs for OOM errors during inference.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Qwen pass skipped` | `QWEN_API_URL` not set | Add `--qwen-api-url` or set env var |
| All frames skipped by pre-screening | `QWEN_CLIP_THRESHOLD` too high | Lower to `0.1` or `0.0` |
| `parse_error` on most frames | Truncated JSON response | Increase `QWEN_TIMEOUT_SEC=60` or `max_tokens` |
| Rolling state contamination | Bad prior frame propagated | Check logs for `parse_error`; state auto-resets |
| Very slow: >20s per frame | Sidecar under-provisioned | Use larger GPU or quantize (`--quantization awq`) |
| HTTP 422 from vLLM | Vision content not supported | Ensure vLLM version supports multimodal; upgrade if needed |
