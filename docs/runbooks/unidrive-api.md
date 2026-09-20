# UniDriveVLA API Operations Runbook

> Covers: sidecar setup (vLLM / Ollama), backend selection by mission type,
> health checks, and troubleshooting the UniDriveVLA adapter.

---

## 1. Architecture overview

```
steps_caption.py
  └─ step_unidrive_analysis()
       ├─ UniDriveVLAModel.extract_batch()
       │   POST /v1/chat/completions      ← any OpenAI-compatible vision sidecar
       │   max_tokens=400, temperature=0.1, timeout=UNIDRIVE_TIMEOUT_SEC
       └─ skip                            ← if UNIDRIVE_API_URL unset or model unavailable
```

**Key design:** UniDriveVLA is treated as an OpenAI-compatible endpoint, not a
direct model invocation.  The pipeline sends frames as base64 JPEG + a structured
JSON output prompt, and any capable vision LLM returns the four-key schema.

---

## 2. Backend selection by mission type

| Mission type | Recommended backend | Model ID |
|---|---|---|
| Road / urban / highway | UniDriveVLA checkpoint | `owl10/UniDriveVLA_Nusc_Large_Stage3` |
| Aerial / drone | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Off-road / maritime | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Low VRAM (< 8 GB) | Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` |

The actual UniDriveVLA checkpoint (`owl10/*`) is trained on multi-camera nuScenes
format data.  For single-camera arbitrary mission video, a general Qwen2.5-VL backend
produces equal or better structured output without domain mismatch.

---

## 3. Environment variables

| Variable | Default | Description |
|---|---|---|
| `UNIDRIVE_ENABLED` | `false` | Enables UniDriveVLA expert analysis pass |
| `UNIDRIVE_API_URL` | `""` | OpenAI-compatible sidecar endpoint (e.g. `http://localhost:8010/v1`) |
| `UNIDRIVE_MODEL` | `owl10/UniDriveVLA_Nusc_Base_Stage3` | Model ID sent in the `model` field |
| `UNIDRIVE_BACKEND` | `vllm` | Backend hint: `vllm` or `ollama` |
| `UNIDRIVE_TIMEOUT_SEC` | `60` | Per-request HTTP timeout |
| `UNIDRIVE_MAX_FRAMES` | `24` | Max frames sampled per video |

---

## 4. Quick start

### Option A — Qwen2.5-VL via Ollama (recommended for aerial / off-road)

```bash
# Pull and serve
ollama pull qwen2.5vl:7b
ollama serve   # default port 11434

# Run pipeline
ssv --mode local \
  --unidrive-api-url http://localhost:11434/v1 \
  --unidrive-model qwen2.5vl:7b
```

### Option B — Qwen2.5-VL via vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8010

ssv --mode local \
  --unidrive-api-url http://localhost:8010/v1 \
  --unidrive-model Qwen/Qwen2.5-VL-7B-Instruct
```

### Option C — Download and serve UniDriveVLA checkpoint

```bash
# Download weights (~4 GB for base, ~16 GB for large)
python -m selfsuvis.scripts.prepare_models --unidrive
python -m selfsuvis.scripts.prepare_models --unidrive --unidrive-model owl10/UniDriveVLA_Nusc_Large_Stage3

# Serve via vLLM (requires Qwen3-VL support in your vLLM version)
python -m vllm.entrypoints.openai.api_server \
  --model owl10/UniDriveVLA_Nusc_Base_Stage3 \
  --trust-remote-code \
  --port 8030

ssv --mode local \
  --unidrive-api-url http://localhost:8030/v1 \
  --unidrive-model owl10/UniDriveVLA_Nusc_Base_Stage3
```

---

## 5. Health check

```bash
curl -s http://localhost:8010/v1/models | python -m json.tool
# Should show the model name in the response

curl -s http://localhost:11434/api/tags | python -m json.tool
# Ollama: should list qwen2.5vl model
```

---

## 6. Verify pipeline output

After a local run, check:

```bash
ls output/<video>/unidrive_analysis.md        # UniDrive per-frame report
ls output/<video>/multi_model_comparison.md   # Qwen vs UniDrive side-by-side (if both enabled)
```

In `unidrive_analysis.md`, each analysed frame should have non-empty fields in the
four-key schema.  `"service_unavailable": true` means the endpoint was unreachable.
`"parse_error": true` means the backend returned non-JSON output — increase
`UNIDRIVE_TIMEOUT_SEC` or switch to a more capable backend model.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Step skipped, "UNIDRIVE_API_URL not set" | Missing CLI flag or env var | Add `--unidrive-api-url` or set `UNIDRIVE_API_URL` |
| `service_unavailable: connection refused` | Sidecar not running | Start vLLM or Ollama server |
| `parse_error: true` on many frames | Backend returns free text, not JSON | Use a more instruction-following model; lower temperature already set to 0.1 |
| Off-domain output for aerial footage | Using driving-specific model | Switch backend to `Qwen/Qwen2.5-VL-7B-Instruct` |
| Slow: each frame takes > 15s | Backend too small for prompt | Use larger model or reduce `UNIDRIVE_MAX_FRAMES` |
| OOM on backend side | Model too large for GPU | Use `Qwen2.5-VL-3B` or enable 4-bit quantisation |
