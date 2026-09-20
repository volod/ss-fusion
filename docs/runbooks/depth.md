# Monocular Depth Estimation Runbook

> Covers: enabling depth estimation, model selection, output format,
> and the current throughput-oriented local defaults.

---

## 1. Architecture overview

```
VideoIndexer
  └─ Depth pass (per frame)
       └─ DepthModel.estimate_batch()    ← loaded in worker VRAM
            → 5-bucket depth percentiles [p10, p25, p50, p75, p90]
            → frame_facts_json["depth"]
```

Depth is **disabled by default** (`DEPTH_ENABLED=false`). Enable for missions
where relative scene geometry matters (obstacle proximity, terrain slope,
structural distances). The output is compact enough for DB storage — only
5 percentile values per frame, not the full depth map. Local runs now default
to a faster auto profile so depth can stay enabled without dominating runtime.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEPTH_ENABLED` | `false` | Enable depth estimation pass |
| `DEPTH_MODEL` | `auto` | Model ID or `auto` for profile-aware selection |
| `DEPTH_AUTO_PROFILE` | `fast` | Auto policy: `fast` or `quality` |
| `DEPTH_BATCH_SIZE` | `8` | Outer batch size used by the local pipeline |
| `DEPTH_IMAGE_MAX_SIDE` | `768` | Resize bound applied before depth inference |
| `DEVICE` | `auto` | Device for inference |

---

## 3. Model selection

| Model ID | Params | VRAM | Notes |
|---|---|---|---|
| `depth-anything/Depth-Anything-V2-Small-hf` | 25 M | ~0.05 GB | Fastest lightweight option |
| `depth-anything/Depth-Anything-V2-Base-hf` | 97 M | ~0.2 GB | **Current local fast-profile default** |
| `depth-anything/Depth-Anything-V2-Large-hf` | 335 M | ~0.7 GB | Best DepthAnything quality |
| `apple/DepthPro-hf` | 1.1 B | ~2.2 GB | Metric depth + focal length; best for precise distances |

**Auto-selection** (`DEPTH_MODEL=auto`) now depends on `DEPTH_AUTO_PROFILE`:

- `DEPTH_AUTO_PROFILE=fast`: prefers `depth-anything/Depth-Anything-V2-Base-hf`
  on CUDA for lower end-to-end latency.
- `DEPTH_AUTO_PROFILE=quality`: prefers `apple/DepthPro-hf` on CUDA.
- CPU-only runs fall back to a transformer depth pipeline suitable for host inference.

---

## 4. Quick start

```bash
# Enable with auto model selection
DEPTH_ENABLED=true ssv --mode local

# Force the higher-quality auto profile
DEPTH_ENABLED=true DEPTH_AUTO_PROFILE=quality ssv --mode local

# Explicit model
DEPTH_ENABLED=true DEPTH_MODEL=depth-anything/Depth-Anything-V2-Large-hf ssv --mode local

# Download weights
python -m selfsuvis.scripts.prepare_models --depth
```

---

## 5. Interpreting output

```json
{"depth": [0.12, 0.28, 0.51, 0.72, 0.89]}
```

Values are normalised to [0, 1] within each frame (relative depth).
- `p10`: near objects (foreground) depth percentile
- `p50`: median scene depth
- `p90`: far objects (background)

High p10 = something close to the camera. Low p90 = short-range scene (indoor, close-up).

Because the stored artifact is a percentile summary rather than a dense per-pixel map,
the local pipeline can safely resize frames before inference in most mission-analysis
workloads. That is the reason `DEPTH_IMAGE_MAX_SIDE=768` is now the default.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Depth pass skipped` | `DEPTH_ENABLED=false` | Set `DEPTH_ENABLED=true` |
| All values near 0.5 | Model convergence failure on blank/uniform frame | Expected for featureless frames (sky, water) |
| `CUDA out of memory` | DepthPro at 1.1B too large alongside other models | Use `auto` or select smaller model |
| Very slow: >2s per frame | Running on CPU or using a heavier explicit model | Set `DEVICE=cuda`, keep `DEPTH_MODEL=auto`, or use `DEPTH_AUTO_PROFILE=fast` |
| First run is much slower | Weights are still downloading into the Hugging Face cache | Re-run after the model finishes caching |
| `ModuleNotFoundError: transformers` | transformers not installed | `pip install transformers` |
