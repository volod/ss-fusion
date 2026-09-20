# World Model + RSSM Temporal Surprise Runbook

> Covers: enabling temporal embeddings, model selection, clip chunking,
> similarity-based temporal anomaly detection, and the lightweight RSSM
> that adds per-frame temporal surprise to the active learning signal.

---

## 1. Architecture overview

Step Q runs two complementary temporal analyses:

```
VideoIndexer / step_world_model (step Q)
  ├─ Part A — Heavy world model  (WORLD_MODEL_ENABLED=true, off by default)
  │    └─ WorldModel.encode_clips()    ← loaded in worker VRAM
  │         frame sequence → clip windows of WORLD_MODEL_CLIP_FRAMES frames
  │         → 768-dim embedding per clip
  │         → frame_facts_json["world_model"]["embedding_id"]
  │         → clip similarity scores vs. mission mean (anomaly score)
  │
  └─ Part B — Lightweight RSSM  (DREAMER_ENABLED=true, on by default)
       └─ RSSMEmbedder.encode_sequence()    ← CPU, ~100K params, ~50ms
            CLIP embedding sequence (already computed in step B)
            → 20 gradient steps of online GRU training per mission
            → surprise_score per frame (cosine dist: predicted vs. observed latent)
            → frame_facts_json["rssm"]["surprise_score"]
            → feeds al_score: 0.35×dino_dist + 0.25×(1-caption_conf) + 0.40×rssm_surprise
```

The heavy world model requires a GPU and a large pre-trained video model. The RSSM is
CPU-friendly, has no pre-trained weights, and trains online on each mission's CLIP
embeddings. They serve different purposes and can run independently.

**When to enable each:**

| Component | Default | Best for |
|---|---|---|
| Heavy world model | Off | Temporal retrieval, route surveillance, activity recognition, change-point detection |
| RSSM | **On** | Active learning — selecting the most informative frames for SSL fine-tuning |

---

## 2. Environment variables

### Part A — Heavy world model

| Variable | Default | Description |
|---|---|---|
| `WORLD_MODEL_ENABLED` | `false` | Enable world model temporal embedding pass |
| `WORLD_MODEL` | `nvidia/Cosmos-1.0-Autoregressive-4B` | HuggingFace model ID or `auto` |
| `WORLD_MODEL_CLIP_FRAMES` | `8` | Frames aggregated into one clip embedding |
| `WORLD_MODEL_STORE_EMBED` | `false` | Store raw embedding vector in DB (large; default off) |

### Part B — RSSM temporal surprise

| Variable | Default | Description |
|---|---|---|
| `DREAMER_ENABLED` | `true` | Enable RSSM per-frame surprise scoring (CPU, no download) |
| `DREAMER_HIDDEN_DIM` | `256` | GRU hidden state dimension |
| `DREAMER_LATENT_DIM` | `32` | Latent posterior dimension |
| `DREAMER_TRAIN_STEPS` | `20` | Online gradient steps per mission |
| `DREAMER_STORE_TEMPORAL` | `false` | Store recurrent state sequence in DB |

---

## 3. Model selection

| Model ID | Params | VRAM | Notes |
|---|---|---|---|
| `facebook/timesformer-base-finetuned-k400` | 122 M | ~0.3 GB | Fast; K400 action recognition features |
| `MCG-NJU/videomae-base` | 122 M | ~0.3 GB | Masked autoencoder; strong temporal features |
| `MCG-NJU/videomae-large` | 307 M | ~0.6 GB | Better quality; good default for edge GPUs |
| `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | 1 B | ~2.0 GB | Strong video-language retrieval |
| `nvidia/Cosmos-1.0-Autoregressive-4B` | 4 B | ~8.0 GB | **Default** — physical world model for robotics |

---

## 4. Quick start

```bash
# RSSM is on by default — no flags needed
ssv --mode local

# Disable RSSM (revert to original two-signal AL formula)
DREAMER_ENABLED=false ssv --mode local

# Enable heavy world model (requires GPU) with default Cosmos model
WORLD_MODEL_ENABLED=true ssv --mode local

# Lightweight world model for edge GPU
WORLD_MODEL_ENABLED=true WORLD_MODEL=MCG-NJU/videomae-large ssv --mode local

# Store world model embeddings in DB (increases storage significantly)
WORLD_MODEL_ENABLED=true WORLD_MODEL_STORE_EMBED=true ssv --mode local

# Download world model weights
python -m selfsuvis.scripts.prepare_models --world-model
```

---

## 5. Clip chunking

`WORLD_MODEL_CLIP_FRAMES` controls the temporal window:

| Value | Window | Best for |
|---|---|---|
| 4 | ~2s at 2 FPS | Short events, fast scenes |
| 8 | ~4s at 2 FPS | **Default** — balanced |
| 16 | ~8s at 2 FPS | Slow scene evolution, long-range patterns |
| 32 | ~16s at 2 FPS | Route-level similarity |

A clip with fewer frames than `WORLD_MODEL_CLIP_FRAMES` is padded or skipped
depending on the model's minimum input requirement.

---

## 6. Temporal anomaly scoring (heavy world model)

Each clip embedding is compared to the mission mean embedding (cosine distance).
Clips with distance > 1.5× the mission standard deviation are flagged in the report
as potential anomaly moments.

---

## 7. RSSM architecture and active learning impact

The RSSM (Recurrent State Space Model) is a lightweight per-mission world model inspired
by DreamerV3 (Romero et al., "Dream to Fly", ICRA 2026). It operates entirely in CLIP
embedding space — no raw pixels, no pre-trained weights, no GPU required.

**Architecture (~100K params):**

```
Encoder:   Linear(clip_dim → 2×latent_dim)  →  μ, log σ²  →  z_k (reparametrised)
Recurrent: GRUCell(z_k, h_{k-1})            →  h_k
Dynamics:  Linear(h_k → latent_dim)         →  z̃_{k+1}  (predicted next latent)
Decoder:   Linear(h_k + z_k → clip_dim)     →  x̂_k  (reconstructed embedding)

Surprise:  cosine_distance(z̃_k, z_k)        →  how unexpected was frame k?
```

**Training:** 20 gradient steps (Adam, lr=3e-4) on the mission's CLIP sequence after
frames are embedded. Total overhead: ~50ms CPU. The model learns the video's temporal
rhythm within a single mission.

**Active learning impact:**

Without RSSM the AL formula weights static per-frame signals:
```
al_score = 0.60 × dino_dist + 0.40 × (1 - caption_confidence)
```

With RSSM enabled (default), temporal surprise gets a 40% weight:
```
al_score = 0.35 × dino_dist + 0.25 × (1 - caption_confidence) + 0.40 × rssm_surprise
```

Frames at scene transitions, first appearances of new objects, or sudden environment
changes rank higher than frames that are visually unusual but temporally predictable
(e.g., a static unusual object in a hover shot). These are better SSL training candidates.

**EMA fallback:** If PyTorch is unavailable, `RSSMEmbedder` falls back to an exponential
moving average of CLIP embeddings. Surprise degrades to `cosine_distance(ema_t, embed_t)`.
The signal is weaker but non-zero, and the AL formula still uses it.

**Output in `frame_facts_json["rssm"]`:**

```json
{
  "surprise_score": 0.312,
  "method": "rssm",
  "model": "RSSMEmbedder"
}
```

---

## 8. Troubleshooting

### Heavy world model

| Symptom | Cause | Fix |
|---|---|---|
| `World model pass skipped` | `WORLD_MODEL_ENABLED=false` | Set `WORLD_MODEL_ENABLED=true` |
| `CUDA out of memory` | Cosmos 4B too large alongside other models | Switch to `MCG-NJU/videomae-large` |
| All clips have identical embeddings | Very static footage (hovering camera) | Expected; cosine similarity will be ~1.0 |
| Very slow: >10s per clip | CPU inference or large model | Set `DEVICE=cuda`; use smaller model |
| `ModuleNotFoundError` | transformers / timm not installed | `pip install transformers timm` |

### RSSM temporal surprise

| Symptom | Cause | Fix |
|---|---|---|
| All surprise scores are 0.0 | DREAMER_ENABLED=false or PyTorch absent | Check env var; EMA fallback produces non-zero scores even without PyTorch |
| `method: ema_fallback` in frame_facts_json | PyTorch not installed | `pip install torch` — or accept EMA as a valid degraded mode |
| Uniformly low surprise scores | Highly repetitive footage | Expected; DINO distance will dominate the AL score |
| Very short mission (<10 frames) | Insufficient sequence for online training | RSSM falls back to EMA; surprise is a simple temporal deviation measure |
| RSSM loss not converging (high train_loss) | Learning rate too high or degenerate embeddings | Reduce `DREAMER_TRAIN_STEPS`; check that CLIP embeddings are L2-normalised |
