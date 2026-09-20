# Essential Technology Stack

This page is the practical technology map for a human learner. It explains the
main technologies that appear in recent code updates, why each one matters in
SelfSuvis, and what a developer needs to understand before writing or reviewing
code in each area. Use it after the runtime guide and before diving into
individual step files.

---

## 1. Python Runtime And Project Layout

### Package structure

SelfSuvis uses a `src/` package layout so the installed package and the editable
development tree share the same import namespace. This means `selfsuvis.*` imports
always resolve relative to `src/`, never to the working directory. The key
sub-packages are:

- `src/selfsuvis/app/` — FastAPI routers, request dependencies, API services, and
  application lifespan hooks.
- `src/selfsuvis/pipeline/` — media IO, local workflows, production indexing,
  mapping, fusion, training, realtime ingestion, and storage adapters.
- `src/selfsuvis/pipeline/core/` — cross-cutting shared utilities that every other
  pipeline module can import without circular dependencies.
- `src/selfsuvis/models/` — local embedding and temporal model wrappers.
- `src/selfsuvis/worker/` — PostgreSQL-backed asynchronous job execution.
- `src/selfsuvis/scripts/` — packaged helper CLIs such as env generation, model
  preparation, migrations, and sidecar helpers.
- `src/selfsuvis/analytics/` — post-run artifact inspection and quality scoring.
- [ss-sens `src/ss_sens/`](https://github.com/volod/ss-sens/tree/v0.1.0/src/ss_sens) — IoT mesh: MQTT/LoRaWAN ingestion and
  rolling site-state management.

### Where to start reading

The key human habit is to start from orchestration, not model internals. For the
local pipeline, begin with
[`ssv_vdp/pipeline/runner.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/runner.py).
For production indexing, begin with
``pipeline/workflows/indexer/``.

### Shared pipeline.core modules

The four modules below are prerequisites for understanding every other part of the
pipeline. They were introduced to eliminate repeated env-loading and startup logic
that was previously scattered across dozens of files.

**`pipeline/core/env.py`** — layered `.env` loading and typed environment access.

The loader merges two sources in priority order:
1. Packaged `env/<APP_ENV>.env` defaults (shipped inside the Python package)
2. Repo-root `.env` overrides (edited locally, never committed with secrets)

Neither source overwrites environment variables that already exist in the shell.
This means `export SOME_VAR=x && selfsuvis ...` always wins over any `.env` value.

Typed accessors (`env_str`, `env_bool`, `env_int`, `env_float`, `env_csv`,
`env_json_dict`) replace raw `os.getenv` calls throughout the codebase. The
benefit is uniform error handling: invalid ints or malformed JSON fall back to the
default and optionally log a warning instead of crashing at import time.

**`pipeline/core/preflight.py`** — startup checks for dependencies, model caches,
and reachable services.

The preflight system runs before frame extraction begins. It checks:
- Required Python packages are importable (prevents late ImportError inside a step)
- Local model weights are cached on disk (prevents a surprise 20-minute HuggingFace
  download mid-run)
- Optional services are reachable (Qdrant, mapper API, drone-dataset cache)

Errors from missing required dependencies stop the run immediately. Warnings from
degraded optional services are logged and allow the run to continue in reduced
capability mode. The split matters for debugging: if a run never produced an output
directory, look for `preflight:` log lines; if the directory exists but artifacts
are thin, the failure is a runtime quality issue, not a setup issue.

**`pipeline/core/sidecars.py`** — shared JSONL sidecar loading and HTTP sidecar
client behavior.

All sidecar loaders sort JSONL rows by `t` or `timestamp` before returning them.
Rows with missing or malformed timestamp fields are silently skipped rather than
raising an exception. This is intentional: bad optional sidecar data should degrade
evidence quality gracefully, not abort a run that may have hours of video.

**`pipeline/core/logging.py`** — structured logger factory.

All pipeline modules call `get_logger(__name__)` instead of
`logging.getLogger(__name__)`. The factory applies a consistent formatter and
propagates to the root handler configured by the CLI. Never use `print` for
diagnostic output in pipeline code; use the logger so the message is associated
with the correct step and suppressed below the configured log level.

---

## 2. API, Worker, Database, And Vector Store

### FastAPI

FastAPI is the web layer. It handles request parsing, dependency injection, and
response serialisation via Pydantic models. The key dependency is
`src/selfsuvis/app/deps.py`, which provides:
- `get_api_key` — validates `X-API-Key` using constant-time HMAC comparison
- `get_db` — yields a connection from the asyncpg pool
- `get_rate_limiter` — enforces per-client token-bucket rate limiting

The lifespan hook in `app/main.py` starts and stops the asyncpg connection pool,
the background worker, and the Qdrant client on startup and shutdown. This is the
correct place to trace any "connection refused" or "pool exhausted" error.

FastAPI routes are organised under `app/routers/`:
- `index.py` — triggers production indexing jobs
- `query.py` — vector search over Qdrant
- `jobs.py` — job queue status
- `realtime.py` — realtime frame and event ingestion
- `site.py` — coop site-state queries
- `cvat.py` — CVAT annotation webhook receiver

### PostgreSQL and asyncpg

PostgreSQL stores the durable, relational state that Qdrant does not hold: jobs,
missions, frames, model provenance, CVAT label mappings, automation state, realtime
map data, and site sensor records. The schema lives in migration scripts under
`src/selfsuvis/scripts/migrate_postgres.py`.

`asyncpg` is the async PostgreSQL driver. It uses prepared statements and binary
protocol, making it significantly faster than `psycopg2` for high-throughput
workloads. The important pattern is that connections are obtained from a pool (never
created per-request) and always released in a `finally` block or via async context
manager. Pool exhaustion under load is the main failure mode; watch for
`asyncpg.exceptions.TooManyConnectionsError` in production logs.

### Qdrant

Qdrant is a purpose-built vector database. SelfSuvis uses it for frame and map-tile
retrieval. Each frame is indexed with two named vectors:
- `clip` — OpenCLIP embeddings for text-image search
- `dino` — DINOv2/v3 embeddings for visual similarity search

Named vectors allow a single Qdrant point (frame record) to be queried from either
embedding space without duplicating the payload. A human learning this system should
understand:
1. What a `collection` is in Qdrant (roughly equivalent to a table)
2. What a `payload` is (the JSON metadata attached to each vector point)
3. The difference between `search` (nearest-neighbor by embedding) and `scroll`
   (paginated retrieval by payload filters)
4. Why approximate nearest-neighbor search (HNSW index) returns fast results but
   not always the true nearest neighbour

### Worker process

The worker claims jobs from the PostgreSQL jobs table, runs the production indexer,
and updates job status. It is a separate OS process from the FastAPI server. The
key distinction for learners is that local runs (triggered from the CLI) bypass the
worker entirely; the worker is only used when jobs are submitted through the API.

---

## 3. Security And Production Boundaries

### APP_ENV as the security gate

The `APP_ENV` environment variable (`dev` by default, `prod` in production) controls
which packaged defaults are loaded and whether certain security checks are enforced
versus warned. The key behaviour:

- `APP_ENV=prod` sets `API_AUTH_REQUIRED=true` automatically unless overridden.
- `API_AUTH_REQUIRED=true` causes `validate_settings()` to raise at startup if
  `API_KEY` is empty — it is a hard error, not a warning.
- `APP_ENV=dev` keeps `API_AUTH_REQUIRED=false` by default so local development
  works without credentials.

This pattern means the security posture is determined by environment name, not by
per-variable audit. Set `APP_ENV` first, then let the remaining settings derive from
it.

### API key authentication

API key comparison uses `hmac.compare_digest` (constant-time string comparison)
rather than `==`. This prevents timing attacks where an attacker measures response
latency to guess the key character by character. The lesson: never use `==` to
compare secrets in server code.

The key is passed in the `X-API-Key` header on every request. The dependency
`get_api_key` in `app/deps.py` handles rejection. If `API_AUTH_REQUIRED=false`
and `API_KEY` is empty, all requests pass (dev mode). If `API_AUTH_REQUIRED=true`,
an empty key or wrong key returns HTTP 401.

### CVAT webhook HMAC-SHA256

The CVAT webhook endpoint (`/webhook/cvat`) verifies the `X-Hook-Secret` header as
an HMAC-SHA256 signature over the raw request body. The verification pattern is:

```
expected = hmac.new(secret.encode(), raw_body, sha256).hexdigest()
hmac.compare_digest(expected, received)
```

When `CVAT_WEBHOOK_SECRET` is empty, the endpoint rejects all requests (fail-closed
behaviour). This is a security default, not a bug: unauthenticated annotation
updates should not affect mission data. To use CVAT webhooks in development, set
`CVAT_WEBHOOK_SECRET` to match the secret configured in CVAT's webhook settings.

### Rate limiting

Rate limiting uses bounded per-client token buckets. Each unique client identifier
(derived from `X-API-Key` or IP after optional proxy header processing) gets its own
bucket. Buckets are bounded in size: if a single client identifier generates
arbitrarily many requests, its bucket state is evicted rather than growing without
limit. This prevents memory exhaustion from spoofed or rotating client keys.

`TRUST_PROXY_HEADERS` should only be enabled when the API is behind a reverse proxy
(nginx, Caddy, etc.) that strips or overwrites incoming `X-Forwarded-For` headers.
Without a trustworthy proxy, any client can forge its source IP and bypass
per-IP rate limiting.

### The fail-closed principle

SelfSuvis security is designed to fail closed: missing configuration produces
errors, not silent fallbacks to unauthenticated access. The specific cases:
- Missing `API_KEY` with `API_AUTH_REQUIRED=true` → startup error
- Missing `CVAT_WEBHOOK_SECRET` → webhook rejects all requests (not open access)
- Malformed webhook signature → HTTP 400 (not 200 with unverified data)

For a learner the takeaway is: when a security-related feature is not working in
development, check whether the relevant secret is missing. The missing-secret
condition produces an explicit warning or error rather than degrading silently.

---

## 4. Media, Frames, And Sidecar Evidence

### FFmpeg and frame extraction

The whole pipeline starts with time. FFmpeg decodes source videos into sampled
frames and timestamps. Key facts:
- Frames are written to `frames/<video_id>/` as JPEG files
- Timestamps are written alongside each frame
- Frame rate is configurable via `FRAME_SAMPLE_RATE` in `.env`
- FFmpeg is called as a subprocess; the media helpers in `pipeline/media/` manage
  argument construction, error detection, and output path resolution

FFmpeg is a mature, stable tool with excellent documentation. When frame extraction
fails, the first thing to check is whether `ffmpeg` is on PATH and whether the input
file is a video format FFmpeg supports.

### Sidecar JSONL files

Sidecar files add external evidence aligned to the same timeline as the frames:
- `.imu.jsonl` — accelerometer and gyroscope samples from the platform
- `.baro.jsonl` — barometric altitude samples
- GPS metadata embedded in video or as a companion `.gpx` or `.kml` file
- MAVLink telemetry logs for autopilot platforms
- ROS bag traces for robotic platforms
- RF, gas, acoustic, ADS-B, environmental, and hyperspectral sidecars

The shared sidecar loader (`pipeline/core/sidecars.py`) produces a time-sorted list
of records, discarding malformed rows silently. The loading result includes a count
of discarded rows so a human can judge sidecar quality without re-inspecting the raw
file.

### Why sidecar failures are tolerated

Sidecars are optional evidence. A video without a GPS sidecar can still produce
valid embeddings, captions, tracking results, and detection counts. A GPS sidecar
that loses signal for 30 seconds of a 5-minute flight should not abort the whole run.
The design principle is: sensor degradation reduces evidence quality and triggers
warnings; it does not trigger run termination unless the degraded modality was
declared required in the run configuration.

### RTSP and MediaMTX

RTSP streams from live cameras are ingested via the realtime path. MediaMTX acts as
an RTSP/RTMP broker: it accepts incoming streams (from drones, IP cameras, or ground
units) and makes them available to the SelfSuvis realtime bridge. This decouples the
camera source from the SelfSuvis ingest client: any camera that can push RTSP can be
ingested without code changes.

---

## 5. Embeddings And Retrieval

### OpenCLIP

OpenCLIP is a family of image-text contrastive models trained on large datasets.
The model learns to embed images and text into a shared vector space so that an
image of a dog and the text "a dog" are close in that space.

In SelfSuvis, CLIP embeddings are used for:
- **Text-to-image search**: a text query is embedded and searched against the frame
  vector collection in Qdrant
- **Image-to-image search**: a query frame is embedded and its nearest neighbours
  are retrieved
- **Semantic similarity scoring**: comparing scene-level embeddings across missions

The CLIP model is loaded via the `open_clip` Python package. Model name and
pretrained weight tag are configured in `CLIP_MODEL` and `CLIP_PRETRAINED` in `.env`.
Common choices: `ViT-B/32` (fast, less accurate) and `ViT-L/14` (slower, more
accurate).

### DINOv2 / DINOv3

DINO models are self-supervised visual models trained without text labels. They
learn visual structure from image patches and produce embeddings that capture
fine-grained visual similarity better than CLIP for cases that have no natural text
description.

In SelfSuvis, DINO embeddings are used for:
- **Visual neighborhood structure**: finding frames that look similar without relying
  on a text description
- **SSL fine-tuning**: the DINO architecture is the base model for the self-supervised
  adaptation step where the system adapts its visual representations to the current
  mission domain
- **Gallery construction**: the `build_gallery.py` script generates `.npz` embedding
  arrays from DINO for the adapted edge model

### Qdrant named vectors and search

A single Qdrant point has two named vectors (`clip` and `dino`), a unique UUID, and
a JSON payload containing the mission ID, video ID, frame path, timestamp, caption,
detection count, and surprise score.

When a user queries "find all frames with dense pedestrian traffic", the pipeline:
1. Encodes the text with OpenCLIP → `clip` vector
2. Runs ANN search against the `clip` collection in Qdrant with a minimum score
   threshold
3. Returns the top-k matching frames with their payload fields

Named vector search means the same frame can be found by visual similarity
(searching `dino`) without re-indexing. This matters for retrieval evaluation:
after SSL fine-tuning, the adapted `dino` vectors should produce better recall than
the base `dino` vectors for mission-specific content.

### Cosine similarity and ANN

Vector similarity in SelfSuvis is measured by cosine similarity: the angle between
two unit-norm vectors. Values near 1.0 mean highly similar; values near 0.0 mean
orthogonal (unrelated); negative values mean the vectors point in opposite
directions. Both CLIP and DINO embeddings are L2-normalised before storage so
cosine similarity equals the dot product.

Qdrant uses the HNSW (Hierarchical Navigable Small World) graph as its approximate
nearest-neighbor index. HNSW is fast for high-dimensional search but may miss true
nearest neighbors for edge cases. The `ef` (search effort) parameter trades speed
for recall; the production default is conservative enough that practical retrieval
quality is close to exact search.

---

## 6. Vision And Language Models

SelfSuvis combines specialist models. No single model replaces the others; they
produce complementary evidence types that later steps combine and cross-validate.

### Gemma (scene analysis and reasoning)

Gemma is the central scene-reasoning model. It runs on video frames (or frame
batches) and produces structured scene descriptions, object listings, domain hints,
and spatial relationship summaries. These outputs are stored in `VideoKnowledge` and
become the primary context for every downstream step.

Gemma runs via Ollama, so the model must be pulled locally before a run
(`ollama pull <GEMMA_API_MODEL>`). The Gemma model tag is configured via
`GEMMA_API_MODEL` in `.env`. Larger models produce richer scene analysis at the cost
of higher latency per frame.

A critical fact: Gemma output is JSON-gated by `json_guard` in the agentic helpers.
If Gemma returns malformed JSON (common when the model is too small or the prompt is
poorly structured), the step logs a warning and uses a safe fallback rather than
crashing. This is the first place to look when scene analysis artifacts are thin or
empty.

### Florence-2 (frame captions)

Florence-2 is a frame-level visual captioning model from Microsoft. It produces a
short natural-language description of each keyframe and a confidence score. The
confidence score is used in the threat primitive layer: low caption confidence is one
evidence source for `visibility_degradation`.

Florence-2 runs locally via the `transformers` library. The model is cached in the
HuggingFace cache directory. Slow first run = first-time download; subsequent runs
are fast. Caption confidence of 0.75 is used as a fallback when the OCR path is
active instead of Florence.

### Whisper (speech-to-text)

Whisper produces timestamped transcripts from audio tracks. In SelfSuvis this is
used when the video or audio sidecar contains mission-relevant speech (operator
voice notes, radio traffic, ATC communication). The transcript segments are stored
with their start and end timestamps and aligned to frames by the sidecar loader.

The ASR window (how wide a time window around each frame is included in context) is
configurable. A too-wide window causes contamination: speech from one scene segment
appears in the context of frames from an adjacent scene. Default is ±2 seconds.

### OCR backends (visible text extraction)

OCR extracts visible text from frames — signs, screen overlays, tail numbers, road
markings, labels, instrument panels. SelfSuvis supports multiple OCR backends
(PaddleOCR, EasyOCR, Tesseract) selected by configuration. OCR runs on prescreened
frames: frames with low text probability (estimated from Florence caption or CLIP
text-saliency) are skipped to avoid wasting time on frames with no text content.

OCR output is stored as a list of `(text, bbox, confidence)` tuples per frame. The
confidence threshold is configurable; below-threshold detections are dropped before
storage.

### Depth models (monocular geometric priors)

Depth estimation runs a monocular depth model (Depth Anything V2 or similar) on
each sampled frame and produces a per-pixel relative depth map. Important
constraint: these are not metric range sensors. They estimate relative depth
structure (near vs far) without physical scale. The output is useful for:
- Estimating near-field occupancy density (fraction of the central image occupied by
  close objects)
- Informing `collision_risk` and `free_space_estimate` in the threat primitive layer
- Providing a geometric prior when IMU/GPS data is absent

Do not treat depth model output as a lidar substitute. Treat it as a soft prior:
"the left half of the frame appears to be closer than the right half" is evidence;
"the detected object is 4.2 metres away" is not.

### YOLO11 / RF-DETR (object detection)

Two detection families serve different needs:

**YOLO11** — the fast detector. Used for real-time-feasible detection on long videos
or when GPU resources are limited. Produces class, bounding box, and confidence for
each detected object. The primary detector for tracked-object counting and occupancy
estimation.

**RF-DETR** — a transformer-based detector. Slower than YOLO but produces track IDs
from its multi-head output, which makes it the preferred detector for the tracking
pipeline. Track IDs from RF-DETR are what the SSL fine-tuning step uses to generate
identity-consistent positive pairs (see [14_temporal_ssl_physical_state.md](14_temporal_ssl_physical_state.md)).

Detection results are stored in `detection_results.json`. The anomaly detector flags
frames where the detection count is more than 2 standard deviations above the
per-video mean — a statistical hint that something unusual is present.

### SAM2 / SAM3 (segmentation)

SAM (Segment Anything Model) produces pixel-precise masks from prompts. In SelfSuvis
it is used in two modes:
1. **Prompted by detection boxes**: each YOLO/RF-DETR bounding box is used as a SAM
   prompt to produce a tighter pixel mask of the detected object
2. **Automatic mask generation**: when no detection results are available, SAM
   generates masks from its own internal proposal mechanism

Masks are stored as RLE-encoded arrays in the detection JSONL. They are used by the
occupancy estimator in the threat primitive layer to compute the fraction of the
near-field image area covered by detected objects.

### Qwen (per-frame multimodal reasoning)

Qwen is the dense per-frame VLM. It runs on a selected subset of frames (bounded
by `QWEN_MAX_FRAMES` to avoid unbounded cost on long videos) with the accumulated
`VideoKnowledge` context as part of its prompt. It produces scene narratives,
object-relation summaries, and anomaly flags that are richer and more contextual
than Florence captions.

Qwen runs via Ollama. The reasoning model (Qwen3 or DeepSeek-R1) is pulled
separately: `ollama pull qwen3:14b`. The step uses a persistence filter: frames
where the scene has not changed significantly since the last Qwen inference are
skipped to avoid redundant compute.

### UniDriveVLA (structured perception + planning)

UniDriveVLA is a multimodal vision-language-action model designed for autonomous
driving-style perception. In SelfSuvis it serves as the expert reasoning step:
given a frame and the accumulated context, it produces:
- Structured perception output: detected objects with type and occupancy role
- Drivable area estimate (fraction of the frame navigable)
- Hazard list (free-text, for human reading)
- Mixture-of-experts outputs: multiple expert heads vote on scene classification

UniDriveVLA runs via an OpenAI-compatible vision endpoint (configured as
`UNIDRIVEN_API_URL`). This means it can be pointed at a local vLLM server, an
Ollama endpoint, or a remote model API without code changes.

The MoE (mixture-of-experts) output agreement score is computed for each frame. Low
agreement between experts (`mean_moe_agreement < 0.5`) flags the frame as
ambiguous. This score is surfaced in the agentic trace for the audit step.

### SceneTok (optional streaming scene encoder)

SceneTok is an optional streaming scene encoder and segmentation decoder. When
enabled (`SCENETOK_ENABLED=true`) and when the API is reachable, it produces
continuous temporal scene tokens that capture scene-level semantic continuity across
frames. It requires at minimum 24 GB VRAM (RTX 4090 class). When unavailable, the
pipeline falls back to loading a local SceneTok-compatible model weight from cache.

---

## 7. Temporal State, Fusion, And Mapping

This section covers the state-estimation layer — where SelfSuvis becomes more than
a captioning pipeline. These technologies require a probabilistic mental model:
systems with state, noise, uncertainty, and explicit failure modes.

### RSSM temporal surprise

The Recurrent State Space Model (RSSM) is a lightweight sequence model trained over
CLIP embeddings. It learns to predict the next frame embedding given the history of
prior embeddings. Frames where the actual embedding is far from the predicted
embedding (high reconstruction error) get a high `surprise_score`.

High surprise means the frame contains content that the recent scene history did not
predict — a useful proxy for scene transitions, sudden events, or anomalies. The
RSSM model is small enough to run on CPU if no GPU is available.

### Kalman filters

Kalman filters are the core state estimator for both platform pose and tracked
objects. The mathematical intuition:
- **State**: a vector of quantities you want to estimate (position, velocity,
  orientation)
- **Prediction step**: advance the state forward in time using a motion model (with
  process noise added)
- **Update step**: pull the state estimate toward a new measurement (weighted by the
  measurement noise vs the prediction uncertainty)
- **Covariance**: a matrix that tracks how uncertain the estimate is

The filter produces a posterior estimate that is more accurate than either the
motion model prediction alone or the raw measurement alone, because it weighs them
by their respective uncertainties.

In SelfSuvis there are two distinct Kalman filters:
1. **Platform Kalman** (in `pipeline/fusion/`): fuses GPS, IMU, and barometric
   altitude into a position/velocity posterior for the whole platform over the video
2. **Object Kalman** (per-track): maintains a 2D state (bounding box centroid +
   velocity) for each tracked object, updated by detector outputs frame by frame

### Mahalanobis gating

Before updating an object Kalman filter, each incoming detection must be associated
with an existing track. Mahalanobis gating rejects detections that are too far from
the predicted track location — where "far" is measured in units of standard
deviations of the track's current uncertainty, not raw pixels.

This matters because a detection 100 pixels away from a track is implausible if the
track uncertainty is small (the object should be near the predicted location) but
plausible if the track uncertainty is large (the object may have moved). Using raw
pixel distance would either over-reject (when uncertainty is high) or accept
impossible associations (when uncertainty is low).

### Hungarian assignment

The Hungarian algorithm solves the one-to-one assignment problem: given N
detections and M active tracks, find the assignment that minimises total cost
(typically the sum of Mahalanobis distances). It runs in O(n³) time, which is
acceptable for the typical frame-to-frame track counts in SelfSuvis.

After assignment, unassigned detections become new track candidates; unassigned
tracks with too many missed frames are terminated.

### RTS smoothing

The Rauch-Tung-Striebel (RTS) smoother runs backward over the track after the
forward Kalman filter has processed all frames. It improves the state estimate at
each frame by incorporating future measurements that were not available when the
forward filter ran. The result is a smoother, more consistent trajectory.

RTS smoothing is applied after the full video is processed. It requires storing the
forward-pass covariances for all frames, which uses memory proportional to the
track length. Long tracks in long videos may hit memory pressure; this is why the
fusion step runs after all frames are extracted, not frame by frame.

### SfM / pycolmap

Structure from Motion (SfM) estimates camera poses and sparse 3D structure from
image correspondences across frames. The key output is a set of 6DoF camera poses
(position and orientation in a consistent world frame) and 3D point cloud
representing scene geometry.

SelfSuvis uses `pycolmap` (Python bindings to COLMAP) as the SfM backend. The
quality of the reconstruction degrades with:
- Short clips (fewer frames = fewer baselines for triangulation)
- Low-texture or repetitive scenes (fewer reliable correspondences)
- High-speed motion with motion blur (feature matching fails)
- Single-viewpoint capture without crossover (no geometric baseline)

When SfM produces fewer than 50 points or fewer than 20 registered poses, the
`map_degraded` flag is set. This is one of the evidence sources for the
`pose_uncertain` threat primitive.

### Gaussian Splat / mapping outputs

After SfM establishes camera poses, the mapping step produces inspectable 3D scene
artifacts. Gaussian Splatting is the dense reconstruction method: it fits 3D
Gaussian ellipsoids to represent the scene, enabling real-time novel view synthesis.
The output `.splat` file can be loaded in a 3D viewer to inspect the reconstructed
scene from any viewpoint. The Gaussian Splat is a visual product, not a navigation
map; it is primarily used for mission review, not for live autonomy decisions.

---

## 8. Local Workflow, LangGraph, And Runtime Adaptation

### The 32-step local runner

The monolithic local runner reports **32 runtime/post-run steps**. All steps run
sequentially in a single Python process. The learning-path documents sometimes use
a broader 36-step conceptual breakdown for study purposes; the runtime view is
always the 32 steps in `runner.py`.

The runner is the correct entry point for understanding which step produces which
artifact and which step depends on which earlier artifact. Read it top-to-bottom as
a script, not as a class hierarchy.

### Startup preflight and the startup contract

Before step 1 runs, the preflight module checks all required conditions:
- Python packages (`open_clip`, `transformers`, `pycolmap`, etc.)
- Cached model weights on disk
- Optional service reachability (Qdrant, mapper API)

A failed required check stops the run before any frames are extracted. This
produces a cleaner failure: the error message names the missing item and suggests
the fix, rather than a cryptic `ImportError` or `FileNotFoundError` deep inside
step 7. Always run `selfsuvis preflight` when setting up a new machine to verify
the environment before a real run.

### LangGraph optional path

The optional LangGraph path (`SELFSUVIS_USE_GRAPH=1`) is an alternative execution
engine for the same pipeline steps. Instead of a sequential `for` loop, the steps
are registered as nodes in a LangGraph state graph with:
- Checkpointing: the graph state is persisted after each node so a failed run can
  resume from the last completed node
- Parallel fan-out: several evidence extraction steps (ASR, OCR, Florence, depth)
  run as parallel branches
- Agentic guardrails: LLM-heavy nodes get additional retry and critique wrappers

LangGraph adds overhead and complexity. The default is the monolithic runner.
LangGraph is useful for very long videos where checkpointing makes recovery
practical, or for experimentation with parallel evidence extraction.

### Runtime gates and optimization

Several steps apply bounded frame selection to avoid wasting compute:

- **OCR**: frames with low text-saliency probability are skipped
- **Qwen**: only up to `QWEN_MAX_FRAMES` frames per video are processed; subsequent
  frames reuse the most recent Qwen context
- **SSL fine-tuning gate**: if the embedding quality diagnostics show clearly
  unhealthy representations, downstream distillation and ONNX export are skipped
  rather than producing a broken edge model

These gates are not failures — they are intentional degraded-mode behaviors. The
analytics step records which gates fired so a human can review coverage.

### analysis_summary.json and local analytics

After the 32-step run completes, the analytics module (`src/selfsuvis/analytics/`)
inspects all output artifacts and writes `analysis_summary.json`. This file
summarises:
- Modality coverage: which sensors, sidecars, and VLMs contributed evidence
- Degradation flags: which steps fired quality-gate warnings
- Tracking quality: mean track length, break rate, n_tracks
- Mapping quality: n_sfm_poses, n_3d_points, map_degraded flag
- Training quality: SSL loss curves, distillation convergence, ONNX export status
- Artifact health: which expected output files are present or missing

The correct human habit after any local run: open `analysis_summary.json` before
reading any other artifact. It gives the coverage picture in one place.

### Utilyze profiling

`scripts/ssv/ssv-utilyze.sh` wraps the `utlz` profiling tool with defaults
suited to selfsuvis local runs. It:
- Disables upstream workload metrics (privacy/telemetry off by default)
- Writes logs to `.data/reports/utilyze.log`
- Passes additional `utlz` flags through verbatim

Install with `scripts/install/install_utilyze.sh` (first time only). Use Utilyze when
diagnosing slow local runs: it shows GPU utilisation, CPU wait, memory pressure, and
per-process resource allocation. If a step that should be GPU-bound shows low GPU
utilisation, Utilyze will reveal whether the bottleneck is CPU-side preprocessing,
memory transfers, or actual model compute.

---

## 9. Training, Distillation, And Edge Export

### Self-supervised fine-tuning (SSL)

Self-supervised learning (SSL) adapts the DINO-style visual encoder to the current
mission domain without human labels. The key concept: instead of using manual
annotations, the system generates its own training signal from the video data.

In SelfSuvis, the positive pairs for contrastive SSL are generated from:
1. **Track-based pairs**: two crops of the same RF-DETR track ID across nearby
   frames are a positive pair (same identity = should have similar embeddings)
2. **Augmentation-based pairs**: two augmented views of the same frame crop are a
   positive pair
3. **Cycle-consistency constraint**: if frame A is close to B, and B is close to C,
   then A and C should be at a predictable distance (prevents representation drift
   along long tracks)

The SSL training runs for a configurable number of epochs on the mission's own frame
crops. After training, the adapted DINO encoder is evaluated: if the
nearest-neighbour recall (mean cosine similarity to top-k neighbours) is worse than
the base model, the adapted weights are discarded and the base model is retained.

### Knowledge distillation

Distillation compresses the larger teacher behavior into a smaller student model.
In SelfSuvis:
- **Stage 1 distillation**: the adapted DINO encoder (teacher) supervises a smaller
  vision transformer (student) using soft targets (probability distributions) rather
  than hard labels
- **Stage 2 distillation**: targets EfficientViT-style architectures optimised for
  edge deployment on Arm/Rockchip hardware

The distillation loss combines three terms: the student-teacher feature similarity
loss, the student self-prediction loss, and the student classification loss (if a
labeled evaluation set is available).

### ONNX export and edge models

After distillation, the student model is exported to ONNX format. ONNX is an
open interchange format for neural network models that can be run by multiple
inference runtimes (ONNX Runtime, TensorRT, RKNN, etc.) without requiring PyTorch.

The export produces two ONNX variants:
- `model_fp32.onnx` — full float32 precision, correct output, larger file
- `model_int8.onnx` — post-training int8 quantization, smaller and faster, slightly
  lower accuracy

RKNN conversion (for Rockchip NPU targets) is a separate step that converts the
ONNX model to RKNN format using the Rockchip RKNN Toolkit. This runs on a Linux
x86 host and produces a `.rknn` binary that can be loaded on RV1106G3 and similar
SoCs.

### Drone detection edge training

The drone detection step trains a YOLO-family object detector on mission-specific
drone image data for deployment on constrained edge hardware (Cortex-A76, RV1106G3).
Key aspects:
- Hard negative injection: non-drone aerial objects (birds, balloons) are included
  in training to reduce false positives in the field
- ONNX fp32 and int8 export: same two-variant export as the visual encoder
- RKNN NPU conversion for edge SoC deployment
- Evaluation on a held-out validation set before the model is promoted to "ready for
  deployment"

The correct question after drone detection training is not "did training finish?"
but "did the evaluation metrics (mAP, precision, recall at IoU 0.5) improve over the
base weights, and was false positive rate on hard negatives acceptable?"

---

## 10. Realtime And coop

### MediaMTX (RTSP/RTMP broker)

MediaMTX manages RTSP and RTMP stream paths. Any camera or drone that can push an
RTSP or RTMP stream can be received by MediaMTX without code changes. SelfSuvis's
realtime bridge connects to MediaMTX to receive live video frames and ingest them
into the realtime processing path. MediaMTX configuration lives in `docker/` and
can be tuned for number of concurrent streams and protocol compatibility.

### Mosquitto MQTT (IoT message broker)

MQTT is a publish-subscribe messaging protocol for constrained IoT devices.
Mosquitto is the MQTT broker: it receives messages from field sensors and makes them
available to subscribers. In SelfSuvis, the coop module subscribes to MQTT
topics from:
- Environmental sensors (temperature, humidity, gas concentration)
- Acoustic sensors (sound event detection)
- Frigate (camera-object event notifications)
- LoRaWAN gateway forwarded messages

MQTT topics follow a tree structure (`site/sensor_id/reading_type`). The subscriber
in `coop/sensors/mqtt_subscriber.py` handles connection, reconnection, and
message dispatch. Always configure `MQTT_BROKER_HOST` and `MQTT_BROKER_PORT` in
`.env`; the default localhost port is 1883.

### ChirpStack (LoRaWAN network server)

ChirpStack is the LoRaWAN network server. LoRaWAN is a low-power wide-area radio
protocol for field sensors that need to transmit small payloads over long distances
with battery-only power. ChirpStack decodes the LoRaWAN uplinks (raw radio payloads)
into JSON application payloads, which are then forwarded to the MQTT broker.

The chain is: sensor → LoRa radio → ChirpStack → MQTT → coop. A learner
should understand that LoRaWAN imposes strict duty-cycle limits (sensors transmit
infrequently, typically every 30–300 seconds) so rolling site state must be designed
for sparse, intermittent updates rather than high-frequency streaming.

### Frigate (camera-object detection)

Frigate is an open-source NVR (network video recorder) with integrated object
detection. It receives RTSP camera streams, runs detection on them locally, and
publishes object detection events to MQTT. In SelfSuvis, Frigate events are one of
the input streams to coop's rolling site state.

The event format includes camera name, detected object class, confidence, bounding
box, and snapshot frame path. coop maps these to site sectors using the camera
geometry configuration.

### coop rolling site state

coop aggregates inputs from all active IoT and camera sources and maintains a
rolling site state: a time-windowed summary of sensor readings, object detections,
acoustic events, and threat signals for each geographic sector of the monitored site.

The rolling state answers questions like:
- "What was the threat level for sector B in the last 5 minutes?"
- "Which cameras have detected any motion in the last 30 seconds?"
- "What was the gas sensor reading at the north perimeter 10 minutes ago?"

The rolling window is configurable per modality. Older readings are expired from
state, but they remain in the PostgreSQL sensor table for retrospective analysis.

### Scene synthesis and realtime threat

coop's scene synthesiser combines the rolling site state with VLM-based scene
interpretation to produce natural-language situation reports for operators. Threat
sector analysis maps sensor evidence to site sectors and produces per-sector threat
levels using the same two-source gate logic as the local pipeline threat primitive
layer.

The operational significance: a local mission video run produces a mission-level
threat assessment after the fact; coop produces a live site-level threat
assessment in near real time. They share the same primitive scoring logic but
different execution paths.

---

## 11. What To Learn First

For a human starting from limited background, learn in this order:

1. **Python package layout, CLI entry points, and artifact directories.** Run `make
   venv`, then `selfsuvis --help`. Understand where output goes.
2. **`pipeline/core/env.py` and `preflight.py`.** Understand how configuration is
   loaded and how startup checks work before reading any model code.
3. **FFmpeg frame extraction and timestamps.** Understand `frames/<video_id>/` and
   why the timestamp is the canonical key for all evidence.
4. **CLIP/DINO embeddings, cosine similarity, and Qdrant retrieval.** Run a text
   search query and inspect the returned frame paths and scores.
5. **`analysis_summary.json`.** After a local run, read this file first. Understand
   every field before looking at individual step artifacts.
6. **FastAPI/PostgreSQL/Qdrant service flow.** Trace a `/query` request from HTTP
   headers through `deps.py`, through the router, into Qdrant search, and back.
7. **Florence/Gemma/Qwen evidence types and how `VideoKnowledge` carries context.**
   See [07_agentic_knowledge_flow.md](07_agentic_knowledge_flow.md).
8. **Depth, detection, segmentation, and tracking basics.** Understand what artifact
   each model produces and which later steps consume it.
9. **Security: `APP_ENV`, `API_AUTH_REQUIRED`, HMAC, fail-closed defaults.**
   Understand why missing configuration produces errors rather than open access.
10. **Sensor fusion: clocks, coordinate frames, calibration, uncertainty.** Read
    [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md).
11. **Kalman filtering, assignment, smoothing, and map registration.** Read
    [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md).
12. **SSL, distillation, ONNX, and edge model evaluation.** Understand what
    "successful adaptation" means from `analysis_summary.json`, not from training
    loss alone.
13. **Realtime streams, MQTT, coop, and operational security boundaries.**
    Read `16_coop_pilot_iot_edge_monitoring.md`.

Do not try to learn every model first. Learn what each artifact means, what
produced it, and what later stage consumes it. The pipeline makes more sense read
from outputs back to code than from code outward.
