# Tracking, World Models, And 3D Mapping: Steps 21-27

This phase turns evidence into structure.
The pipeline shifts from "what is in this frame?" to "what persists across frames, how does it evolve, and where does it exist in space?"

The core move is from frame-wise snapshots to temporal continuity and spatial geometry.

Note: these step numbers are part of the broader conceptual learning path.
In the current local runner, the corresponding capabilities are grouped into the 23
top-level execution steps documented in [`pipeline.md`](../reference/pipeline.md).

---

<a id="step-21-yolo--sam-detection-and-segmentation"></a>
## Step 21. YOLO + SAM detection and segmentation

**What it does:**
Run YOLO (or RF-DETR) over each keyframe to produce fast bounding box detections.
Then run SAM (Segment Anything Model) to refine those detections into pixel-accurate masks.
Combine box + mask into a structured per-frame object list.

**Why it matters:**
Step 8 provided approximate detections as context labels.
This step provides pixel-level masks that enable:
- Spatial reasoning: where exactly is the object, how large is it, is it occluded?
- Instance separation: which pixels belong to vehicle A vs vehicle B?
- Foundation for tracking: a mask is a far better starting region for tracking than a bounding box.
- Semantic graph building: objects with masks can be connected spatially ("vehicle A is to the left of building B").

**Implementation:**
- [`ssv_vdp/steps/yolo_sam.py`](../../packages/ss-fusion/src/ssv_vdp/steps/yolo_sam.py)
- [`pipeline/vision/yolo.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/yolo.py)
- [`pipeline/vision/sam.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/sam.py)

**Key concepts:**

*YOLO vs RF-DETR:*
YOLO11 is a single-stage anchor-based detector optimized for speed.
RF-DETR is a transformer-based detector with better accuracy on small or overlapping objects but slower inference.
The pipeline supports both; `RFDETR_ENABLED` controls which is active.
For aerial footage with many small vehicles, RF-DETR typically outperforms YOLO.

*SAM prompt modes:*
SAM can be prompted in multiple ways:
- Box prompt: give a bounding box from YOLO → SAM refines it to a mask.
- Point prompt: click one or more points inside an object → SAM grows a mask.
- Auto-mask (no prompt): SAM generates all possible masks in the image without guidance.
Box-prompted SAM is used here because YOLO boxes provide good spatial priors.

*Mask quality vs computational cost:*
SAM produces high-quality masks but is expensive.
The pipeline only runs SAM on frames where YOLO produced high-confidence detections.
Low-confidence frames get no masks to save compute.

**Output artifact:**
Per-frame object list with bounding box, class label, confidence, and RLE-encoded mask.
Optionally: overlay images showing masks on frames.

**Human focus:**
- Understand why bounding boxes are insufficient for spatial reasoning: a box around a vehicle includes road, sky, and adjacent vehicles.
- Learn what SAM's "everything" mode produces vs prompted mode: everything mode is much slower and noisier.
- Know the NMS (Non-Maximum Suppression) step: YOLO produces many overlapping boxes; NMS keeps only the highest-confidence one per object.
- Understand mask IoU (Intersection over Union) as the standard quality metric for segmentation.

**Common failure modes:**
- YOLO misses small objects at altitude → SAM never runs; those objects are invisible to later tracking.
- YOLO box is inaccurate → SAM mask extends into wrong region.
- SAM out of memory on high-resolution frames → reduce input resolution or disable SAM on long videos.
- Motion blur → YOLO produces wide uncertain boxes; masks are coarse.

---

<a id="step-22-gemma-directed-tracking"></a>
## Step 22. Gemma directed tracking

**What it does:**
Sample frames from the video, send them to Gemma with a structured question: "What object categories should be tracked in this mission?".
Parse the JSON response to extract a list of object categories and approximate bounding box regions.
Use SAM to segment those Gemma-specified objects.
Use RF-DETR to track Gemma-priority classes across the frame sequence using IoU-based multi-frame matching.

**Why it matters:**
Standard tracking (Step 21) tracks everything the detector finds.
That includes irrelevant background objects that add noise without mission value.
Gemma-directed tracking inverts the priority: language understanding steers perception.
If Gemma determines "this is a vehicle convoy", tracking focuses on vehicles and ignores trees and buildings.
This is the step where reasoning starts directing perception, not just describing it.

**Implementation:**
- [`ssv_vdp/steps/gemma_tracking.py`](../../packages/ss-fusion/src/ssv_vdp/steps/gemma_tracking.py)
- [`pipeline/vision/rfdetr.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/rfdetr.py) — `RFDETRTracker`

**Key concepts:**

*Language-guided perception:*
Traditional computer vision: detect everything, then filter by label.
Language-guided perception: ask what to look for first, then run targeted detection.
The distinction matters at scale: running full detection on 1000 frames is expensive; running targeted detection on 1000 frames for 2 classes is much cheaper.

*IoU-based tracking (greedy matching):*
The tracker maintains a set of active tracks.
For each new frame, it computes the IoU (Intersection over Union) between each new detection and each existing track.
A new detection is matched to the track with highest IoU if that IoU exceeds 0.45.
If no track matches, a new track is started.
Track IDs are reset per video (not persistent across missions).

*Context accumulation into VideoKnowledge:*
Tracking results are stored in `frame_facts_json["gemma_tracking"]`.
This allows later steps (Qwen, report generator) to query what was being tracked and what the tracking found.

**Output artifact:**
`gemma_tracking_summary.md` in the video output directory: which categories Gemma prioritized, which tracks were maintained, and how track continuity compares across frames.

**Human focus:**
- Understand the IoU matching criterion: two boxes with IoU < 0.45 are considered different objects; tracks break at that threshold.
- Learn when tracking breaks: fast motion, occlusion, re-entry of an object after it left the frame.
- Know what happens when Gemma is unavailable: the system falls back to tracking all high-confidence YOLO detections.
- Understand why tracking priority matters for a mission report: a tracked vehicle that disappears is more significant than a detected tree in frame 23.

**Common failure modes:**
- Gemma unavailable → directed tracking skips; only default YOLO tracking runs.
- Gemma returns ambiguous categories ("object") → tracker targets too broad a class; everything matches.
- Fast-moving objects → IoU between consecutive frames is below 0.45 even for the same object; track breaks and restarts.
- Object leaves frame temporarily → new track ID assigned on re-entry; track history breaks.

---

<a id="step-23-world-model-video-embeddings"></a>
## Step 23. World model video embeddings + RSSM temporal surprise

**What it does:**
Two complementary temporal passes run here:

1. **Heavy world model** (`WORLD_MODEL_ENABLED=true`): Encode short video clips (groups of consecutive kept frames) into temporal embeddings that capture motion and appearance jointly using a large pre-trained video model (V-JEPA2, VideoMAEv2, Cosmos, etc.).

2. **RSSM temporal surprise** (`DREAMER_ENABLED=true`, always on by default): Train a lightweight Recurrent State Space Model (RSSM) on the mission's CLIP embedding sequence, then compute a per-frame *surprise score* — how unexpected was each frame given the RSSM's prediction from prior context?

**Why the RSSM matters:**
The RSSM surprise signal is derived from the DreamerV3 architecture (Romero et al., ICRA 2026, "Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight"). In the original paper, the RSSM is used to train a drone control policy entirely in an imagined latent world. Here, the key insight is repurposed for passive video analysis:

> A frame with high RSSM surprise was unexpected given the video's recent temporal context. These frames are the most informative for annotation — they represent genuinely novel scenes, abrupt environment changes, or unusual events that a static per-frame distance metric would miss.

**RSSM architecture (lightweight, CPU-only):**
```
Encoder:   CLIP_embed (512-d) → Linear → [μ_k, log σ²_k] → z_k (32-d, reparameterized)
Recurrent: GRU(z_k, h_{k-1}) → h_k (256-d recurrent state)
Dynamics:  Linear(h_k) → z̃_{k+1} (predicted next latent)
Surprise:  cosine_distance(z̃_k, z_k) — the prediction error
```

Online training: 20 gradient steps on the current mission's CLIP embeddings. No pre-trained weights. ~50ms on CPU for a typical mission.

**Impact on hydrated edge models (Steps 28–30):**
The RSSM surprise score feeds directly into the active learning formula:
```
With RSSM:    al_score = 0.35×dino_dist + 0.25×(1-caption_confidence) + 0.40×rssm_surprise
Without RSSM: al_score = 0.60×dino_dist + 0.40×(1-caption_confidence)
```
The 40% weight on RSSM surprise means temporally-novel frames — scene transitions, first appearances of new objects, environment changes — rank higher in the annotation queue. Better-selected training frames → better SSL fine-tuning (Step 28) → better distilled edge models (Step 29) → more accurate hydrated ONNX exports (Step 30).

**Implementation:**
- [`models/rssm_model.py`](../../packages/ss-perception/src/selfsuvis/models/rssm_model.py) — `RSSMEmbedder`: encoder, GRU, dynamics, online training
- [`pipeline/analysis/active_learning.py`](../../packages/ss-fusion/src/selfsuvis/pipeline/analysis/active_learning.py) — `assign_al_tags` with `rssm_surprises` parameter
- ``pipeline/workflows/indexer/`` — `_run_al_rssm_pass`, `_run_world_model_pass`
- [`pipeline/vision/world.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/world.py) — `WorldModel` (heavy video backbone)

**RSSM output stored in `frame_facts_json["rssm"]`:**
```json
{
  "surprise_score": 0.82,
  "method": "rssm",
  "model": "RSSMEmbedder"
}
```
When `DREAMER_STORE_TEMPORAL=true`, also stores the 256-d recurrent state `h_k` for downstream temporal similarity search.

**Key concepts:**

*Temporal vs spatial features:*
CLIP and DINO process one frame at a time.
Video models (V-JEPA2, VideoMAEv2) process a stack of frames simultaneously.
The RSSM adds a third perspective: sequential prediction — what should come next based on what came before?

*Why RSSM runs on CLIP embeddings, not raw pixels:*
The original DreamerV3 encodes raw pixels. Here we operate in CLIP embedding space because:
1. CLIP embeddings are already computed (zero extra cost)
2. They carry rich semantic features that matter for annotation priority
3. The RSSM stays CPU-friendly (~100K parameters vs hundreds of millions for pixel-space models)

*Surprise ≠ motion:*
A high-surprise frame is not necessarily one with a lot of motion.
A sudden appearance of a new object type, a lighting change, or an environment transition produces high surprise even with slow camera motion.
Low-surprise frames may have fast motion through a uniform environment (fields, water) — the RSSM has learned to predict these.

*EMA fallback:*
If PyTorch is unavailable, the RSSM falls back to an exponential moving average (EMA) of CLIP embeddings. The surprise signal degrades gracefully to a simpler temporal novelty measure.

**Output artifact:**
Per-frame `rssm_surprise` in `frame_facts_json["rssm"]`.
Per-frame `al_score` and `al_tag` populated in `frames` table (written to PostgreSQL by worker).
World model clip embeddings in `frame_facts_json["world_model"]` when `WORLD_MODEL_ENABLED=true`.

**Human focus:**
- Understand the intuition: the RSSM learns the video's temporal rhythm and flags frames that break it.
- Learn how this connects to active learning: frames worth annotating are those where the model's predictions fail, not just frames that look different from a random average.
- Know when RSSM surprise helps most: long monotonous missions where DINO distance is uniformly low, but specific frames mark important environmental transitions.
- Compare a run with `DREAMER_ENABLED=true` vs `DREAMER_ENABLED=false`: check whether the top-scored `needs_annotation` frames correspond better to actual scene changes.

**Common failure modes:**
- Very short mission (< 10 frames) → RSSM has insufficient sequence for training; falls back to EMA.
- First frame always has surprise=0 (no prior context to predict from); this is expected.
- Highly repetitive mission → RSSM surprise is uniformly low; DINO distance dominates the AL score.
- PyTorch unavailable → EMA fallback activates; surprise signal is coarser but still useful.

---

<a id="step-24-qwen-detailed-captioning"></a>
## Step 24. Qwen detailed captioning

**What it does:**
For each keyframe, build a full multi-source context string from `VideoKnowledge.context_for_frame(t_sec)` and send it — along with the frame image — to Qwen-VL.
Request a structured JSON response containing: vehicle groups, road surface, traffic conditions, infrastructure, and safety observations.

**Why it matters:**
This is the densest reasoning step in the pipeline.
Qwen receives not just the image but accumulated context from every earlier step:
- Florence caption (what this frame looks like)
- Scene segment (what phase of the mission this is)
- ASR text (what was said near this frame)
- OCR text (what text was visible)
- Depth profile (near/far geometry)
- Detected objects (what was found by detection)
- Previous Qwen state (what the last frame contained)

The output is not a caption but a structured observation: a reasoning result that downstream steps can consume directly.

**Implementation:**
- [`ssv_vdp/steps/caption/`](../../packages/ss-fusion/src/ssv_vdp/steps/caption)
- [`pipeline/vision/qwen.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/qwen.py)
- [`ssv_vdp/steps/common.py`](../../packages/ss-fusion/src/ssv_vdp/steps/common.py) — `VideoKnowledge.context_for_frame()` and `update_qwen_state()`

**Key concepts:**

*Multimodal prompt packing:*
Qwen receives a prompt that combines: a system message with domain context, a user message with the full context string, and the image.
The prompt is carefully ordered: system → context → image → question.
This ordering matters: models attend differently to information presented early vs late in context.

*Rolling state:*
After processing each frame, `update_qwen_state()` stores the result in `VideoKnowledge._last_qwen`.
The next frame's context includes a `[Prior frame state]` line showing what Qwen found in the previous frame.
This lets Qwen track continuity: "the three vehicles present in the prior frame are now four" is a valid observation.

*Structured JSON output:*
Qwen is prompted to return JSON, not free text.
This makes the output machine-readable for later report generation.
If the output is not valid JSON, `parse_error` is flagged and `update_qwen_state()` skips storing it (to avoid propagating bad state).

**Output artifact:**
`detailed_captions.md` in the video output directory: per-frame structured observations in Markdown table format.
Raw JSON responses saved alongside for debugging.

**Human focus:**
- Read a full `context_for_frame()` string and understand what information each line comes from.
- Compare the output for the same frame with and without prior context: how does the rolling state change the response?
- Find a frame where the wrong prior state from the previous frame propagated into an incorrect Qwen observation.
- Understand why JSON structured output is better than free text for downstream reasoning.

**Common failure modes:**
- Qwen API unavailable → step skipped for that frame; output has gaps.
- Context string too long → prompt exceeds model context length; earlier evidence is truncated.
- Prior state is wrong (from a bad previous frame) → error propagates forward until a clear scene change resets it.
- JSON parse failure → Qwen returned free text or malformed JSON; that frame's state is not stored; chain resets.

---

<a id="step-25-unidrivevla-expert-analysis"></a>
## Step 25. UniDriveVLA expert analysis

**What it does:**
Send selected keyframes (or the full sequence) to UniDriveVLA, a vision-language-action model pretrained on driving and outdoor autonomy scenarios.
Receive domain-specific structured analysis: scene understanding, object relationships, trajectory predictions, and recommended actions.

**Why it matters:**
Qwen is a general-purpose multimodal reasoner.
UniDriveVLA is a domain-specific expert trained on driving scenarios.
For missions involving road networks, vehicle behavior, and outdoor navigation, UniDriveVLA provides:
- Higher-quality scene understanding for domain-specific events (turn signals, lane changes, right-of-way conflicts)
- Action-oriented interpretation ("the vehicle should slow down") rather than just description ("a vehicle is near the intersection")
- Standardized driving-domain taxonomy for cross-mission comparison

**Implementation:**
- [`pipeline/vision/unidrive.py`](../../packages/ss-perception/src/selfsuvis/pipeline/vision/unidrive.py) — thin OpenAI-compatible HTTP adapter
- [`ssv_vdp/steps/caption/`](../../packages/ss-fusion/src/ssv_vdp/steps/caption) — `step_unidrive_analysis()`
- [`pipeline/core/config/`](../../packages/ss-perception/src/selfsuvis/pipeline/core/config) — `UNIDRIVE_*` settings
- [`pipeline/core/preflight.py`](../../packages/ss-perception/src/selfsuvis/pipeline/core/preflight.py) — startup cache/dependency checks for local runs and production startup
- Model prep: `python -m selfsuvis.scripts.prepare_models --unidrive`

**Key concepts:**

*VLA (Vision-Language-Action) models:*
A VLA extends a VLM (Vision-Language Model) with an action head.
Where a VLM produces text descriptions, a VLA produces both descriptions and action recommendations.
UniDriveVLA is trained on large-scale driving datasets; its action head is calibrated for road navigation scenarios.

*Adapter design — why not run UniDriveVLA directly:*
The upstream UniDriveVLA checkpoint (`owl10/UniDriveVLA_Nusc_Base_Stage3`) is trained on
multi-camera nuScenes format data.
It expects a specific input format that does not match arbitrary single-camera mission video.
The selfsuvis implementation uses a thin HTTP adapter (`UniDriveVLAModel`) that talks to
any OpenAI-compatible vision endpoint and requests the UniDriveVLA structured output schema.
This means **any capable VLM** can be the backend — the driving structure comes from the
prompt, not from the model's training data.
For non-road missions (aerial, maritime, off-road), point `--unidrive-api-url` at
a Qwen2.5-VL-7B sidecar, which handles the schema equally well without domain mismatch.

*Domain-specific vs general-purpose reasoning:*
General VLMs (Qwen, GPT-4V) can reason about any scene but lack calibrated priors for specific domains.
Domain-specific VLAs have strong priors for their training domain but perform poorly outside it.
For aerial drone footage of rural terrain, a driving-specific model may be less useful than a general model.
The adapter pattern lets you swap backends depending on mission type without changing the pipeline.

*Output structure:*
The adapter normalises any backend response into a four-key schema:
- `understanding`: scene summary, traffic context, risk level, key agents
- `perception`: object list with salience, drivable-area estimate, lane structure
- `planning`: recommended action, trajectory hint, hazards
- `mixture_of_experts`: consensus summary, expert agreement, disagreement points
Recommended actions are advisory interpretations ("reduce speed"), not robot commands.

*Backend selection:*
- Road / urban missions: use `owl10/UniDriveVLA_Nusc_Large_Stage3` if vLLM bridge is available
- Aerial / off-road / maritime: use `Qwen/Qwen2.5-VL-7B-Instruct` as backend (avoids domain mismatch)
- Low VRAM (< 8 GB): use `Qwen/Qwen2.5-VL-3B-Instruct`
- If `UNIDRIVE_API_URL` is empty, the local pipeline now expects the HF weights to be cached already;
  startup preflight catches cold-cache runs before processing begins

**Output artifact:**
`unidrive_analysis.md` in the video output directory: per-frame UniDriveVLA structured analysis.
`multi_model_comparison.md` when both Qwen and UniDrive are enabled (side-by-side comparison).

**Human focus:**
- Compare `detailed_captions.md` (Qwen) vs `unidrive_analysis.md` (UniDriveVLA) for the same frame: what does the domain expert add?
- Read `pipeline/vision/unidrive.py` and understand how `_build_user_content()` packs image + context into the OpenAI message format.
- Find a frame where the domain mismatch (driving model on non-driving footage) causes UniDriveVLA to produce wrong or irrelevant output.
- Try replacing the backend with a Qwen2.5-VL-7B sidecar and compare output quality for aerial footage.
- Understand that VLA "actions" are advisory interpretations, not robot commands.

**Common failure modes:**
- `UNIDRIVE_API_URL` not set → step skipped; output uses Qwen analysis only.  Enable with `--unidrive-api-url`.
- Backend model is not vision-capable → HTTP 400 or empty response; adapter returns `service_unavailable`.
- Mission domain does not match backend training domain → off-domain output; switch to a general VLM backend.
- High-altitude aerial footage with driving-domain model → vehicles interpreted as abstract dots; quality drops.
- Backend JSON parse failure → adapter records `parse_error: true` and logs the raw response at DEBUG level.

---

<a id="step-26-base-model-search-test"></a>
## Step 26. Base model search test

**What it does:**
Run a set of fixed test queries against the baseline embedding store (Step 2, before any fine-tuning).
For each query, retrieve the top-K most similar frames and record the results.
Compare results to expected matches if ground truth is available, or record for human review.

**Why it matters:**
This is the diagnostic step before adaptation.
If the baseline retrieval is already excellent, fine-tuning (Steps 28-31) may not be needed.
If the baseline retrieval is weak, the search test identifies which queries fail and what kinds of scenes are confused.
The test result is the baseline that Step 31 (post-fine-tuning search test) will compare against.
Without this step, you cannot know whether fine-tuning improved anything.

**Implementation:**
- [`ssv_vdp/steps/embed.py`](../../packages/ss-fusion/src/ssv_vdp/steps/embed.py)

**Key concepts:**

*What makes a good search test:*
Use queries that span different failure modes:
- Easy positive: "red truck on highway" when one clearly exists.
- Hard positive: "truck viewed from above at night" — tests whether CLIP handles aerial perspective.
- Hard negative: "empty road" — should not retrieve frames with vehicles.
- Out-of-distribution: "submarine" — should retrieve nothing or distant neighbors.

*Precision at K (P@K):*
For each query, count the fraction of top-K results that are actually relevant.
P@5 = 3/5 means 3 of the top 5 retrieved frames are relevant.
This is the standard metric for retrieval evaluation.

*Recall at K (R@K):*
For each query, what fraction of all relevant frames appear in the top K results?
Useful when you care about finding everything, not just getting a few good results.

**Output artifact:**
Search test report: per-query top-K results with frame paths, similarity scores, and (if available) relevance labels.
Summary table: P@K and R@K for each query.

**Human focus:**
- Run three or four queries on your own mission data and inspect the top-5 results manually.
- Identify which query types fail and why (wrong CLIP vocabulary, aerial perspective confusion, domain mismatch).
- Record your findings: you will compare this against Step 31 results after fine-tuning.

**Common failure modes:**
- No relevant frames in the corpus for a query → P@K is undefined; the metric does not tell you the query was out-of-distribution.
- Duplicate or near-duplicate frames dominate results → retrieval looks good numerically but all results show the same moment.
- Query text uses domain vocabulary that CLIP was not trained on → retrieval is random.

---

<a id="step-27-3d-map-and-gaussian-splat"></a>
## Step 27. 3D map and Gaussian Splat

**What it does:**
Run SfM (Structure-from-Motion) using pycolmap on the dense frame set extracted at `SFM_FPS`.
Recover camera poses and a sparse 3D point cloud.
Optionally run nerfstudio splatfacto to produce a dense 3D Gaussian Splat (3DGS) from the SfM poses.
Store both the sparse structure and the dense rendering model.

**Why it matters:**
This is the shift from frame-wise evidence to persistent spatial structure.
A 3D map enables:
- GPS-registered spatial search: "what did the system see at this location at 47.12°N, 8.45°E?"
- Cross-mission change detection: compare the 3D structure from mission A to mission B at the same location.
- View synthesis: the Gaussian Splat can render novel viewpoints not in the original video.
- Robot pose advisory: `POST /query/pose` uses the map to answer "what should I expect to see here?"

**Implementation:**
- [`ssv_vdp/steps/map.py`](../../packages/ss-fusion/src/ssv_vdp/steps/map.py)
- [`pipeline/mapping/`](../../packages/ss-mapping/src/selfsuvis/pipeline/mapping)
- ``docs/gaussian_splat.md``

**Key concepts:**

*Structure-from-Motion (SfM):*
SfM recovers 3D structure and camera poses from a set of 2D images.
It works by finding matching feature points (SIFT, ORB, SuperPoint) across overlapping images, then solving for the 3D positions that are consistent with all observations.
pycolmap is the production-quality open-source implementation used here.
Requires: multiple overlapping views of each scene point (why `SFM_FPS=2` instead of 1).

*Sparse vs dense reconstruction:*
SfM output is sparse: typically 50,000-500,000 3D points for a 1000-frame sequence.
These points are at feature matches, not at every surface.
Dense reconstruction (MVS: Multi-View Stereo) fills in the surface; nerfstudio splatfacto does this implicitly with Gaussian primitives.

*3D Gaussian Splatting (3DGS):*
3DGS represents the scene as millions of 3D Gaussian ellipsoids, each with position, shape, opacity, and color.
Rendering is fast (real-time on GPU) because the Gaussians are splatted (projected) onto the camera plane.
The resulting model supports: view synthesis from arbitrary angles, real-time flythrough, and density queries.

*Pose status gate:*
The pipeline only runs splatfacto after SfM succeeds (`pose_status = success`).
If SfM fails (insufficient overlap, poor feature matches, featureless surfaces), the Gaussian Splat step is skipped.

*Short-clip matching policy:*
Short aerial clips often fail with purely sequential matching even when the frames are sharp.
The local mapper now switches pycolmap to exhaustive matching automatically for short clips, which
raises the chance of connecting the sequence into one component when there are only a few seconds of
usable views.

*Degraded-map recovery path:*
The pipeline no longer collapses immediately to a tiny PCA placeholder when SfM is sparse.
If only part of the sequence localizes, it interpolates the recovered trajectory and enriches the
map with semantic pseudo-3D anchors derived from detections, tracks, and coarse depth cues.
This produces a richer degraded map that is honest about its quality instead of pretending to be a
full reconstruction.

*Map quality advisor:*
Every completed local run now writes a companion advisor:

- `3d_map/map_quality_advisor.json`
- `3d_map/map_quality_advisor.md`

The advisor measures source-video properties that dominate reconstruction quality:

- clip duration
- resolution
- exposure consistency
- sharpness
- feature richness
- adjacent-frame matchability
- parallax proxy
- object scale / field size
- coarse camera-angle hint

That artifact exists to answer the question "was this video even suitable for a high-quality map?"
before you waste time tuning thresholds.

**Output artifact:**
`maps/{mission_id}/splat.ply` — the 3DGS scene model (large file: 100 MB-2 GB depending on duration).
Sparse point cloud: `sparse_reconstruction/` directory with COLMAP format files.
Per-frame `pose_json`: camera pose (rotation + translation) for each frame that was successfully localized.
Local-run diagnostics: `3d_map/map_stats.json`, `3d_map/map_quality_advisor.json`, and `3d_map/map_quality_advisor.md`.

**Human focus:**
- Understand why SfM needs overlap: a single frame cannot provide any 3D information alone.
- Learn the SfM failure modes: texture-less surfaces (white walls, water), pure rotation without translation, too-fast camera motion.
- Understand what a Gaussian Splat represents: not a mesh, not a point cloud, but a probabilistic volumetric representation.
- Know the scale ambiguity problem: SfM without GPS produces a map in arbitrary units; GPS registration resolves this.
- Learn to separate "mapper degraded" from "capture was poor" by reading `map_quality_advisor.md`.
- Inspect whether weak mapping comes from low parallax, too-short duration, too-high altitude, or low resolution rather than from SfM code.

**Common failure modes:**
- Too few overlapping views (low `SFM_FPS`) → SfM produces a degenerate map or fails entirely.
- Featureless terrain (uniform grass, water, sand) → insufficient feature matches; SfM fails.
- Pure rotation (drone pivoting in place without translation) → SfM has degenerate geometry; reconstruction is wrong.
- Long video with scene changes → SfM may split into disconnected sub-models.
- nerfstudio container not running → splatfacto step skipped; only sparse SfM output exists.
- Short clip with mostly forward drift → enough matches for partial localization, but too little baseline for dense reliable geometry.
- High-altitude nadir video with tiny vehicles and lane markings → map stays sparse even when ORB/SIFT matching looks healthy.

### High-quality capture requirements for aerial 3D mapping

The advisor and rebuilt drone run make one point very clear: high-quality 3D mapping is dominated
by capture geometry, not only by reconstruction software.

For very high quality mapping, optimize for these conditions:

- Duration: `25-40 s` over the same region of interest is a strong practical target.
- Resolution: at least `1280x720`, preferably `1920x1080` or higher.
- Parallax: include real lateral viewpoint change, not only forward motion or hover.
- Camera angle: use at least one pass around `25-40°` off nadir instead of only straight-down footage.
- Field scale: static infrastructure should be large enough that lane markings, curbs, poles, and roof edges occupy meaningful pixels.
- Overlap: maintain high forward overlap and a second cross-track pass when possible.
- Exposure: lock exposure and white balance if the camera allows it.
- Motion: keep translational speed low enough to avoid blur and to preserve correspondences.

### Drone learning-path case study

The current `drone_mission` run is a useful example because it is not a blur problem.
Measured signals were approximately:

- duration: `10.2 s`
- resolution: `854x480`
- brightness CV: `0.009`
- sharpness mean: `671`
- median ORB keypoints: `1200`
- match inlier ratio: `0.92`
- parallax proxy: `0.0049`
- median vehicle area: `0.0004` of frame

Interpretation:

- exposure and sharpness are already good enough
- matching is not the main bottleneck
- parallax is weak
- scene scale is too small
- the clip is short

So the bottleneck is capture geometry: the aircraft is effectively too high, too overhead, too
wide, and too brief for a very high quality map.

### Practical flight plan for very high quality area mapping

For a road, junction, or built-up outdoor area, use a repeatable three-pass plan:

1. Oblique orbit or arc
   Fly a slow arc around the area of interest at roughly `25-40°` off nadir to establish
   cross-view geometry and facade / roadside structure.
2. Along-track pass
   Fly one straight pass roughly aligned with the road or dominant axis of the scene with high
   forward overlap.
3. Cross-track pass
   Fly a second pass roughly perpendicular to the first to create the side overlap that nadir-only
   single-pass footage usually lacks.

Operational guidance:

- reduce altitude or narrow FOV until vehicles and lane markings are materially larger in frame
- keep speed low enough that road texture stays crisp
- avoid sudden yaw spins during mapping segments
- keep the scene mostly static when possible; heavy moving traffic reduces correspondence quality
- if available, record GPS / IMU and keep it time-aligned for later registration and smoothing

---

## End Of Phase: What You Should Understand

After Steps 21-27, a human should be able to answer:

- Which objects persist across frames and with what track IDs?
- What language context is fed into the densest reasoning step (Qwen) and where does each piece come from?
- Do the temporal clip models capture genuinely different information than frame embeddings?
- Can the scene be placed into a usable 3D spatial structure?
- What does the baseline retrieval performance look like before adaptation?

If you cannot answer those questions, especially the last two, do not proceed to adaptation.
Adaptation is only valuable if you know what the baseline is and where it fails.

## Related Docs

- [Sensors and fusion: Steps 9-20](04_sensor_steps_09_20.md)
- [Adaptation and audit: Steps 28-35](06_adaptation_eval_steps_28_35.md)
- [Agentic knowledge flow](07_agentic_knowledge_flow.md)
- `3D Gaussian Splat`

---

## Learning Resources — Tracking, Mapping, and Temporal Reasoning (Steps 21-27)

Resources are ordered basics → deep dive. The common thread across this phase is learning to reason about *structure* — spatial, temporal, and semantic — rather than per-frame snapshots.

---

### Step 21 — YOLO + SAM Detection and Segmentation

**Why it matters:** SAM is the pivot from bounding-box thinking to pixel-level instance understanding. The pipeline uses SAM masks to decouple *what* is in the scene from *where* it is at pixel resolution — the prerequisite for tracking that survives partial occlusion.

**Basics**
- Meta AI SAM2 project page and demo: [ai.meta.com/sam2](https://ai.meta.com/sam2). Interactive demo builds intuition for prompt types (point, box, mask) before reading the code.
- HuggingFace SAM2 docs: [huggingface.co/docs/transformers/model_doc/sam2](https://huggingface.co/docs/transformers/model_doc/sam2)

**Core papers**
- Kirillov et al., "Segment Anything" (Meta AI, 2023). The SAM-1 paper. Section 2 (task and model) explains the promptable segmentation formulation and the SA-1B dataset (1B masks). The ambiguity design (returning multiple masks for ambiguous prompts) is directly relevant to understanding SAM's behaviour on the box-prompt path. [arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
- Ravi et al., "SAM 2: Segment Anything in Images and Videos" (Meta AI, 2024). Adds streaming memory (object pointers stored across frames) for video segmentation. Section 3.2 (memory bank) explains the temporal propagation mechanism. [arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)

**Deep dive**
- He et al., "Mask R-CNN" (2017). Historical context: the first strong instance segmentation model. Reading it makes SAM's design simplifications (no class prediction, no instance limit) deliberate and principled rather than accidental. [arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)

---

### Step 22 — Gemma Directed Tracking and RF-DETR

**Why it matters:** Language-guided tracking is the architectural answer to the question "which objects matter for this mission?" A general-purpose tracker maintains all tracks with equal priority; Gemma's `tracking_priority` list encodes mission-specific semantics into the tracker.

**Basics**
- RF-DETR documentation and model zoo: [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr). Start with the inference quickstart before reading the tracking integration.

**Core paper**
- Cai et al., "RF-DETR: DETR-based Object Detector Fine-Tuned via Radio Frequency Sensing" — the model used for tracking in Step 22. RT-DETR backbone with Roboflow fine-tuning optimizations.
- Bewley et al., "Simple Online and Realtime Tracking" (SORT, 2016). The greedy IoU matching algorithm that underpins the track-ID assignment in `RFDETRTracker`. Two pages — read this to understand exactly what happens at IoU < 0.45 (a new track ID is assigned). [arxiv.org/abs/1602.00763](https://arxiv.org/abs/1602.00763)

**Deep dive**
- Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2022). The current state-of-the-art in multi-object tracking — explains detection confidence thresholding and low-confidence track recovery. Directly relevant to understanding SORT's failure modes at speed. [arxiv.org/abs/2110.06864](https://arxiv.org/abs/2110.06864)
- Cao et al., "OC-SORT: Observation-Centric SORT on Video Wild" (2022). Addresses the SORT limitation of track fragmentation during occlusion — the exact scenario where `RFDETRTracker`'s IoU < 0.45 produces false new track IDs. [arxiv.org/abs/2203.14360](https://arxiv.org/abs/2203.14360)

---

### Step 23 — World Models and RSSM Temporal Surprise

**Why it matters:** Temporal surprise from the RSSM carries 40% of the active learning signal. A frame that is visually unremarkable (low DINO distance) but temporally surprising (the RSSM did not predict it) represents a genuine scene transition the static metrics miss — exactly the frames most valuable for annotation.

**Basics — World Models**
- Schmidhuber, "A Possibility for Implementing Curiosity and Boredom in Model-Building Neural Controllers" (1991). The original two-page paper proposing prediction error as an intrinsic curiosity signal — the conceptual ancestor of RSSM surprise. Understanding this paper makes the pipeline's `rssm_surprise` intuitive rather than mechanical.
- Ha & Schmidhuber, "World Models" (2018). Blog-post-length paper with visual explanations of an MDN-RNN world model and a learned controller. The clearest non-mathematical introduction to the world model paradigm. [arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

**Core papers — RSSM lineage**
- Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2019). The paper that introduced the RSSM architecture: encoder, GRU recurrent model, dynamics predictor, decoder. Section 3 (RSSM) maps directly to `models/rssm_model.py`. [arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023). The current pinnacle of the Dreamer line: symlog predictions, KL balancing, free bits. Section 2 (world model) is the architectural reference for the RSSM hyperparameters (`DREAMER_HIDDEN_DIM`, `DREAMER_LATENT_DIM`). [arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- Romero et al., "Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight" (ICRA 2026). The direct inspiration for the temporal surprise signal in this pipeline. Section III (RSSM for drone perception) and Section IV (experiments) show that RSSM surprise identifies genuinely novel flight situations. [rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf](https://rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf)

**Basics — Heavy world models (WORLD_MODEL_ENABLED)**
- HuggingFace VideoMAE docs: [huggingface.co/docs/transformers/model_doc/videomae](https://huggingface.co/docs/transformers/model_doc/videomae)
- Wang et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training" (2022). The pre-training method for the `MCG-NJU/videomae` checkpoints. Section 3 (tube masking) explains why 90% masking ratio works for video but not images. [arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602)

**Deep dive**
- NVIDIA Cosmos technical blog (2024). Describes the 4B physical world model underlying `nvidia/Cosmos-1.0-Autoregressive-4B`. The autoregressive token prediction approach differs fundamentally from the RSSM continuous latent approach — understanding both clarifies when each is appropriate.

---

### Step 24 — Qwen Detailed Captioning

**Why it matters:** Qwen receives the richest context of any model in the pipeline — the full `context_for_frame()` string with prior description, segment, audio, visible text, depth, objects, and prior frame state. Its output is the densest single-frame reasoning artifact and the primary source for the synthesis step.

**Basics**
- HuggingFace Qwen2.5-VL model page: [huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). Model card covers context length, vision token encoding, and recommended inference parameters.
- Qwen2.5-VL system prompt and JSON response guidelines — the README on the model page gives concrete examples of structured output prompting.

**Core paper**
- Bai et al., "Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities" (2023). Explains the visual token compression (4:1 reduction via cross-attention), bounding-box grounding, and multi-task instruction tuning. [arxiv.org/abs/2308.12966](https://arxiv.org/abs/2308.12966)

**Deep dive**
- Team Qwen, "Qwen2.5-VL Technical Report" (2025). Updated architecture: dynamic resolution, Temporal Vision Transformer (TVT), and video-specific improvements. Section 3 (model architecture) and Section 5 (evaluation) are the key sections. [arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)

---

### Step 25 — UniDriveVLA Expert Analysis

**Why it matters:** UniDriveVLA is a Vision-Language-Action model trained specifically on driving scenarios. Its `planning` output (recommended action, trajectory hint, hazards) is qualitatively different from VLM descriptions — it reasons about what to *do*, not just what is *there*.

**Basics**
- See the runbook: [docs/runbooks/unidrive-api.md](../runbooks/unidrive-api.md) — backend selection, setup, and expected output schema.
- Chen et al., "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models" (2024). Explains the design space for VLAs on driving data and why domain-specific fine-tuning on NuScenes-style annotations outperforms general VLMs. [arxiv.org/abs/2402.12289](https://arxiv.org/abs/2402.12289)

**Core paper**
- Sima et al., "DriveLLM: Charting the Path Toward Full Autonomous Driving with Large Language Models" (2023). Maps the four-component output (understanding, perception, planning, metacognition) that UniDriveVLA's schema normalizes. [arxiv.org/abs/2312.09245](https://arxiv.org/abs/2312.09245)

**Deep dive**
- Shao et al., "SparseOccupancy and Planning" — occupancy-based world representation that underlies advanced VLA planning outputs.
- Waymo, "Scaling Self-Supervised End-to-End Driving with Robust Reward Labels" (2024). Shows how large-scale pretraining on real driving data produces emergent planning capabilities — relevant for non-road missions where UniDriveVLA is replaced with Qwen. [arxiv.org/abs/2405.10314](https://arxiv.org/abs/2405.10314)

---

### Steps 26 and 27 — 3D Mapping: SfM and Gaussian Splatting

**Why it matters:** Step 27 produces the spatial scaffold that turns this pipeline from a video analysis tool into a persistent spatial memory system. A mission with valid 3DGS output can be visualized, re-queried from arbitrary viewpoints, and compared geometrically to future missions.

**Basics — Structure-from-Motion**
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (2nd ed., Cambridge, 2004). Chapters 7 (fundamental matrix), 9 (essential matrix), and 18 (affine and projective reconstruction). The mathematical foundation for everything pycolmap does. Available through most university library systems.
- Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., 2022). Chapter 7 (feature detection, description, matching) and Chapter 11 (structure from motion). Accessible alternative to Hartley & Zisserman.

**Core paper — SfM**
- Schönberger & Frahm, "Structure-from-Motion Revisited" (CVPR, 2016). The COLMAP paper — the algorithm underlying pycolmap. Section 3 (incremental SfM) explains the initialization heuristic, the bundle adjustment schedule, and the outlier filtering that determines whether a reconstruction succeeds or degenerates. This is the required reading before debugging `pose_status = failed`.

**Basics — Neural Radiance Fields and Gaussian Splatting**
- Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020). The NeRF paper — the predecessor to 3DGS. Understanding the implicit radiance field representation (and why it is slow) makes the 3DGS explicit-primitive approach clearly motivated. [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)

**Core paper — 3DGS**
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH, 2023). The 3DGS paper. Section 4 (3D Gaussian representation), Section 5 (adaptive density control), and Section 6 (fast differentiable rasterizer) are the three technical contributions. [arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079)

**Deep dive**
- Luiten et al., "Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis" (2023). Extends 3DGS to dynamic scenes — relevant for missions with moving objects. Shows how track IDs from Step 22 could in principle be used to initialize dynamic Gaussian groups. [arxiv.org/abs/2308.09713](https://arxiv.org/abs/2308.09713)
- nerfstudio documentation: [docs.nerf.studio](https://docs.nerf.studio). The training and export pipeline wrapping splatfacto. Pay attention to `ns-train splatfacto --help` for the parameters that control splat quality vs. training time.
