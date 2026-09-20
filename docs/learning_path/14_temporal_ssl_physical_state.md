# Temporal SSL And Track-Aware Representation Learning

This document is a study companion to
[`06_adaptation_eval_steps_28_35.md`](06_adaptation_eval_steps_28_35.md)
and the code in
[`pipeline/training/ssl.py`](../../packages/ss-fusion/src/selfsuvis/pipeline/training/ssl.py) and
[`ssv_vdp/steps/ssl.py`](../../packages/ss-fusion/src/ssv_vdp/steps/ssl.py).

It explains one specific evolution of the SSL fine-tuning step:
the shift from **frame augmentation** to **temporal track augmentation** as the source
of positive pairs, and how **cycle-consistency loss** adds long-horizon identity stability.

---

## 1. The Problem With Frame-Only Pairs

The current DINOv3 fine-tuning (Step 16) runs contrastive learning with two pair types:

| Approach | Positive pair construction |
|---|---|
| `augment` | Two random augmentations of the *same frame* |
| `temporal` | Consecutive frames within ±`max_gap` positions in the sequence |

Both are valid SSL signals.  Both have a shared limitation:

**They say nothing about object identity across time.**

Consider a mission video with a vehicle entering from the left at t=0 and exiting right at t=30.
A temporal pair (frame[5], frame[8]) shows similar backgrounds, similar road texture, similar scene lighting.
The model learns "nearby frames look similar."
It does not learn "the vehicle at t=5 is the same entity as the vehicle at t=8."

This distinction matters for:
- **Retrieval:** queries like "vehicle with damaged rear" require identity-consistent embeddings, not scene-consistency embeddings.
- **Tracking robustness:** embeddings that drift across object appearances cause track breaks and false re-ID.
- **Anomaly detection:** recognising that an object's appearance has genuinely changed (vs the camera moved) requires a stable per-object baseline.

---

## 2. Track IDs As A Free Supervision Signal

RF-DETR (Step 22) produces **persistent track IDs** via greedy IoU assignment.
Each detection gets a `track_id` that stays constant across frames as long as the
object is continuously visible.  The output lives in `gemma_tracking_results.json`:

```json
{
  "frames": [
    {
      "frame_path": "/data/frames/vid/frame_0001.jpg",
      "t_sec": 0.033,
      "detections": [
        {
          "track_id": 3,
          "bbox_norm": [0.42, 0.18, 0.61, 0.55],
          "label": "car",
          "confidence": 0.91
        }
      ]
    }
  ]
}
```

Track IDs are free — no human labels, no extra model.
They provide a per-object identity signal across the entire video.

The insight behind track-aware SSL:

> Two bbox crops of the same `track_id` at different times are a *strong* positive pair.
> The model must learn features that remain stable as the object moves, rotates,
> changes scale, and is partially occluded.  That is a harder and more useful invariance
> than "nearby frames look similar."

---

## 3. Three Pairing Approaches — Priority Order

The SSL step now selects among four approaches in priority order:

```
track_cycle  →  track  →  temporal  →  augment
```

### `track_cycle` (highest priority)

Requires: at least `batch_size` triplets (A, B, C) from the same track,
where A = appearances[i], B = appearances[i+k], C = appearances[i+2k], k ∈ [2, 5].

**Dataset:** `TrackTripletDataset` — crops each appearance around `bbox_norm` with 15 % padding.

**Loss:** `CycleConsistencyLoss`

```
L = NTXent(z_A, z_B) + NTXent(z_B, z_C) + 0.3 · NTXent(z_A, z_C)
```

The first two terms enforce adjacent-pair consistency.
The third term (λ=0.3) is the cycle term: it enforces that the object's embedding
at time t and time t+2k are consistent through the intermediate frame at t+k.

**Why the cycle term matters:**
Without it, the model can satisfy NTXent(A,B) and NTXent(B,C) by treating B as a
"bridge" without requiring A and C to be similar.  Over a long track, embeddings
can drift — the car at frame 1 and the car at frame 60 are far apart in embedding
space even though they're the same vehicle.  The cycle term prevents this.

λ=0.3 because the long-horizon pair is genuinely harder: the object has moved more,
may have rotated, may be partially occluded.  A lower λ avoids over-constraining
the representation on uncertain pairs.

### `track` (second priority)

Requires: at least `batch_size` pairs from the same track.

**Dataset:** `TrackPairDataset` — same bbox-crop logic, returns (A, B) pairs.

**Loss:** `NTXentLoss` — identical to the existing temporal/augment approaches.

The advantage over temporal pairs: the positive pair is guaranteed to be the same object,
not just a nearby frame (which may contain a completely different object that replaced the first).

### `temporal` (third priority)

Existing behaviour — consecutive frames, no track information.

### `augment` (fallback)

Existing behaviour — two augmentations of the same frame.

---

## 4. How RSSM Surprise Improves Track-Pair Quality

The RSSM temporal surprise score (DreamerV3-inspired, computed at Step 14) measures
how surprising each frame is relative to the learned world-model prediction.

High RSSM surprise = the object's appearance or motion deviated from expectation.

For SSL training, RSSM surprise interacts with track-aware pairs in two ways:

**1. RSSM selects harder positive pairs implicitly.**
Frames flagged `needs_annotation` by the active-learning scorer (which combines RSSM
surprise + DINO distance + caption confidence) tend to contain objects in unusual states:
turning, occluded, illumination-changing, or partially out of frame.
If these frames appear in a track, the positive pair drawn from them forces the model
to learn more general appearance invariance than pairs from static, repetitive segments.

**2. RSSM surprise can be used to weight pair sampling (future enhancement).**
Currently all appearances within a track are sampled uniformly.
A natural extension is to up-sample pairs where at least one member has high RSSM
surprise, biasing the training distribution toward harder cases.

This creates a virtuous cycle identical to the one described in
[`06_adaptation_eval_steps_28_35.md`](06_adaptation_eval_steps_28_35.md):

```
RSSM → better AL frame selection
     → harder track pairs in SSL
     → better object-identity-aware backbone
     → better DINO distances in next mission's AL scoring
     → cycle continues
```

---

## 5. Implementation Pointers

| Component | File | Class / function |
|---|---|---|
| Bbox crop helper | `pipeline/training/ssl.py` | `_crop_bbox()` |
| Pair dataset | `pipeline/training/ssl.py` | `TrackPairDataset` |
| Triplet dataset | `pipeline/training/ssl.py` | `TrackTripletDataset` |
| Cycle loss | `pipeline/training/ssl.py` | `CycleConsistencyLoss` |
| Track map extraction | `ssv_vdp/steps/ssl.py` | `_extract_track_map()` |
| Pair/triplet counting | `ssv_vdp/steps/ssl.py` | `_count_potential_pairs()`, `_count_potential_triplets()` |
| SSL fine-tuning step | `ssv_vdp/steps/ssl.py` | `step_ssl_finetune()` |
| Finetune stats report | `ssv_vdp/steps/report.py` | `write_finetune_stats_md()` |

The pairing approach chosen for each run is reported in `finetune_stats.md`
under the **"Pair Construction Strategy"** heading.

---

## 6. What A Human Should Study Here

### The core question

Why is "same object, two times" a better positive pair than "nearby frames"?

The answer is that nearby frames can contain *different objects*.
A scene with many moving actors changes rapidly.  Track IDs provide
*object-level temporal identity* — a guarantee that the two crops show the same entity.
That identity is exactly the invariance that retrieval and re-ID depend on.

### The cycle-consistency idea

Cycle consistency comes from video self-supervised learning research.
The original formulation (Wang and Gupta, 2015; Dwibedi et al., 2019) used
correspondence matching: "does this patch tracked forward to frame B and then
backward again arrive at the same place in frame A?"

The contrastive variant used here is simpler:
instead of explicit correspondence, the cycle loss is an NT-Xent term on (z_A, z_C).
The model is forced to make the same-object embedding stable across the full temporal
span of the triplet, not just adjacent frames.

This is closer to the approach in **TimeSformer** and **VideoMAE** which learn
temporal consistency across non-adjacent frames via masked prediction or cross-frame
attention.

### Reading list

- Wang and Gupta, "Unsupervised Learning of Visual Representations using Videos" (ICCV 2015)
  — the original cycle-consistency idea in video learning.
  [https://arxiv.org/abs/1505.00056](https://arxiv.org/abs/1505.00056)

- Dwibedi et al., "Temporal Cycle-Consistency Learning" (CVPR 2019)
  — the canonical temporal cycle formulation.
  [https://arxiv.org/abs/1904.07846](https://arxiv.org/abs/1904.07846)

- Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training" (NeurIPS 2022)
  — masked video modeling as an alternative temporal SSL paradigm.
  [https://arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602)

- Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?" (TimeSformer, 2021)
  — divided space-time attention; useful context for how temporal dependencies are captured.
  [https://arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095)

- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO, 2021)
  — the backbone used here; understanding the student-teacher EMA architecture is required
  before modifying any SSL hyperparameter.
  [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)

---

## 7. Common Failure Modes For Track-Aware SSL

| Symptom | Likely cause | What to check |
|---|---|---|
| Approach falls back to `temporal` or `augment` | RF-DETR disabled or produced zero detections | `RFDETR_ENABLED` env var; `gemma_tracking_results.json` → `n_unique_track_ids` |
| Track pairs exist but approach stays `temporal` | Not enough pairs to fill one batch | Check `n_tracks × mean_track_length` vs `batch_size`; reduce `batch_size` or lower `min_gap` |
| Loss is higher with `track_cycle` than `temporal` | Tracks are too short — triplets span big appearance changes | Inspect `mean_track_length_frames` in the tracking summary; normal if tracks are short |
| Cycle term dominates (loss oscillates) | λ too high or all triplets have large temporal gaps | λ=0.3 is conservative; if oscillation persists, reduce to 0.1 |
| SSL gate triggers (`best_loss > 10.0`) after switch to track-cycle | Tracks are noisy (ID switches, merged detections) | Read `gemma_tracking_summary.md`; if `median_track_length_frames < 3`, the track quality is too low for triplets |

---

## 8. Physical Scene Layer — Occupancy and Free Space

The **physical state summary** (`physical_state_summary.json`) is produced by
`ssv_vdp/steps/physical_state.py` immediately after full-state fusion.
It aggregates the outputs of three upstream steps into a single compact belief dictionary
that SSL fine-tuning, report generation, and threat-primitive extraction can consume without
re-running any models.

### 8.1 What Is Near-Field Occupancy?

A classical occupancy grid divides the space around the platform into cells and marks each cell
as occupied or free based on sensor returns.  In aerial or vehicle imagery without a ranging
sensor, the equivalent proxy is **bbox coverage of the central image region**.

The pipeline uses a 2D analogue:

- The **central region** is `cx, cy ∈ [0.3, 0.7]` in normalised image coordinates
  (the middle 40 % × 40 % of the frame — roughly where near-field objects appear in
  a forward-facing camera with typical optics).
- For each confirmed or smoothed track whose bounding-box centre falls in this region,
  its box area `(x2 − x1) × (y2 − y1)` is summed into `frame_area`.
- `frame_area` is clamped to 1.0 and averaged across all frames to produce
  `near_field_occupancy_density ∈ [0, 1]`.

A value near 0 means the central field of view is clear.
A value near 1 means tracked objects nearly fill the near-field region (congested scene).

The threshold choice of `[0.3, 0.7]` is a heuristic calibrated for missions where the
platform is a ground vehicle or low-altitude drone:
at typical operating altitudes the threat zone for collision is roughly the central quarter
of the image.

### 8.2 Free Space Estimate

`free_space_estimate` combines two independent occupancy signals:

```
effective_occupancy = max(near_field_density, depth_near_ratio × 0.4)
free_space_estimate = max(0.0, 1.0 − effective_occupancy)
```

`depth_near_ratio` is the mean fraction of "near" depth pixels across all depth frames
(from `step_depth_estimation`).  It is discounted by 0.4 because near depth pixels
include ground plane and wall geometry that does not threaten the platform path — only a
fraction of "near" depth pixels correspond to actual obstacles.

Using the *maximum* of the two signals is a pessimistic (conservative) lower bound:
if either sensor says the space is occupied, the estimate treats it as occupied.

### 8.3 Platform Pose Confidence

The Kalman covariance trace (`cov_trace`) from the platform filter measures positional
uncertainty in ENU metres².  The pipeline maps it to a [0, 1] confidence score:

```
pose_confidence = 1.0 / (1.0 + mean_cov_trace / 10.0)
```

Calibration anchors (from `fusion/summaries._sample_quality`):

| cov_trace | quality label | confidence |
|-----------|---------------|------------|
| ≤ 10      | good          | ≥ 0.50     |
| ≤ 40      | degraded      | ≥ 0.20     |
| > 40      | uncertain     | < 0.20     |

When `platform_status ≠ "ok"` (no GPS fix, IMU-only, or fusion skipped), confidence
returns 0.0 — the pipeline treats pose as completely unknown.

---

## 9. Kalman Filter Basics — What `pipeline/fusion/filters/` Implements

All four fusion layers in the probabilistic state fusion subsystem use a
**constant-velocity Kalman filter** as their base dynamical model.

### 9.1 State Vector and Transition

For the object tracker (`object_filter.py`), the state vector is:

```
x = [cx, cy, w, h, vcx, vcy]ᵀ
```

where `cx, cy` is the normalised bbox centre, `w, h` are normalised width and height,
and `vcx, vcy` are the corresponding velocities (normalised coords per frame).

The state transition is:

```
x̂_{k|k−1} = F · x_{k−1|k−1}
P_{k|k−1}  = F · P_{k−1|k−1} · Fᵀ + Q
```

where `F` is a constant-velocity matrix:

```
F = [[1, 0, 0, 0, dt, 0 ],
     [0, 1, 0, 0, 0,  dt],
     [0, 0, 1, 0, 0,  0 ],
     [0, 0, 0, 1, 0,  0 ],
     [0, 0, 0, 0, 1,  0 ],
     [0, 0, 0, 0, 0,  1 ]]
```

`Q` is the process noise covariance; `P` is the estimation error covariance.

### 9.2 Update Step

When a detection is available, the filter updates using the observation `z = [cx, cy, w, h]`:

```
y   = z − H · x̂_{k|k−1}           (innovation)
S   = H · P_{k|k−1} · Hᵀ + R       (innovation covariance)
K   = P_{k|k−1} · Hᵀ · S⁻¹         (Kalman gain)
x̂_k = x̂_{k|k−1} + K · y
P_k  = (I − K · H) · P_{k|k−1}
```

`R` is the observation noise covariance (diagonal, `_OBS_NOISE = 0.005`).

The `bbox_std` field in `ObjectStateSample` is `sqrt(diag(P))` for the first four
state dimensions — the estimated positional uncertainty in normalised bbox coordinates.
`mean_bbox_uncertainty` in the physical state summary averages these across all
active tracks and frames.

### 9.3 Mahalanobis Gating

Before the Hungarian assignment solver assigns a detection to a track, the filter
checks whether the detection falls within the track's statistical confidence ellipse:

```
d² = (z − H·x̂)ᵀ · S⁻¹ · (z − H·x̂)
```

If `d² > χ²(4 DOF, p=0.99) ≈ 13.28`, the detection is gated out (cost set to `1e9`).
This prevents fast-moving or occluded objects from stealing nearby detections.

### 9.4 RTS Backward Smoother

After the forward pass, confirmed tracks are smoothed with a **Rauch-Tung-Striebel (RTS)**
backward pass (`rts_smoother.py`).  The RTS smoother propagates future observations
back in time to reduce positional uncertainty at earlier frames:

```
# Backward pass (from last frame to first):
G_k  = P_{k|k} · Fᵀ · P_{k+1|k}⁻¹          (smoother gain)
x̂^s_k = x̂_{k|k} + G_k · (x̂^s_{k+1} − F · x̂_{k|k})
P^s_k  = P_{k|k} + G_k · (P^s_{k+1} − P_{k+1|k}) · G_kᵀ
```

After smoothing, `ObjectStateSample.track_state` is set to `"smoothed"`.
Smoothed positions are strictly more accurate than filtered positions for
offline analysis (all future observations have been used).

### 9.5 How the Physical State Summary Uses These Primitives

| Summary field | Kalman primitive used |
|---|---|
| `platform_pose_confidence` | `cov_trace = tr(P)` from the platform filter RTS smoother |
| `near_field_occupancy_density` | smoothed `bbox_norm` from object filter |
| `tracked_object_velocities` | smoothed `velocity_norm = [vcx, vcy]` from object filter |
| `mean_bbox_uncertainty` | `mean(sqrt(diag(P))[:4])` averaged across all tracks and frames |
| `free_space_estimate` | derived from `near_field_density` + depth `near_ratio` |
| `confirmed_tracks` | count of tracks that reached the `confirmed` lifecycle state |

### 9.6 Where To Read The Code

| Concept | File | Symbol |
|---|---|---|
| Object KF predict/update | `pipeline/fusion/filters/object_filter.py` | `ObjectKalmanFilter` |
| Mahalanobis gating | `pipeline/fusion/filters/object_filter.py` | `.is_gated()`, `.mahalanobis_distance_sq()` |
| RTS smoother | `pipeline/fusion/filters/rts_smoother.py` | `rts_smooth()` |
| Full object fusion pass | `pipeline/fusion/object_state.py` | `run_object_state_fusion()` |
| Near-field density calc | `pipeline/fusion/object_state.py` | `summarize_object_frame_dicts()` |
| Physical state step | `ssv_vdp/steps/physical_state.py` | `step_physical_state()` |
| VideoKnowledge deposit | `ssv_vdp/steps/common.py` | `VideoKnowledge.add_physical_state()` |

---

## Related Docs

- [Adaptation and eval steps 28-35](06_adaptation_eval_steps_28_35.md) — full SSL context
- [Tracking and mapping steps 21-27](05_tracking_mapping_steps_21_27.md) — how RF-DETR produces the track IDs used here
- [Future directions](18_future_directions.md) — broader remaining SSL roadmap and next-stage expansion priorities
- [Probabilistic fusion deep dive](12_probabilistic_fusion_deep_dive.md) — full Kalman/RTS/Mahalanobis math with worked example

---

[← Adaptation and eval](06_adaptation_eval_steps_28_35.md) | [Future directions →](18_future_directions.md)
