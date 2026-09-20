# Probabilistic State Fusion — Mathematical Deep Dive

This document explains the full implementation from mathematical first
principles to working code. It is the reference for anyone who needs to
understand, modify, or extend the fusion subsystem.

Read after:

- [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md)
- [10_probabilistic_state_fusion_architecture.md](10_probabilistic_state_fusion_architecture.md)

The code lives in `src/selfsuvis/pipeline/fusion/`.

---

## 1. The Kalman Filter — What It Computes And Why

### The Problem

You want to know where a drone is at each video frame timestamp. You have:

- GPS readings, roughly once per frame, accurate to ±5 m, possibly missing
- Barometer altitude, accurate to ±2.5 m, from a sidecar file
- IMU accelerometer, faster than frames, but biased and noisy
- SfM camera poses, accurate but in an unknown local frame at unknown scale

None of these is correct on its own. The Kalman filter is the optimal
algorithm for combining them when:

- the motion model is linear (constant velocity)
- the noise is Gaussian
- you trust your noise covariance values

### State and Notation

The platform state is a 6-vector:

```
x = [px, py, pz, vx, vy, vz]
```

where `p` is position in ENU metres and `v` is velocity in m/s.

The *posterior distribution* at time k is Gaussian:

```
p(x_k | z_1:k) = N(x̂_k, P_k)
```

`x̂_k` is the mean (best estimate) and `P_k` is the 6×6 covariance matrix
(uncertainty).

### Predict Step

Between measurements, the filter propagates belief forward using the
constant-velocity model:

```
x̂_{k+1|k} = F · x̂_k + B · a_k
P_{k+1|k} = F · P_k · F^T + Q
```

Where:

```
F = I_6 with F[:3, 3:] = I_3 · dt     (position integrates velocity)

B[:3, :] = 0.5 · I_3 · dt²            (acceleration → position)
B[3:, :] = I_3 · dt                   (acceleration → velocity)

Q = diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
q_pos = (σ_pos · dt)²                 (grows with elapsed time)
q_vel = (σ_vel · dt)²
```

The `dt` is the interval since the last event. `σ_pos` and `σ_vel` are
configured in `STATE_FUSION_PROCESS_POS_STD_M` and `STATE_FUSION_PROCESS_VEL_STD_MPS`.

When a semantic prior is active, `σ_pos` and `σ_vel` are multiplied by
`prior.process_noise_scale` before `Q` is built. This means a busy
intersection makes the filter less confident between measurements.

**Code:** `PlatformStateFilter.predict()` in `fusion/filters/platform.py`.

### Update Step

When a measurement arrives:

```
y = z - H · x̂_{k|k-1}              (innovation = actual − predicted)
S = H · P_{k|k-1} · H^T + R         (innovation covariance)
K = P_{k|k-1} · H^T · S^{-1}        (Kalman gain)
x̂_{k|k} = x̂_{k|k-1} + K · y
P_{k|k} = (I - K · H) · P_{k|k-1}
```

For GPS position, `H = [I_3 | 0_3]` and `z = [px, py, pz]` from GPS.
For barometer, `H = [0 0 1 0 0 0]` and `z = [pz]`.

`R` is the measurement noise covariance, diagonal:

- GPS: `R = diag([σ_gps², σ_gps², σ_gps²])`
- Barometer: `R = [σ_baro²]`
- SfM position: `R = diag([σ_sfm², σ_sfm², σ_sfm²])`

`σ_gps` is scaled by `prior.gps_noise_scale` at runtime.

**Code:** `PlatformStateFilter._update()` in `fusion/filters/platform.py`.

### Why `pinv` Instead of Direct Inversion

The code uses `np.linalg.pinv(S)` for the Kalman gain. This is slower but
numerically safer. Direct inversion of `S` can fail or produce garbage when
`R` and `P` are near-degenerate. The pseudoinverse degrades gracefully to
zero gain, which is safe — the filter just ignores the bad measurement.

---

## 2. Visual-Pose Constraints — Umeyama Sim(3) Alignment

### The Problem

SfM (pycolmap) recovers camera poses in a local frame. There are three
unknowns between the SfM frame and GPS-ENU:

1. **Rotation** — arbitrary orientation
2. **Translation** — arbitrary origin
3. **Scale** — SfM is up to scale (no metric ground truth)

These three degrees of freedom define a Sim(3) transformation:

```
p_ENU ≈ s · R · p_SfM + t
```

where `s` is a positive scalar, `R` is a 3×3 rotation matrix, and `t` is a
3-vector translation.

Given N co-located pairs `{(p_SfM_i, p_ENU_i)}` where both SfM and GPS are
available, we want the optimal `(s*, R*, t*)`.

### The Umeyama Algorithm

Reference: Umeyama (1991), *Least-squares estimation of transformation
parameters between two point patterns*, IEEE T-PAMI.

**Step 1.** Centre both point sets:

```
μ_src = (1/N) Σ p_SfM_i
μ_dst = (1/N) Σ p_ENU_i
σ²_src = (1/N) Σ ‖p_SfM_i - μ_src‖²
```

**Step 2.** Compute the cross-covariance:

```
Σ_cross = (1/N) (p_ENU_c)^T · p_SfM_c
```

where `p_c` denotes centred points. `Σ_cross` is 3×3.

**Step 3.** SVD:

```
U, D, Vt = svd(Σ_cross)
```

**Step 4.** Handle reflections (prevent `R` from becoming an improper rotation):

```
det_sign = sign(det(U @ Vt))
S_diag = [1, 1, det_sign]
R = U · diag(S_diag) · Vt
```

**Step 5.** Recover scale and translation:

```
s = (1/σ²_src) · trace(diag(D) · diag(S_diag))
t = μ_dst - s · R · μ_src
```

**Step 6.** Transform all SfM positions to ENU and record residual:

```
p̂_ENU_i = s · R · p_SfM_i + t
RMSE = sqrt((1/N) Σ ‖p_ENU_i - p̂_ENU_i‖²)
```

The resulting measurement noise: `σ_sfm = max(0.5, STATE_FUSION_SFM_POS_STD_M + RMSE)`.

If RMSE is small (SfM and GPS agree well), the SfM measurements are trusted
nearly as much as GPS. If RMSE is large, they are weighted lower.

**Code:** `_umeyama_sim3()` in `fusion/visual_pose.py`.

### Why This Matters In Practice

Between GPS updates (say, every 0.5 s) the filter relies only on the
process model, and position uncertainty grows by `q_pos = (σ_pos · dt)²`.
SfM provides an additional position measurement at every frame where a pose
was recovered, which can be every 0.5 s but at a different time offset than
GPS. This roughly halves the worst-case covariance growth between GPS pings.

---

## 3. RTS Backward Smoother

### The Problem With The Forward Filter

The Kalman filter is *causal*: at time `k`, it uses only data from `1..k`.
This means early frames are uncertain even though we have all the data from
the full mission. The filter is the correct algorithm for real-time use, but
for offline analysis on recorded video we can do better.

### The RTS Smoother

The Rauch-Tung-Striebel smoother runs a backward pass over the filtered states.

**Forward pass** (already computed): for each frame `k`, we have `(x̂_{k|k}, P_{k|k})`.

**Backward pass** (new): starting from the last frame `N`:

```
for k = N-2 down to 0:
    # predicted state and covariance from k to k+1
    F_k = build_cv_F(dt = t[k+1] - t[k])
    Q_k = build_cv_Q(dt)
    x̂_{k+1|k} = F_k · x̂_{k|k}
    P_{k+1|k} = F_k · P_{k|k} · F_k^T + Q_k

    # smoother gain
    G_k = P_{k|k} · F_k^T · P_{k+1|k}^{-1}

    # smoothed estimates
    x̂_{k|N} = x̂_{k|k} + G_k · (x̂_{k+1|N} − x̂_{k+1|k})
    P_{k|N} = P_{k|k} + G_k · (P_{k+1|N} − P_{k+1|k}) · G_k^T
```

The key intuition is `G_k`: it is the "correlation" between the filtered state
at `k` and the filtered state at `k+1`. When G is large, information from
later frames has a big correction to apply to the state at `k`.

**What the smoother achieves:**

- `P_{k|N} ≤ P_{k|k}` element-wise — the smoothed covariance is always ≤ filtered
- Early frames benefit from constraints imposed by later GPS/SfM measurements
- The smoothed trajectory is the minimum-variance linear-Gaussian estimate
  given all measurements

**Approximation note:** Between consecutive frame samples, the forward filter
may have processed intermediate measurements. The backward pass uses only the
CV prediction `F·P·F^T + Q`, not the intermediate filter history. This is an
approximation: intermediate measurements made the forward covariance smaller
than a pure CV prediction would predict. So `P_{k+1|k}` is slightly
overestimated, making `G_k` slightly conservative. In practice this is a tiny
effect and makes the smoother stable.

**Code:** `rts_smooth()` in `fusion/filters/rts_smoother.py`.

---

## 4. Probabilistic Object Tracking

### Per-Object Kalman Filter

Each RF-DETR track gets an independent Kalman filter with state:

```
x = [cx, cy, w, h, vcx, vcy]
```

All dimensions are in normalised image coordinates (0–1).

Transition: position integrates velocity over `dt`.

```
F_obj = I_6 with F_obj[0, 4] = dt, F_obj[1, 5] = dt
```

Observation: `H = [I_4 | 0_{4×2}]` — we observe `[cx, cy, w, h]`,
not velocity.

The Joseph form is used for the covariance update for numerical stability:

```
P_new = (I - K·H) · P · (I - K·H)^T + K · R · K^T
```

This is algebraically equivalent to the standard form but keeps `P`
positive semi-definite even under floating-point errors.

**Code:** `ObjectKalmanFilter` in `fusion/filters/object_filter.py`.

### Mahalanobis Distance Gating

For each (track, detection) pair, the gate test is:

```
d² = (z - H·x̂)^T · (H·P·H^T + R)^{-1} · (z - H·x̂)
```

If `d² > 13.28` (chi-squared threshold, 4 DOF, p = 0.99), the pair is
**infeasible**: the detection is too far from the predicted track position
given the track's uncertainty.

The threshold 13.28 means: if the track and detection are truly the same
object, there is a 99% chance that `d² < 13.28`. Setting the threshold to
this value keeps the false-rejection rate at 1%.

This is strictly better than a fixed IoU threshold because:

- It accounts for *how uncertain* the track prediction is. A track that hasn't
  been updated for several frames has large `P`, so the gate is larger — it
  correctly accepts detections further away.
- It uses the full covariance structure, not just the bbox overlap.

**Code:** `ObjectKalmanFilter.mahalanobis_distance_sq()` and
`ObjectKalmanFilter.is_gated()` in `fusion/filters/object_filter.py`.

### Hungarian Optimal Assignment

After computing the cost matrix (Mahalanobis distances, with infeasible
pairs set to `1e9`), the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
finds the minimum total cost assignment.

Greedy IoU matching (the original RF-DETR tracker) minimises local choices.
The Hungarian algorithm minimises *global* cost. The difference matters when
two detections are close together: greedy can steal a match from a better
candidate.

The assignment is only used when both tracks and detections exist. After
assignment:

- Matched pairs: Kalman update.
- Unmatched tracks: `mark_missed()` — increment miss counter.
- Unmatched detections: spawn new tentative track.

**Code:** `_build_cost_matrix()`, `run_object_state_fusion()` in
`fusion/object_state.py`.

### Track Lifecycle

```
[tentative]  hit ≥ CONFIRM_HITS  →  [confirmed]
[tentative]  miss ≥ MAX_MISS     →  [deleted]
[confirmed]  miss ≥ MAX_MISS     →  [deleted]
```

Defaults: `CONFIRM_HITS = 3`, `MAX_MISS = 5`.

The intuition: require 3 consecutive hits before trusting a track avoids
outputting short spurious tracks from clutter. Allow 5 misses before deleting
a track avoids re-spawning the same object repeatedly during occlusions.

### RTS Smoother Per Track

After the forward pass, each confirmed track has a `history` list of
`ObjectFilterHistory` records. These are passed to `rts_smooth()` with
the object process noise parameters to get smoothed bounding-box trajectories.

The smoothed trajectories are smoother (less jitter), have smaller covariance,
and are the correct estimate given all frame observations for that track.

---

## 5. Semantic Priors

### The Insight

A Kalman filter's process noise `Q` quantifies how much the true state can
change between measurements. For a drone on a highway, velocity is nearly
constant: small `Q` is right. For a car at an urban intersection, velocity
can change sharply: large `Q` is right.

Gemma and Qwen already classify the scene. Rather than ignoring this
information, we use it to set `Q` and `R` appropriately before the filter runs.

### Noise Scale Derivation

**Scene type → process noise scale:**

| Scene type | Scale | Reasoning |
|---|---|---|
| highway | 0.6× | Nearly constant velocity, CV model fits well |
| aerial | 0.7× | Drone flight is smooth between manoeuvres |
| rural | 1.0× | Moderate dynamics, neutral prior |
| urban_street | 1.6× | Stops at lights, lane changes |
| intersection | 2.0× | Highest state-change density |
| parking | 1.8× | Slow, erratic motion |

These are multiplicative scales on `σ_pos` and `σ_vel` before `Q` is built.
A scale of 1.6 means process variance is 2.56× larger in urban scenes vs rural.

**RSSM temporal surprise → temporal noise scale:**

The RSSM model assigns a surprise score ∈ [0, 1] to each frame. High surprise
means the video contains an unexpected transition — a sudden camera move, a
new scene type, etc.

The mean RSSM surprise for a video drives a linear scale:

```
temporal_scale = 1.0 + 2.0 × mean_surprise
```

Range: [1.0 (no surprise), 3.0 (maximum surprise)].

**Combined process noise scale:**

```
combined = process_noise_scale × temporal_scale
```

Both scales are applied, so a surprising urban intersection gets up to
6× the baseline process noise.

**Urban canyon → GPS noise inflation:**

If Gemma detects objects suggesting tall structures (buildings, overpasses,
tunnels), GPS is likely multipath-affected. We inflate `σ_gps` by 2.5×, which
widens the GPS measurement covariance from 25 m² to 156 m². This tells the
filter to trust GPS less and rely more on the CV prediction and SfM constraints.

**Object speed priors:**

After each Kalman update for an object track, the velocity estimate is clamped
to a per-label maximum realistic speed:

- person: 3.5 m/s, car: 40 m/s, truck: 30 m/s, bicycle: 10 m/s, etc.

This is a hard constraint that prevents the filter from converging to a
physically implausible velocity (which can happen when a detection makes a
sudden jump due to ID switch or occlusion).

**Code:** `build_semantic_prior()` in `fusion/semantic_priors.py`.
**Code:** `_apply_speed_prior()` in `fusion/object_state.py`.

---

## 6. Coordinate Frames

Every measured quantity must be expressed in a known frame. The current
system uses:

| Frame | Origin | Usage |
|---|---|---|
| WGS-84 | Earth centre | Raw GPS lat/lon/alt |
| ENU | First GPS fix of mission | Platform state, all position measurements |
| SfM local | Arbitrary; defined by first posed frame | Camera centres from pycolmap |
| Normalised image | (0,0) top-left, (1,1) bottom-right | Object bbox state |

The ENU frame is established in `_build_gps_measurements()` by taking the
first non-null GPS sample as origin and converting all subsequent GPS samples
via `gps_to_enu()` (`pipeline/mapping/gps_registration.py`).

The SfM → ENU transform is established by the Umeyama alignment and applies
only to `sfm_position` measurements.

Object states are never projected into ENU. They live in image space because
we do not have calibrated camera-to-platform extrinsics at inference time.

---

## 7. Putting It All Together — A Worked Example

Say the video is 25 s of drone footage over an urban intersection with:

- 51 frames at 2 fps (frame timestamps t = 0, 0.5, 1.0, …, 25.0)
- GPS available at every frame (50 measurements)
- No IMU or barometer sidecar
- 10 SfM poses recovered from the background thread
- RF-DETR produces 51 frames × ~20 detections/frame
- Gemma classifies the scene as `urban_street`
- RSSM mean surprise = 0.6

**Semantic prior:**

```
process_noise_scale = 1.6  (urban_street)
temporal_noise_scale = 1.0 + 2.0 × 0.6 = 2.2
combined = 1.6 × 2.2 = 3.52
gps_noise_scale = 1.0  (no urban canyon objects detected)
```

**Map-state fusion:**

- GPS: 50 measurements, `σ_gps = 5.0 m`
- SfM: 10 poses, 10 co-located with GPS, Umeyama RMSE = 0.9 m,
  `σ_sfm = 2.0 + 0.9 = 2.9 m`
  → 10 additional `sfm_position` measurements
- Platform filter forward pass: 51 frame samples, 60 position updates,
  initial `P` trace ~25 m², final `P` trace ~2.1 m²
- RTS backward pass: final smoothed `P` trace ~1.8 m² (14% reduction)
- The early frames (where only forward GPS was available) benefit most

**Object-state fusion:**

- Hungarian assignment replaces greedy IoU
- Tracks with 3+ confirmed hits: suppose 31 out of 47 total
- RTS smoother runs over 31 confirmed tracks
- Output: per-frame smoothed bbox positions with velocity estimates

**Full fusion JSON summary excerpt:**

```json
{
  "semantic_prior": {
    "scene_type": "urban_street",
    "process_noise_scale": 3.52,
    "gps_noise_scale": 1.0,
    "temporal_noise_scale": 2.2
  },
  "map_state": {
    "sfm_alignment": {"scale": 0.94, "rmse_m": 0.9, "n_aligned_frames": 10},
    "diagnostics": {
      "mean_cov_trace_raw": 3.4,
      "mean_cov_trace_smoothed": 2.9,
      "sfm_measurements": 10,
      "gps_measurements": 50,
      "smoother_applied": true
    }
  },
  "object_state": {
    "track_count": 47,
    "confirmed_tracks": 31,
    "diagnostics": {"rts_smoothed_tracks": 31}
  }
}
```

---

## 8. Key Design Choices And Their Rationale

### Why constant velocity, not constant acceleration?

Constant acceleration requires either accurate IMU data or a second
derivative estimate, which is noisy from GPS alone. Constant velocity with
a large enough process noise covers most scenarios without over-fitting.

### Why Mahalanobis gating instead of IoU threshold?

IoU is scale-independent: a large box and a small box 10% overlapping
score the same as two medium boxes 10% overlapping. Mahalanobis is scale-
aware and uncertainty-aware. A track that has been occluded for 3 frames
has grown covariance; the gate correctly grows to allow detections that are
farther away.

### Why RTS and not particle smoother or factor graph?

For a linear Gaussian system (CV model + Gaussian measurements), RTS is
the *exact* minimum-variance smoother. It is also O(N) in the number of
frames. A particle smoother or factor graph is needed only when the process
model is nonlinear or the noise is non-Gaussian. Both are future extensions.

### Why semantic priors as noise scales and not as measurements?

If Gemma says "intersection," there is no direct geometric measurement
that corresponds to this fact. Using it as a prior on noise lets the
physical filter remain statistically consistent. If we fed semantic outputs
as position or velocity measurements, we would be hallucinating constraints
that have no geometric basis.

### Why Sim(3) and not SE(3)?

SfM is inherently scale-ambiguous. Forcing the alignment to use fixed scale
(SE(3)) would introduce systematic error wherever the SfM scale estimate
differs from the GPS metric scale. Umeyama's Sim(3) absorbs the scale
mismatch and produces metric-consistent measurements.

---

## 9. Reading The Artifacts

After a run that produces `full_state_fusion.json`, these are the most
informative fields:

| Field | What it tells you |
|---|---|
| `map_state.sfm_alignment.rmse_m` | How well SfM and GPS agreed. < 2 m = good. > 10 m = suspect GPS or degenerate SfM |
| `map_state.diagnostics.mean_cov_trace_smoothed` | Mean position+velocity uncertainty after smoothing. < 5 = good for 5 m GPS. > 50 = filter is struggling |
| `map_state.diagnostics.mean_cov_trace_raw` vs `smoothed` | Ratio shows how much the smoother helped. 10–30% reduction is typical |
| `object_state.confirmed_tracks` vs `track_count` | Large gap (e.g. 5 confirmed out of 50 total) means many false-positive spawns, likely noisy detections |
| `semantic_prior.process_noise_scale` | > 3 means the scene was classified as highly dynamic and/or high RSSM surprise |
| `platform.diagnostics.mean_innovation_norm` | Mean Kalman residual norm. Should be roughly `σ_gps`. Much larger = systematic GPS error or coordinate frame mismatch |

If `map_state.status == "skipped"`, the most common reasons are:

1. No GPS in the video — check that the video file has GPS atoms or a
   GPS sidecar.
2. All GPS samples are null — the GPS extractor ran but found no valid fixes.
3. SfM alignment skipped — fewer than `STATE_FUSION_SFM_MIN_FRAMES` (6)
   co-located frames.

---

## 10. Extension Points

The four-layer architecture is designed to add sensors without touching
existing code.

**Adding LiDAR range constraints:**

1. Add `"lidar_range"` to `PlatformMeasurement.kind`.
2. In `summaries.py`, build a range measurement with appropriate `H` and `R`.
3. Add an update path in `PlatformStateFilter` (range = scalar measurement,
   `H` = unit vector from platform to obstacle).

**Adding camera-to-world object projection:**

1. In `object_state.py`, load camera intrinsics and platform-to-camera
   extrinsics.
2. After each `ObjectKalmanFilter` update, project the bbox centroid into
   ENU using the current platform pose from the map-state smoother.
3. The result is a world-frame track position that can be stored separately.

**Adding a turn-rate model for non-CV platforms:**

1. In `fusion/filters/platform.py`, add a `CTRV` (constant turn-rate velocity)
   variant of `predict()` that takes yaw rate as control input.
2. The rest of the pipeline is unchanged; only the `F`, `B`, `Q` matrices differ.

---

## Related Code

| Concept | File |
|---|---|
| Kalman filter (platform) | `fusion/filters/platform.py` |
| Kalman filter (object) | `fusion/filters/object_filter.py` |
| RTS smoother | `fusion/filters/rts_smoother.py` |
| Umeyama alignment | `fusion/visual_pose.py` |
| Semantic priors | `fusion/semantic_priors.py` |
| Object fusion pipeline | `fusion/object_state.py` |
| Map fusion + smoother | `fusion/map_state.py` |
| Full orchestration | `fusion/summaries.py` |
| Workflow step | `workflows/local/steps_fusion.py` |
| Config surface | `pipeline/core/config.py` (STATE_FUSION_*, OBJECT_FUSION_*, MAP_FUSION_*) |

## Related Deep Dives

- [03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md)
- [09_probabilistic_state_fusion_requirements.md](09_probabilistic_state_fusion_requirements.md)
- [10_probabilistic_state_fusion_architecture.md](10_probabilistic_state_fusion_architecture.md)
- [11_probabilistic_state_fusion_implementation_order.md](11_probabilistic_state_fusion_implementation_order.md)
