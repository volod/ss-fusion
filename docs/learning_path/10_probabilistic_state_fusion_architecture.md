# Probabilistic State Fusion Architecture

This document describes the actual subsystem architecture that was built.

For the mathematical deep dive into each component, see
[12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md).

## Design Goal

Build a fusion subsystem that can:

- estimate platform state over time from heterogeneous sensors
- tighten position estimates with visual-pose constraints when SfM succeeds
- track dynamic objects with proper uncertainty and optimal assignment
- smooth the full trajectory in a backward pass after all data is seen
- adapt noise parameters to scene semantics from VLM outputs
- remain useful when only a subset of sensors or models is available

## Architectural Principle

Five layers, cleanly separated:

1. **Sidecar ingestion** — load raw telemetry, validate, normalize time
2. **Typed measurements** — convert to `PlatformMeasurement` with explicit covariance
3. **Fusion engine** — predict/update Kalman cycles + Mahalanobis gating
4. **Smoother** — RTS backward pass for minimum-variance estimates
5. **Downstream consumers** — `frame_facts_json`, `full_state_fusion.json`, markdown reports

Each layer talks to the next only through its typed output. No layer reaches
into the internals of another.

## Actual Package Layout

```
src/selfsuvis/pipeline/fusion/
  __init__.py              # public API
  measurements.py          # PlatformMeasurement typed measurement
  state.py                 # PlatformFusionResult, FullFusionResult, ...
  sidecars.py              # IMU and barometer sidecar loaders
  summaries.py             # run_platform_state_fusion(), run_full_state_fusion()
  visual_pose.py           # Umeyama Sim(3) alignment, SfM → ENU measurements
  semantic_priors.py       # VLM-grounded noise adaptation
  object_state.py          # probabilistic object fusion pipeline
  map_state.py             # map-state fusion + RTS trajectory smoothing

  filters/
    __init__.py
    platform.py            # 6-state constant-velocity Kalman filter
    object_filter.py       # per-object KF with Mahalanobis gating + history
    rts_smoother.py        # Rauch-Tung-Striebel fixed-interval smoother
```

The workflow integration lives in:

```
src/ssv_vdp/
  steps_fusion.py          # step_platform_state_fusion(), step_full_state_fusion()
```

## Layer 1 — Sidecar Ingestion

`fusion/sidecars.py` loads:

- `<video>.imu.jsonl` — per-row `{"t": float, "ax": float, "ay": float, "az": float}`
- `<video>.baro.jsonl` — per-row `{"t": float, "alt": float}`

Discovery is best-effort. Missing sidecars do not fail the run.

## Layer 2 — Typed Measurements

All physical observations are `PlatformMeasurement` instances:

```python
@dataclass(frozen=True)
class PlatformMeasurement:
    kind: str  # "gps_position" | "barometer_altitude" | "imu_accel" | "sfm_position"
    t_sec: float
    values: Tuple[float, ...]
    covariance: Tuple[Tuple[float, ...], ...]
    source: str  # "video_gps" | "imu_sidecar" | "sfm_umeyama" | ...
    frame: str  # "enu" | "enu_assumed"
    quality: str  # "nominal" | "approx_world_frame"
```

Object observations are implicit in the per-frame detection dict from RF-DETR.

## Layer 3 — Fusion Engine

Three parallel engines:

### 3a — Platform Filter (`filters/platform.py`)

State: `x = [px, py, pz, vx, vy, vz]`

Constant-velocity model with optional acceleration control. Three measurement
update paths: `update_position`, `update_altitude`, `set_acceleration`.

### 3b — Object Filters (`filters/object_filter.py`)

One `ObjectKalmanFilter` per track. State: `[cx, cy, w, h, vcx, vcy]` in
normalised image coordinates.

Association: Hungarian algorithm on Mahalanobis distance² matrix. Gate at
χ²(4 DOF, p=0.99) = 13.28. Infeasible pairs excluded from the solver.

### 3c — Visual-Pose Constraints (`visual_pose.py`)

Umeyama Sim(3) alignment: given SfM camera centres and GPS-ENU positions for
the same frames, finds the optimal rotation R, translation t, and scale s such
that `p_ENU ≈ s·R·p_SfM + t`. The aligned positions become `sfm_position`
measurements fed into the platform filter.

## Layer 4 — Smoothers

### 4a — Platform RTS (`map_state.py` calls `filters/rts_smoother.py`)

After the forward Kalman pass, runs the Rauch-Tung-Striebel backward sweep.
At each step `k` backward from `N-1` to `0`:

```
G_k = P_{k|k} F_{k+1}^T  P_{k+1|k}^{-1}
x_{k|N} = x_{k|k} + G_k (x_{k+1|N} − F_{k+1} x_{k|k})
P_{k|N} = P_{k|k} + G_k (P_{k+1|N} − P_{k+1|k}) G_k^T
```

The predicted covariance `P_{k+1|k}` is approximated from the CV model using
the inter-frame `dt`. This is exact when no measurements arrive between
consecutive frames and a conservative underestimate otherwise.

### 4b — Object RTS (`object_state.py`)

The same RTS smoother is run per confirmed track after all frames are
processed. `ObjectKalmanFilter.history` accumulates `(t, x_filtered,
P_filtered, x_pred, P_pred)` at each update step.

## Layer 5 — Downstream Consumers

### `full_state_fusion.json`

Written by `step_full_state_fusion()`. Contains all four fusion results in a
single JSON. Schema:

```json
{
  "platform": { "status": "...", "posterior_samples": [...], "diagnostics": {...} },
  "map_state": {
    "sfm_alignment": {"scale": 1.02, "rmse_m": 0.87, "n_aligned_frames": 21},
    "smoothed_samples": [
      {"t_sec": 0.5, "position_enu_m": {...}, "velocity_enu_mps": {...},
       "covariance_diag": [0.4, 0.4, 0.5, 0.2, 0.2, 0.3], "cov_trace": 2.0,
       "quality": "good"}
    ],
    "diagnostics": {...}
  },
  "object_state": {
    "track_count": 47,
    "confirmed_tracks": 31,
    "per_frame": [[{"track_id": 1, "label": "car", "bbox_norm": [...], ...}]]
  },
  "semantic_prior": {
    "scene_type": "urban_street",
    "process_noise_scale": 1.6,
    "gps_noise_scale": 1.0,
    "temporal_noise_scale": 1.8
  }
}
```

### `frame_facts_json["state_fusion"]`

The platform posterior sample for a frame is optionally injected here.
The full fusion result is not stored per-frame to keep the DB compact.

### `VideoKnowledge`

`knowledge.add_state_fusion(posterior_samples)` injects the ENU position
and quality into the context that Gemma and DeepSeek consume in later steps.

## Complete Data Flow

```
video + sidecars
  → gps_extraction (media/gps.py)
  → build_gps_measurements_scaled()  ─┐
  → load_imu_sidecar()               ─┤
  → load_baro_sidecar()              ─┤  typed measurements
  → build_sfm_frame_positions        ─┤  (PlatformMeasurement)
  → align_sfm_to_enu() (Umeyama)     ─┘
         │
         ▼
  event timeline  (sorted by t_sec, priority)
         │
         ▼
  PlatformStateFilter  (forward KF pass)
     predict(dt) → update_position / update_altitude / set_acceleration
         │
         ▼
  per-frame posterior samples  (x, P recorded at frame timestamps)
         │
         ├─→ state_fusion.json  (GPS-only baseline, backward compatible)
         │
         ▼
  rts_smooth()  (backward pass)
         │
         ▼
  MapFusionResult  (smoothed trajectory with full covariance)
         │
         ├──────────────────────────────────┐
         │                                  │
  per-frame RF-DETR detections          SemanticPrior
         │                              (proc/gps scale from Gemma/RSSM)
         ▼
  ObjectKalmanFilter × N  (per track, forward pass)
     predict → Mahalanobis gate → Hungarian assign → update → history
         │
         ▼
  rts_smooth() per confirmed track
         │
         ▼
  ObjectFusionResult  (per-frame smoothed bboxes with uncertainty)
         │
         ▼
  FullFusionResult.to_dict()
         │
         ▼
  full_state_fusion.json
```

## Degradation Behavior

| Available data | Outcome |
|---|---|
| GPS only | Platform KF runs, SfM skipped, RSSM prior skipped |
| GPS + IMU | IMU sets acceleration control, tighter velocity estimates |
| GPS + baro | Altitude dimension updated independently |
| GPS + SfM (≥6 poses) | Umeyama alignment, extra position measurements |
| GPS + SfM + tracking | All four layers active |
| No GPS | `status="skipped"`, no posterior, no error |
| No tracking results | Object layer `status="skipped"`, others unaffected |
| SfM < 6 poses | Visual pose skipped, GPS-only map fusion |

## Configuration Surface

```
# Platform state
STATE_FUSION_ENABLED              (bool, default true)
STATE_FUSION_GPS_POS_STD_M        (float, default 5.0)
STATE_FUSION_BARO_ALT_STD_M       (float, default 2.5)
STATE_FUSION_IMU_ACCEL_STD_MPS2   (float, default 1.5)
STATE_FUSION_PROCESS_POS_STD_M    (float, default 0.75)
STATE_FUSION_PROCESS_VEL_STD_MPS  (float, default 1.5)
STATE_FUSION_INIT_VEL_STD_MPS     (float, default 3.0)
STATE_FUSION_CONTEXT_GAP_SEC      (float, default 1.0)

# Visual pose
STATE_FUSION_SFM_POS_STD_M        (float, default 2.0)
STATE_FUSION_SFM_MIN_FRAMES       (int, default 6)

# Object fusion
OBJECT_FUSION_ENABLED             (bool, default true)
OBJECT_FUSION_OBS_NOISE           (float, default 0.005)
OBJECT_FUSION_CONFIRM_HITS        (int, default 3)
OBJECT_FUSION_MAX_MISS            (int, default 5)

# Map smoothing
MAP_FUSION_SMOOTH                 (bool, default true)
```

## Related Deep Dives

- [09_probabilistic_state_fusion_requirements.md](09_probabilistic_state_fusion_requirements.md)
- [11_probabilistic_state_fusion_implementation_order.md](11_probabilistic_state_fusion_implementation_order.md)
- [12_probabilistic_fusion_deep_dive.md](12_probabilistic_fusion_deep_dive.md)
