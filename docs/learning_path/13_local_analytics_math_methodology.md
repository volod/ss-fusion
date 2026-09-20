# Local Analytics Math And Methodology

This deep dive explains the derived diagnostics emitted by the local analytics
step. These metrics do not prove that a run is correct. They are triage signals:
they tell you where to inspect raw artifacts first.

## Inputs

The diagnostics are computed from artifacts already produced by the local
pipeline:

- frame metadata: `frames_metadata.json`
- captions: `scene_captions.md`, `detailed_captions.md`
- detections and masks: `yolo_sam_results.json`
- directed tracking: `gemma_tracking_results.json`
- temporal surprise: `rssm_temporal.json`
- adaptation: `finetune_stats.md`, `distill_stats.md`, `edge_models/`
- mapping: `3d_map/map_stats.json`
- runtime health: `runtime_metrics.json`

## Modality Completeness

Completeness is the mean availability of eight evidence channels:

```text
C = mean(F, Q, A, O, W, T, M, E)
```

Where:

- `F` is Florence caption coverage.
- `Q` is Qwen detailed-caption coverage.
- `A` is ASR frame coverage.
- `O` is OCR frame coverage.
- `W` is `1` when the world model produced usable output, otherwise `0`.
- `T` is `1` when directed tracking produced detections, otherwise `0`.
- `M` is `1` for a non-degraded 3D map, `0` for degraded or absent mapping.
- `E` is `1` when the edge ONNX model exists, otherwise `0`.

Interpretation:

- `C > 0.75`: broad multimodal evidence is present.
- `0.45 <= C <= 0.75`: usable run, but inspect missing channels.
- `C < 0.45`: the run is likely too sparse for confident synthesis.

## Detection Distribution

Detection density is the mean number of detections per frame:

```text
density = total_detections / n_frames
```

Detection count variation uses the coefficient of variation:

```text
CV = std(counts_per_frame) / mean(counts_per_frame)
```

High `CV` means object evidence is temporally uneven. That can be correct for a
video with scene cuts, but it can also indicate detector instability.

Class diversity uses normalized Shannon entropy:

```text
H = -sum_i p_i log(p_i)
H_norm = H / log(K)
```

Where `p_i` is the class-count fraction and `K` is the number of non-empty
classes. `H_norm` is `0` for a single dominant class and approaches `1` when
classes are evenly distributed.

## Tracking Fragmentation

The directed-tracking diagnostics separate object volume from temporal identity:

```text
fragmentation = unique_track_ids / total_tracked_detections
persistence = mean_track_length_frames / n_frames
```

Good tracking usually has low fragmentation and moderate persistence. A short
aerial road video may have many vehicle detections, but if every frame creates
new IDs then `fragmentation` rises and `persistence` falls.

Rules of thumb:

- `fragmentation < 0.10`: stable identity assignment.
- `0.10 <= fragmentation <= 0.30`: inspect track visualizations.
- `fragmentation > 0.30`: likely fragmented tracking, especially when
  persistence is also low.

## Temporal Surprise

RSSM surprise is treated as a scalar time series:

```text
surprise_std = std(surprise_scores)
peak_rate = n_peak_frames / n_frames
peak_detection_overlap = peaks_with_detections / n_peak_frames
```

`surprise_std` measures how concentrated temporal novelty is. High peak overlap
with detections suggests the temporal model reacts to visible object changes.
Low overlap suggests camera motion, compression artifacts, blur, lighting, or an
embedding-space issue.

## Map Quality

Sparse map diagnostics normalize map output by video length:

```text
points_per_pose = sfm_points / max(sfm_poses, 1)
pose_coverage = clamp(sfm_poses / n_frames, 0, 1)
```

`pose_coverage` asks whether the mapper recovered camera poses for a reasonable
fraction of frames. `points_per_pose` asks whether the poses produced enough
geometry to be useful.

For short aerial clips, a low number of points may be expected when there is
little parallax or many repeated road textures. The diagnostic should trigger
inspection of `3d_map/map_stats.json`, sparse-map viewers, and source frames.

## Adaptation Efficiency

The distillation metric rewards retained retrieval quality under compression:

```text
adaptation_efficiency = distilled_R@1 / compression_ratio
```

This is intentionally conservative. A very small student with poor Recall@1 is
not useful just because it compresses well. Conversely, a student with high
Recall@1 and moderate compression is a stronger edge-deployment candidate.

## Artifact Normalization

Artifact density helps catch unexpectedly sparse or explosive outputs:

```text
artifact_density_per_frame = file_count / n_frames
artifact_mb_per_min = total_artifact_MB / video_minutes
```

Low density may mean stages were skipped. Very high density may mean annotation,
cache, or media artifacts are growing faster than expected.

## Quality Score

The quality score is a weighted triage score:

```text
quality =
100 * clamp01(
    0.35 * modality_completeness
  + 0.20 * tracking_quality
  + 0.15 * map_quality
  + 0.15 * training_quality
  + 0.15 * world_model_quality
  - warning_penalty
)
```

Where:

- `tracking_quality = clamp01(0.7 * persistence + 0.3 * (1 - fragmentation))`
- `map_quality = 0.6 * clamp01(points / 200) + 0.4 * clamp01(poses / n_frames)`
- `training_quality = distilled_R@1` when available, otherwise `1` if an edge model exists.
- `world_model_quality` is `1` when the world model is usable, otherwise `0`.
- `warning_penalty = min(0.25, 0.05 * n_warnings)`.

Interpretation:

- `80-100`: strong run, inspect only high-risk final reasoning claims.
- `55-80`: usable run with specific weak stages.
- `30-55`: partial run; use raw artifacts, not the final narrative, as the source of truth.
- `<30`: triage run; rerun with fixed models, telemetry, or memory settings before trusting synthesis.

## Methodology Checklist

When reading a completed run:

1. Start with warnings and `quality_score`.
2. Check `modality_completeness` to see whether final synthesis had enough evidence.
3. Inspect tracking fragmentation before trusting object-state fusion.
4. Inspect map pose coverage before trusting spatial claims.
5. Compare RSSM surprise peaks against frame images.
6. Treat adaptation efficiency as an edge-readiness signal, not as semantic correctness.
7. Always verify final narrative claims against the artifact family that supports them.

## Code Pointers

- Diagnostics dataclass: `src/selfsuvis/analytics/models.py`
- Diagnostics computation: `src/selfsuvis/analytics/loader.py`
- Pipeline summary payload: `src/ssv_vdp/pipeline/runner.py`
- HTML report rendering: `src/selfsuvis/visualization/report.py`
