# Local Run Artifact Analysis

This deep dive is for the moment after a local run has finished and you need to answer:

- Which artifacts are trustworthy?
- Which stages silently degraded?
- What should I inspect first?
- How do the artifacts relate back to the code?

## Why This Matters

The local pipeline is intentionally ambitious. A run can complete and still contain failures that
only show up in the artifacts:

- empty Florence captions
- zero RF-DETR tracks after successful Gemma segmentation
- world-model clip failures with RSSM fallback still succeeding
- good ONNX export but weak distillation quality

Treat the run directory as evidence, not as proof that every stage worked.

## New Startup Contract

Current local runs have a stricter contract than older notes in this learning path:

- `ssv --mode local` now runs a startup preflight before frame extraction
- missing local dependencies or uncached model weights are treated as startup errors
- degraded-but-allowed conditions are still logged as warnings
  - Qdrant unreachable
  - mapper API unreachable
  - drone-detection dataset cache still cold

This changes how you read a failed run:

- if the run never started, inspect `preflight:` log lines rather than the output directory
- if the run started and later degraded, the remaining issues are usually runtime quality limits
  rather than missing packages or first-run downloads

## Recommended Inspection Order

1. `final_stats.md`
   Use this to identify slow stages, skipped stages, and aggregate metrics.
2. `scene_captions.md` and `detailed_captions.md`
   Check whether caption quality is informative or silently empty.
3. `yolo_sam_results.json` and `detection_comparison.md`
   Confirm object counts and detector agreement.
4. `gemma_tracking_results.json`
   Verify whether Gemma-directed tracking actually found tracks or only masks.
5. `rssm_temporal.json`
   Identify surprise peaks and compare them to scene changes.
6. `finetune_stats.md`, `distill_stats.md`, and `edge_models/`
   Determine whether adaptation produced a useful deployable edge artifact.
7. `3d_map/`
   Confirm that map generation produced actual geometry, not just placeholder outputs.
   Read `map_stats.json` first, then `map_quality_advisor.md` to distinguish reconstruction failure
   from capture failure.

## Use The Analytics Tooling

Generate a report:

```bash
ssv --mode analyse --run-dir .data/local_runs/drone_mission
```

Module form:

```bash
python -m ssv --mode analyse --run-dir .data/local_runs/drone_mission
```

This gives you:

- a timeline chart
- detection charts
- embedding PCA and similarity plots
- training charts
- an HTML report with warnings and artifact inventory
- a diagnostics block with modality completeness, quality score, fragmentation, map coverage, temporal surprise dispersion, and adaptation efficiency

For the equations behind those diagnostics, read
[Local analytics math and methodology](13_local_analytics_math_methodology.md).

## Questions To Ask Per Artifact Family

### Per-frame artifacts

- Are frame timestamps monotonic and plausible in `frames_metadata.json`?
- Are Florence captions non-empty and semantically stable?
- Do Qwen captions add information beyond Florence, or just paraphrase?
- Does ASR inject relevant context or obvious contamination?

### Detection and tracking artifacts

- Do YOLO counts match what you can visually verify in annotated frames?
- Did Gemma tracking produce track IDs, or only SAM masks?
- Are zero detections a model problem or a label-vocabulary problem?
- Is `tracking_fragmentation = unique_track_ids / total_detections` high enough to suggest broken temporal identity?

### Temporal artifacts

- Do RSSM surprise peaks align with visible state changes?
- If surprise is high everywhere, is the embedder unstable or the scene genuinely dynamic?
- Do surprise peaks overlap with detections, or are they driven by camera motion, blur, or model noise?

### Adaptation artifacts

- Did SSL loss improve meaningfully?
- Did the student retain retrieval quality after compression?
- Does the ONNX artifact exist and match the reported dimensions?

### Mapping artifacts

- Does `3d_map/map_stats.json` show real poses and points?
- Does `3d_map/map_stats.json` report true `sfm_poses`, or mostly interpolated frame anchors?
- Did the mapper fall back to `sfm_sparse+semantic_pseudo3d` or another degraded recovery path?
- Is the Gaussian splat viewable, or was only a placeholder file written?
- Is pose coverage high enough for the video length, and are there enough points per pose?
- Does `3d_map/map_quality_advisor.json` blame the source video or the mapper?
- Are the advisor's measured signals consistent with what you see by eye:
  short clip, weak parallax, high altitude / wide FOV, low resolution, blur, or exposure flicker?
- If the advisor says exposure and sharpness are good but parallax is poor, stop tuning models and
  redesign the capture path.

### How To Read A Degraded 3D Map

The current mapping stack separates two different situations that used to be blurred together:

- The mapper failed because the source video was weak.
- The mapper degraded gracefully and produced a richer pseudo-3D map from sparse SfM, detections,
  tracks, and depth priors.

When `map_stats.json` reports degraded quality, inspect these fields together:

- `sfm_poses`
  Real poses recovered by pycolmap.
- `frame_anchor_count`
  Additional interpolated anchors used to stabilize the output.
- `point_count`
  Total points in the final sparse or pseudo-3D cloud.
- `quality_note`
  Why the fallback was activated.

Interpretation:

- low `sfm_poses`, low `point_count`: the source video is usually too weak even for fallback
- low `sfm_poses`, high `point_count`, semantic pseudo-3D note: the run recovered a usable but not
  metric-quality map
- high `sfm_poses`, high `point_count`: this is the regime where Gaussian splatting and spatial
  reasoning become trustworthy

### Drone Mission Example

In the learning-path drone run, the map advisor reported a common aerial failure pattern:

- exposure consistency was strong
- sharpness was strong
- feature richness was strong
- adjacent-frame matchability was strong
- parallax was weak
- field scale was weak
- clip duration was too short

That combination means:

- the video is clean enough for feature matching
- the aircraft did not move through geometry in a way that supports robust triangulation
- the camera was too high, too wide, too overhead, or all three

For that class of run, the right corrective action is not "change SfM library".
It is "capture a longer, lower, more oblique, more cross-linked flight path".

## New Failure Pattern: Startup-Clean, Runtime-Degraded

After the preflight changes, a useful mental split is:

- **startup-clean**:
  all required local models were already cached, and the run did not depend on surprise installs
- **runtime-degraded**:
  the run still completed with lower-quality artifacts because the scene, clip, or hardware was weak

Examples:

- Stage 2 distillation warning about mismatched CUDA tensor dtypes was a code bug and should no longer appear
- SceneTok being skipped despite `SCENETOK_ENABLED=true` in `.env` was a CLI/env bug and should no longer appear
- a short 10-second aerial clip still produces degraded mapping because that is a capture constraint, not a startup issue

## Code Traceback Map

When an artifact looks wrong, trace it back from the run directory to code:

- captions: `src/selfsuvis/pipeline/vision/florence.py`
- local caption/report orchestration: `src/ssv_vdp/steps/caption.py`
- report writers: `src/ssv_vdp/steps/report.py`
- analytics loader: `src/selfsuvis/analytics/loader.py`
- report generation: `src/selfsuvis/visualization/report.py`
- mapping advisor: `src/selfsuvis/pipeline/mapping/quality_advisor.py`
- sparse-map builder and degraded fallback: `src/selfsuvis/pipeline/mapping/builder.py`
- SfM matching policy: `src/selfsuvis/pipeline/mapping/sfm.py`

## Practical Exercises

1. Find one completed run with a warning in the analytics report. Explain the warning using raw artifacts.
2. Compare `scene_captions.md` and `detailed_captions.md` for ten frames. Identify where Qwen adds materially new information.
3. For each RSSM peak frame, open the corresponding image and decide whether the peak is justified.
4. Inspect `gemma_tracking_results.json` and explain why tracking succeeded or failed.
5. Verify the reported ONNX and checkpoint sizes against actual files on disk.
