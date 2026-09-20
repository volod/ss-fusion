# Local Research Pipeline (Playground 2) -- `ssv_vdp`

Standalone package in the ss-fusion workspace (`packages/ss-fusion`, console script
`ssv`) that runs a 36-step per-mission research workflow over a single video:
perception, physical-SIGINT sensor fusion, tracking and 3D reconstruction, SSL
adaptation, edge distillation, threat inference, and reasoning audit. Its outputs
(fine-tuned embedders, ONNX exports, annotated data) feed back into the production
server.

## Entry points

| Command | Source | Purpose |
| --- | --- | --- |
| `ssv` | `ssv_vdp/cli.py` | Main runner: `ssv --mode local --videos-dir .data/videos` |
| `ssv-export` | `scripts/export_onnx.py` | ONNX export of adapted models, with a `model-artifact` manifest per file |
| `ssv-gallery` | `scripts/build_gallery.py` | HTML gallery of run artifacts |
| `ssv-split-audio` / `ssv-prepare-audio` / `ssv-play-drone` | `scripts/` | Drone-audio dataset tooling |

Cosmos3 world-model step: `ssv --mode local --cosmos3` (local diffusers,
hardware-selected variant) or `--cosmos3-api-url http://localhost:8000` (vLLM-Omni
sidecar). Startup preflight checks model caches, packages, sidecar presence, and
service reachability before any heavy work starts.

## Orchestration

`pipeline/graph.py` builds a LangGraph state machine over `PipelineState`
(`pipeline/state.py`); `pipeline/runner.py` (`run_local()`) drives it with
per-phase helpers under `pipeline/runner_helpers/`.

```
Phase 1: init -> extract_frames -> index_vectors
Phase 2: gemma_analysis -> [parallel: florence / asr / ocr / depth / detection]
         -> tracking (yolo_sam, gemma_tracking, rf-detr) -> mapping (SfM, splat)
Phase 3: ssl_finetune -> [ssl_gate] -> distill -> onnx_export -> ft_search -> compare
Phase 4: multi_model_compare -> synthesis -> audit -> emit_analytics -> END
```

The canonical 36-step table lives in `docs/reference/pipeline.md`; the
walk-through is `docs/guide/local_path.md` with per-phase deep dives under
`docs/learning_path/`.

## Step families (`packages/ss-fusion/src/ssv_vdp/steps/`)

| Family | Modules | What happens |
| --- | --- | --- |
| Perception core | `embed`, `caption/` (Florence, Gemma), `perception/`, `scenetok` | CLIP+DINOv3 embedding, captioning, scene analysis, ASR/OCR/depth/detection |
| Physical SIGINT | `fusion`, `physical_state`, `field_state` | RF/SDR, thermal, multispectral, event-camera, LiDAR, radar, GNSS-R, IMU, atmospheric, gas/radiation, acoustic sidecars fused into time-aligned context (mock sidecars when no hardware) |
| Tracking and 3D | `yolo_sam`, `gemma_tracking`, `semantic_graph`, `map` | YOLO+SAM segmentation, Gemma-directed RF-DETR tracking, semantic environment graph, pycolmap SfM + nerfstudio splat |
| Adaptation | `ssl`, `distill`, `adaptation/`, `edge/`, `model_advisor` | DAE + track-aware contrastive SSL (MoCo-style), edge distillation (ViT-S/14, EfficientViT-B1 ONNX), multi-model comparison |
| Threat and policy | `threat_primitives`, `local_threat`, `global_threat`, `threat_contradictions`, `threat_eval`, `policy` | Evidence-gated threat primitives (two-source gate), clip-level assessment, fixed-vocabulary action recommendation, contradiction surfacing |
| Specialist | `drone_detection`, `drone_audio`, `drau_eval`, `cosmos3` | Drone visual/acoustic detection, world-model embedding step |
| Reporting | `report`, `report_helpers/`, `agentic/processor.py` | Synthesis report, Qwen3 reasoning audit, `agentic_flow.md` audit log |

## Run artifacts

Each run writes under `$DATA_DIR` (default `.data/` in this project):
`analysis_summary.json` (modality coverage, degradation, artifact health),
`threat_primitives.json`, `local_threat_assessment.json`, `policy_decision.json`,
`agentic_flow.md`, gallery/report HTML, ONNX exports, and fine-tune checkpoints.
`ssv_vdp/scripts/analyse_local_run.py` inspects a completed run.

`export_onnx.py` exports the way the pipeline's export step does: a single-input wrapper over a
float32 CPU copy, xformers and antialiased position-embedding interpolation off, the TorchScript
exporter (`dynamo=False`, which keeps `dynamic_axes`), and opset 18 by default. Before this it
failed on torch 2.12 (dynamo exporter), with `--device cuda` (CPU trace input), at opset 17
(antialiased bicubic upsample), and in `--quantize` (onnxruntime takes `weight_type`). On the
`nar` run's SSL checkpoint it exports with a parity max difference below 1e-5 and quantizes to
INT8 (346 MB to 90 MB).

## Feedback loop into production

- SSL/fine-tune artifacts are consumed by worker `SUPERVISED_FINETUNE` / `REEMBED` jobs.
- Distilled ONNX edge models are the deployment candidates for edge nodes.
- Active-learning frame tags flow to CVAT (video repository).

## Known open research themes

`docs/learning_path/18_future_directions.md` records what is deliberately not
implemented yet: full cross-modal temporal SSL, environmental field models (GPR),
systematic cross-sensor calibration / formal contradiction modeling, and
cross-mission global threat inference. Forward tasks in
[`../plan.md`](../plan.md) reference these where relevant.
