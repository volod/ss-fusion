# Runtime And Study Guide

This guide explains how to approach the repo as a human learner.
The goal is not only to run the pipeline once, but to understand why each stage exists and how evidence moves through the stack.

## Who This Is For

This guide is for a person who needs one of these outcomes:

- understand the repo well enough to modify it
- debug a bad local run without guessing
- explain the pipeline to another engineer or operator
- learn the perception stack without reading every file in order

If you only want to run the system, use [`README.md`](../../README.md) and [`local_path.md`](../guide/local_path.md).
If you want to understand the system, start here.

## Source Of Truth

For runtime behavior, the main local source is:

- [`src/ssv_vdp/pipeline/runner.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/runner.py)

Supporting step modules live here:

- [`src/ssv_vdp/steps/embed.py`](../../packages/ss-fusion/src/ssv_vdp/steps/embed.py)
- [`src/ssv_vdp/steps/caption/`](../../packages/ss-fusion/src/ssv_vdp/steps/caption)
- [`src/ssv_vdp/steps/yolo_sam.py`](../../packages/ss-fusion/src/ssv_vdp/steps/yolo_sam.py)
- [`src/ssv_vdp/steps/gemma_tracking.py`](../../packages/ss-fusion/src/ssv_vdp/steps/gemma_tracking.py)
- [`src/ssv_vdp/steps/map.py`](../../packages/ss-fusion/src/ssv_vdp/steps/map.py)
- [`src/ssv_vdp/steps/ssl.py`](../../packages/ss-fusion/src/ssv_vdp/steps/ssl.py)
- [`src/ssv_vdp/steps/distill.py`](../../packages/ss-fusion/src/ssv_vdp/steps/distill.py)
- [`src/ssv_vdp/steps/report.py`](../../packages/ss-fusion/src/ssv_vdp/steps/report.py)

The learning path is broader than the runner implementation itself. That is intentional:

- the runner is the execution path
- the learning path is the conceptual path
- some conceptual steps are grouped or optional in code

Current reality:

- the monolithic local runner reports **32 runtime/post-run steps**
- the learning path sometimes uses a more granular conceptual breakdown
- production indexing is a different execution path from local full-analysis

Treat these as three different views of the same system, not as contradictions.

## Current Codebase Shape

The repo uses a `src/` layout. The package roots you should keep in your head are:

```text
src/selfsuvis/app
src/selfsuvis/models
src/selfsuvis/pipeline
src/selfsuvis/scripts
src/selfsuvis/worker
tests/
docs/
```

Do not mentally map the project as top-level `pipeline/`, `models/`, and `worker/` folders anymore.
Some older prose still uses those names conceptually, but the code lives under `src/selfsuvis/...`.

## Best Mental Model

Read the system in five layers:

1. Input and memory: frames, embeddings, retrieval
2. Evidence extraction: captions, speech, OCR, depth, detections
3. Physical sensing: RF, thermal, inertial, acoustic, environmental side channels
4. Structured reasoning: tracking, temporal embeddings, multimodal experts, map building
5. Adaptation and audit: SSL, distillation, export, evaluation, final synthesis

That model is still the easiest way to think about the pipeline even though the current
runner is organized as 32 runtime/post-run steps.

## How A Human Should Study It

Do not start by reading every file.
Use this order instead:

1. Generate or inspect one real local mission output directory.
2. Read ``architecture.md`` and [`pipeline.md`](../reference/pipeline.md).
3. Read [`local_path.md`](../guide/local_path.md).
4. Read the deep-dive doc for the phase you care about.
5. Open the code modules only after you know what question you are trying to answer.

Good questions:

- What evidence is created here?
- What artifact is written here?
- What later step depends on this output?
- What can fail silently?
- What should a human verify manually?

Weak questions:

- What does every line do?
- Which model is coolest?
- Can I memorize all options before I run anything?

## One Required Conceptual Detour

Before you go deep on Steps 9-20, read
[03_sensor_fusion_fundamentals.md](03_sensor_fusion_fundamentals.md).

That session covers the concepts that most often cause confusion in this repo:

- time alignment vs actual simultaneity
- coordinate frames vs image coordinates
- sensor absence vs negative evidence
- calibration errors vs model errors
- uncertainty and contradiction handling

Without those ideas, the sensor phase turns into a list of modalities.
With them, it becomes a coherent system.

## What Changed Recently

If you are returning to the repo after an older version, these are the changes that matter most:

- the project now uses `pyproject.toml` as the dependency source of truth
- environment bootstrapping is centered on `ssv-env`
- tests are now mostly organized to mirror `src/selfsuvis/`
- shared `pipeline.core` helpers now handle layered env loading, model/cache
  preflight checks, sidecar HTTP clients, device resolution, and startup logging
- the local CLI fails early when required cached models or dependencies are
  missing, instead of failing deep inside a model step
- production security is stricter: `API_KEY` is required when auth is enabled,
  CVAT webhooks require `CVAT_WEBHOOK_SECRET`, and rate-limiting state is bounded
- local-run analytics writes `analysis_summary.json` and emits coverage,
  degradation, mapping, tracking, training, and artifact-quality diagnostics
- the architecture now explicitly includes:
  - semantic environment graph generation
  - Gemma-directed tracking
  - UniDriveVLA expert analysis
  - resource-aware `.env` generation
  - realtime bridge runtimes for pose and occupancy sidecars
  - coop MQTT/RTSP site monitoring

## What To Inspect After A Real Run

A strong first pass is to inspect artifacts in this order:

1. `scene_captions.md`
2. `asr_subtitles.md`
3. `multimodal_features.md`
4. `detailed_captions.md`
5. `unidrive_analysis.md`
6. `comparison.md`
7. `multi_model_comparison.md`
8. `3d_map/`
9. `agentic_flow.md`

That order mirrors the move from raw evidence to higher-level reasoning.

If Gemma-directed tracking is enabled, add:

10. `gemma_tracking_summary.md`
11. `gemma_tracking_results.json`

If semantic graphing is enabled, add:

12. `3d_map/semantic_environment_graph.json`
13. `3d_map/semantic_environment_graph.md`

## Practical Study Rules

- Study outputs before internals.
- Compare neighboring steps, not isolated steps.
- Separate representation problems from reasoning problems.
- Separate “this step exists in the repo” from “this step is currently enabled in my run”.
- Keep notes on inputs, outputs, and failure modes for each step.
- When multiple modalities disagree, treat synchronization and calibration as suspects before blaming the model.

## Human Route Through The Repo

If you want a practical code-reading order, use this:

1. `src/ssv_vdp/pipeline/runner.py`
2. `src/ssv_vdp/steps/common.py`
3. `src/selfsuvis/pipeline/workflows/indexer.py`
4. one deep-dive file from `docs/learning_path/`
5. the specific implementation module for the stage you care about

That keeps you moving from orchestration to evidence flow to specialized internals.

## Where To Go Next

- For Steps 1-8: [02_perception_core_steps_01_08.md](02_perception_core_steps_01_08.md)
- For Steps 9-20: [04_sensor_steps_09_20.md](04_sensor_steps_09_20.md)
- For Steps 21-27: [05_tracking_mapping_steps_21_27.md](05_tracking_mapping_steps_21_27.md)
- For Steps 28-35: [06_adaptation_eval_steps_28_35.md](06_adaptation_eval_steps_28_35.md)
- For context accumulation: [07_agentic_knowledge_flow.md](07_agentic_knowledge_flow.md)
- For the core technologies behind the latest implementation: [17_essential_technology_stack.md](17_essential_technology_stack.md)

---

## Learning Resources — Foundations

This pipeline sits at the intersection of computer vision, NLP, robotics, and systems engineering.
The resources below are organized basics → deep dive. Sections map to the five layers of the mental model.

---

### Layer 1 — Representation: how machines see

**Basics**
- Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016). Chapters 9 (CNNs) and 15 (representation learning) are the conceptual base for every embedding step in this pipeline. Freely available at [deeplearningbook.org](https://www.deeplearningbook.org).
- Zhang et al., *Dive into Deep Learning* — interactive textbook with runnable notebooks. Chapter 8 (RNNs) and Chapter 11 (attention) are directly relevant to temporal models. Freely at [d2l.ai](https://d2l.ai).

**Deep dive**
- Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (2020). Foundational paper for ViT, which underpins CLIP, DINOv2, Florence-2, and every other large vision model in this pipeline. [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- Vaswani et al., "Attention Is All You Need" (2017). The architecture underlying every transformer in this pipeline. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

### Layer 2 — System design: multimodal pipelines at scale

**Basics**
- Szeliski, *Computer Vision: Algorithms and Applications* (2nd ed., 2022). Covers feature extraction, matching, SfM, and dense reconstruction — the substrate for Steps 1, 27. Freely at [szeliski.org/Book](https://szeliski.org/Book).
- Prince, *Understanding Deep Learning* (2023). Concise, mathematically precise treatment of modern architectures. Freely at [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook).

**Deep dive**
- Bommasani et al., "On the Opportunities and Risks of Foundation Models" (Stanford HAI, 2021). Maps the landscape of large pre-trained models and their downstream use — directly applicable to every step that loads a pre-trained backbone. [arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258)

---

### Layer 3 — Robotics context: why spatial memory matters

**Basics**
- Thrun, Burgard & Fox, *Probabilistic Robotics* (MIT Press, 2005). Chapters 2-4 (probability, Kalman filter, particle filter). The conceptual grounding for sensor fusion and pose estimation.
- Barfoot, *State Estimation for Robotics* (Cambridge, 2017). Rigorous treatment of SE(3) pose representation, EKF, and batch nonlinear least squares. Directly relevant to Steps 15-16 (IMU, GPS) and pycolmap (Step 27).

**Deep dive**
- Cadena et al., "Past, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perception Age" (IEEE TRO, 2016). Survey of SLAM up to the deep-learning era — gives context for why pycolmap does what it does. [arxiv.org/abs/1606.05830](https://arxiv.org/abs/1606.05830)

---

### Layer 4 — Self-improvement loop: active learning and continual adaptation

**Basics**
- Settles, "Active Learning Literature Survey" (2009). The authoritative introduction to uncertainty sampling, query-by-committee, and information-theoretic selection — the conceptual basis for `al_tag`. Available at [burrsettles.com/pub/settles.activelearning.pdf](http://burrsettles.com/pub/settles.activelearning.pdf).

**Deep dive**
- Ren et al., "A Survey of Deep Active Learning" (2021). Covers neural-network-specific active learning strategies including learned uncertainty, core-set selection, and BALD. [arxiv.org/abs/2009.00236](https://arxiv.org/abs/2009.00236)

---

### HuggingFace ecosystem entry point

The pipeline's Python models (CLIP, DINOv2, Florence-2, Whisper, SAM, Qwen) all load via HuggingFace Transformers or HuggingFace Hub. The following pages are the authoritative API references:
- Transformers library overview: [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- Model Hub (search, filter by task): [huggingface.co/models](https://huggingface.co/models)
- Datasets (for training data): [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
- Optimum (ONNX export, quantization): [huggingface.co/docs/optimum](https://huggingface.co/docs/optimum)
