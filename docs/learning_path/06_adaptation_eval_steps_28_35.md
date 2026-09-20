# Adaptation, Evaluation, And Audit: Steps 28-36

This phase asks the practical engineering questions:

- Can the model adapt to mission-specific data without labels?
- Can it be compressed for edge deployment?
- Did it actually improve, or just change?
- Can a human audit and trust the result?

The key discipline in this phase is **separating adaptation from evaluation**.
You cannot evaluate improvement with the same process that produced the improvement.
Steps 28-31 are the adapt-then-measure loop.
Steps 32-35 are the human-facing outputs of that loop.

Note: these numbers are conceptual study buckets, not the current literal top-level
step numbers in `src/ssv_vdp/pipeline/runner.py`. The current local
runner groups this phase into fewer top-level runtime steps.

---

<a id="step-28-ssl-dino-fine-tuning"></a>
## Step 28. SSL DINO fine-tuning

**What it does:**
Run self-supervised fine-tuning of the DINOv3 backbone on the mission frames extracted in this run.
No labels are required: DINO uses self-distillation between an online (student) and exponential-moving-average (teacher) network.
The result is a backbone whose representations are better calibrated to this specific mission's visual distribution.

**Why it matters:**
DINOv3 pretrained on internet images may encode features useful for general photography but not for specific mission content.
A mission with industrial equipment, military vehicles, or unusual terrain is far from the DINO pretraining distribution.
Self-supervised fine-tuning narrows this distribution gap using the mission data itself, without requiring any human labels.
Even 20-50 training epochs on a single mission's frames often produces measurable improvement in retrieval recall.

**RSSM-guided frame selection for SSL:**
Step 23 computes a per-frame RSSM temporal surprise score (DreamerV3-inspired) for each frame.
The SSL fine-tuning contrastive loss improves most when the positive pairs are diverse — temporally-novel frames carry richer learning signal than repetitive background frames.
Frames tagged `needs_annotation` (high `al_score` combining RSSM surprise + DINO distance + caption confidence) are the best candidates for contrastive pairs in the "temporal" augmentation approach.
This creates a virtuous cycle: RSSM → better AL tags → better SSL training data → better fine-tuned backbone → better edge models (hydrated ONNX exports at Step 30).

**Implementation:**
- [`ssv_vdp/steps/ssl.py`](../../packages/ss-fusion/src/ssv_vdp/steps/ssl.py)
- [`pipeline/training/ssl.py`](../../packages/ss-fusion/src/selfsuvis/pipeline/training/ssl.py)

**Key concepts:**

*DINO (Distillation with No labels):*
DINO trains a student network to produce similar embeddings to a teacher network.
The teacher is an exponential moving average (EMA) of the student's weights — it moves slowly and acts as a stable target.
Both networks see the same image through different random augmentations (crops, flips, color jitter).
The student is forced to match the teacher's output despite seeing a different view.
This encourages the student to learn augmentation-invariant features: the essence of the object, not the specific cropping.

*SSL gate (`_SSL_GATE_MAX_LOSS = 10.0`):*
If the training loss never drops below 10.0, the fine-tuning failed.
In that case, the pipeline skips the downstream distillation and export steps and uses the baseline model instead.
This prevents a failed training run from producing a worse model than the baseline.
The gate is logged clearly; inspect the loss sparkline in the run output.

*Loss sparkline:*
The step logs an ASCII sparkline of the training loss curve using Unicode block characters (▁▂▃▄▅▆▇█).
A monotonically decreasing curve is expected.
A flat curve means the optimizer is stuck. An oscillating curve means learning rate is too high.

*Augmentation strategy:*
Mission aerial footage differs from internet photos in: scale, viewpoint (nadir), color distribution, and object size.
The default augmentations (random crop, flip, color jitter, Gaussian blur) are conservative.
For strong domain shift, custom augmentations aligned to the mission type (e.g., scale-preserving crops for aerial data) can help.

**Output artifact:**
Fine-tuned DINO checkpoint: `dino_finetuned.pt` in the video output directory.
Training loss curve: per-epoch loss values and ASCII sparkline.
Loss analysis: `{first_loss, best_loss, best_epoch, drop_pct, converged}`.

**Human focus:**
- Understand why self-supervised learning works without labels: augmentation-invariance is the signal.
- Learn to read the loss sparkline: flat = stuck, steady decrease = converging, spike = too high LR.
- Know when fine-tuning helps: missions with rare domain-specific content.
- Know when it does not help: missions that closely match the pretraining distribution (ordinary street scenes).
- Understand the SSL gate: it exists to protect downstream steps from a failed run.

**Common failure modes:**
- Too few training frames (< 50) → the model overfits to a handful of examples; embeddings collapse.
- Learning rate too high → loss oscillates or diverges; SSL gate triggers.
- GPU OOM during fine-tuning → reduce batch size in `FinetuneConfig`; check VRAM before training.
- All frames are visually identical (static camera) → augmentation crops look too similar; model learns nothing.

---

<a id="step-29-knowledge-distillation"></a>
## Step 29. Knowledge distillation

**What it does:**
Train a smaller student network to reproduce the output distribution of the fine-tuned DINO teacher.
The student sees the mission frames and is trained to match the teacher's embeddings using KL divergence or MSE loss on the embedding vectors.
The result is a compact model that retains the teacher's mission-specific knowledge.

**Why it matters:**
The fine-tuned DINO teacher is a large model (ViT-B/16 or ViT-L/14).
It may not be deployable on edge hardware with strict memory and latency budgets.
Distillation transfers the teacher's learned structure into a smaller architecture:
- ViT-B/16 → MobileViT, EfficientViT, or a 4-layer ViT.
- Speed gain: 3-10x inference speedup.
- Size gain: 5-20x smaller model file.
The student does not need raw pixels from the mission: it learns from the teacher's output on those pixels.

**Implementation:**
- [`ssv_vdp/steps/distill.py`](../../packages/ss-fusion/src/ssv_vdp/steps/distill.py)
- [`pipeline/training/distill.py`](../../packages/ss-fusion/src/selfsuvis/pipeline/training/distill.py)

**Key concepts:**

*Soft targets vs hard targets:*
Hard targets: one-hot class labels. The student learns to classify.
Soft targets: the teacher's full output distribution (embedding vector). The student learns the geometric structure.
Soft targets carry much more information: the teacher expresses "this frame is 70% similar to class A and 30% similar to class B" rather than just "class A."

*Temperature scaling:*
In classification distillation, a temperature parameter T softens the teacher's probability distribution.
High T: all classes get more equal weight; student learns relative similarities.
Low T: distribution is peaked; student sees the hard winner.
For embedding distillation (used here), temperature scaling is applied differently: to the cosine similarity kernel.

*Teacher-student capacity gap:*
If the student is too small relative to the teacher, it cannot fit the teacher's knowledge.
The practical rule: student should be at least 1/4 the parameter count of the teacher for good distillation.

*RSSM → better teacher quality:*
The RSSM surprise-guided AL scoring (Step 23) ensures the SSL teacher (Step 28) was fine-tuned on the most informative frames.
A teacher trained on temporally-diverse, high-surprise frames transfers richer mission-specific knowledge to the student.
The hydrated ONNX model exported at Step 30 is therefore better calibrated to the mission's actual visual distribution.

**Output artifact:**
Student model checkpoint: `student_model.pt` in the video output directory.
Distillation loss curve: per-epoch loss values.
`{teacher_dim, student_dim, distill_loss_final}`.

**Human focus:**
- Understand the "dark knowledge" idea: teacher soft outputs encode more information than hard labels.
- Learn the practical tradeoff: smaller student = faster inference = higher distillation loss.
- Know that distillation always involves some information loss; the student approximates the teacher, it does not copy it.
- Understand when distillation is worth doing: only when deployment constraints exist. If you have a GPU, just use the teacher.

**Common failure modes:**
- Student architecture is too small → distillation loss is stuck high; student cannot represent teacher's distribution.
- Teacher was poorly fine-tuned (high SSL gate loss) → distillation transfers a bad representation; student is worse than the baseline.
- Training on too few frames → student overfits to a handful of teacher outputs; generalizes poorly.

---

<a id="step-30-drone-detection-edge-training"></a>
## Step 30. Drone detection edge training

**What it does:**
Train a YOLOv8n drone detector using a public dataset (`lgrzybowski/seraphim-drone-detection-dataset`) plus up to 80 mission frames injected as hard negatives.
Export two edge-ready models:
- `drone_yolo8n_a76.onnx` — fp32 ONNX for Arm Cortex-A76 (Raspberry Pi 5, Jetson Orin A76 cores, onnxruntime CPU)
- `drone_yolo8n_rv1106_int8.onnx` — int8 ONNX for Rockchip RV1106G3 (Luckfox Pico / IPC-AI, onnxruntime on Cortex-A7)
- `drone_yolo8n_rv1106.rknn` — (optional) RKNN NPU model for the RV1106G3, generated when `rknn-toolkit2` is installed

**Why it matters:**
Deploying object detection to constrained edge hardware requires a different engineering workflow than cloud inference.
YOLOv8n (nano, ~3 MB) fits within the RV1106G3's NPU budget and runs at 8-15 ms per frame — fast enough for real-time airspace monitoring.
The int8 quantization step (via `onnxruntime.quantization.quantize_dynamic`) reduces model size by ~4× with minimal accuracy degradation when the calibration set is well-chosen.
Mission frames injected as hard negatives directly counteract the primary failure mode of drone detectors: false positives on sky, buildings, and foliage from the deployment environment.

**Implementation:**
- [`ssv_vdp/steps/drone_detection.py`](../../packages/ss-fusion/src/ssv_vdp/steps/drone_detection.py)
- Experimental standalone student-model helper: [`pipeline/training/drone_detector.py`](../../packages/ss-fusion/src/selfsuvis/pipeline/training/drone_detector.py)
- Operational runbook: [`docs/runbooks/drone-detection.md`](../runbooks/drone-detection.md)

**Important scope note:**
Step 30 currently executes the YOLOv8n workflow in `steps_drone_detection.py`.
The newer `pipeline/training/drone_detector.py` module is a standalone training/export helper for a custom MobileNetV3-small student detector and is not yet called by the runner.
When you review the code, separate "what the pipeline runs today" from "what exists as a training API for a future migration".

**Key concepts:**

*Hard negative injection:*
A hard negative is an image with no bounding boxes (empty label file) that the detector sees during training.
The training loss penalizes any box the detector fires on a hard negative frame.
Mission frames make ideal hard negatives: they contain the exact sky, terrain, and structure the deployed detector will encounter, ensuring false-positive suppression is calibrated to the deployment environment, not a generic dataset.

*YOLOv8n augmentation strategy:*
Mosaic (p=1.0) — combines four images into one, exposing partial drones at artificial frame edges and forcing the model to localize drones under occlusion.
Copy-paste (p=0.10) — pastes drone crops from positive examples onto hard-negative backgrounds, directly synthesizing the primary failure case (drone over sky/foliage with no other drones present).
Scale ±30% — simulates altitude variation; a drone at 200 m appears the same pixel size as a different drone at 100 m with half the wingspan.

*ONNX int8 dynamic quantization:*
`onnxruntime.quantization.quantize_dynamic()` applies post-training quantization without a calibration dataset.
Weights are quantized to INT8; activations are quantized at runtime.
This is less accurate than static (calibration-based) quantization but requires no labeled calibration set and is sufficient for object detection where box coordinates are float outputs.

*RKNN export:*
The Rockchip NPU on the RV1106G3 requires a `.rknn` binary compiled from ONNX by `rknn-toolkit2`.
The toolkit is optional: if absent, the step produces the int8 ONNX fallback and logs a warning.
The NPU path delivers 8-15 ms; the Cortex-A7 CPU fallback delivers 80-150 ms.

**Output artifacts:**
All outputs land in `.data/local_runs/{video_name}/drone_detection/`:
- `exports/drone_yolo8n_a76.onnx` — fp32 model for Cortex-A76
- `exports/drone_yolo8n_rv1106_int8.onnx` — int8 model for RV1106G3
- `exports/drone_yolo8n_rv1106.rknn` — (optional) RKNN NPU model
- `test_a76.py` — inference test script for Cortex-A76
- `test_rv1106.py` — inference test script for RV1106G3
- `drone_detection_report.md` — training metrics, model sizes, and deployment commands
- Cross-run model advisor updated at `.data/local_runs/model_run_advisor.md`

**Human focus:**
- Read `drone_detection_report.md` and locate: mAP@50, final box loss, fp32 model size, int8 model size.
- Run `test_a76.py` on one frame from the mission to verify the exported model loads and inferences correctly.
- Understand what "hard negative" means by opening `drone_detection/dataset/train/labels/` and finding the empty `.txt` files that correspond to mission frames.
- Understand why the demo uses only `batch_001` (~400 images) and what mAP@50 improvement you would expect from the full 4-batch dataset.
- Read `pipeline/training/drone_detector.py` and identify which public functions it adds (`DroneDetectorConfig`, `run_drone_detection_training`, `export_drone_detector_onnx`, `export_drone_detector_rknn`) and which of them are not yet invoked by Step 30.

**Common failure modes:**

| Symptom | Cause | Fix |
|---|---|---|
| `HuggingFace download failed` | No internet / rate-limit | Pre-populate `_drone_detection_cache/train_images/` manually |
| `YOLOv8n training failed: ultralytics not installed` | Missing dep | `pip install ultralytics` |
| mAP@50 < 0.30 after 5 epochs | Too few images | Download batch_002–004 to the cache directory |
| High false-positive rate on sky | Insufficient hard negatives | Set `_MAX_NEGATIVES = 150` in `steps_drone_detection.py` |
| `rknn-toolkit2 not found` | Normal on x86 machines | Install from Airockchip releases to enable NPU export |

---

### Standalone student-model helper

**Why it matters:** The custom student-model module is the first step toward replacing the generic YOLOv8n demo with a smaller detector tailored for Cortex-A76 and RV1106-class deployment. It adds a dedicated training surface under `pipeline/training/` instead of embedding all logic inside the workflow step, which is the right long-term layering if the project later supports multiple training backends.

**Public API introduced**
- `DroneDetectorConfig` — hyperparameters and edge-target export settings
- `run_drone_detection_training()` — standalone training entrypoint
- `export_drone_detector_onnx()` — ONNX export with quantization fallback
- `export_drone_detector_rknn()` — RKNN conversion helper

**Current limitation:** this helper is not yet integrated into `runner.py` or `steps_drone_detection.py`, so a normal local run still produces YOLOv8n artifacts and metrics. If you are validating behaviour from pipeline outputs, inspect Step 30. If you are reviewing the new training API itself, inspect `pipeline/training/drone_detector.py`.

**Human focus:**
- Trace the call graph from `runner.py` Step 30 and confirm which module actually executes.
- Compare the artifact contract in `steps_drone_detection.py` with the return payload from `run_drone_detection_training()`.
- Check whether the standalone helper reports real evaluation metrics or only training-loss-derived proxies before using it for deployment decisions.

---

<a id="model-run-advisor"></a>
## Model Run Advisor (runner Step 31)

**What it does:**
After all per-video steps complete, the runner calls `write_model_run_advisor()` once for the entire run.
It aggregates analytics from every video's `analysis_summary` dict and emits two artifacts:
- `.data/local_runs/model_run_advisor.json` — machine-readable optimization plan
- `.data/local_runs/model_run_advisor.md` — human-readable report with recommended `.env` updates and rerun command

Three categories of findings are evaluated:
1. **VLM captioning quality** — Qwen parse errors and caption coverage. Low coverage or high parse-error counts indicate the running model is too small for the JSON schema required by the pipeline.
2. **SfM pose recovery** — degraded maps or zero pose coverage. Signals a capture problem (short clip, no parallax, motion blur), not a model problem.
3. **Artifact volume** — artifact density > 4096 MB/min. Flags expensive full-recipe runs during a hyperparameter-tuning session.

The advisor also reads the drone detection summary (Step 30) and appends an edge deployment profile: mAP@50, which export files were generated, and whether `rknn-toolkit2` was available.

**Why it matters:**
Without a cross-run advisor, diagnosing a degraded run requires reading raw log files across multiple steps.
The advisor collapses that work into a single document with a concrete recommended `.env` block and a ready-to-paste rerun command.
The model recommendations are hardware-aware: they read actual VRAM and RAM from the run context, not static defaults.

**Implementation:**
- [`ssv_vdp/steps/model_advisor.py`](../../packages/ss-fusion/src/ssv_vdp/steps/model_advisor.py)

**Key concepts:**

*Hardware-aware model recommendation:*
`_recommend_qwen_model()` and `_recommend_reasoning_model()` use actual VRAM and RAM readings to select the largest model tier that fits:
- `qwen2.5vl:7b` when `vram_gb ≥ 12` or `free_vram_gb ≥ 10` or `ram_gb ≥ 48` (can offload to CPU RAM)
- `qwen2.5vl:3b` below those thresholds
- Reasoning model: `deepseek-r1:14b` (≥ 24 GB VRAM), `qwen3:14b` (≥ 12 GB), `qwen3:8b` (RAM ≥ 32 GB), `gemma4:e4b` (fallback)

This means the same pipeline code emits a different recommendation on a workstation with 24 GB VRAM than on a laptop with 8 GB — without any manual configuration.

*Sequential VLLM graph profile:*
The advisor outputs a `sequential_vllm_graph_profile` block that specifies execution order:
1. `gemma_analysis` — scene understanding
2. `qwen_captioning` — detailed structured captioning
3. `unidrive` — VLA planning
4. `reasoning_audit` — final quality audit

The critical constraint is `OLLAMA_MAX_LOADED_MODELS=1`: only one model is in VRAM at a time, so each step gets the full VRAM budget.
Setting `OLLAMA_KEEP_ALIVE=0` evicts the model immediately after inference, freeing headroom for the next one.
This sequential pattern trades latency (models load and unload per step) for correctness: no OOM and no KV-cache interference between models on a 12 GB card.

*Cross-run scope:*
Unlike per-video step outputs (which land in `.data/local_runs/{video_name}/`), the advisor writes to `.data/local_runs/` at the run level and aggregates evidence from all videos in one invocation.
If three videos were processed and two had degraded maps, the `sfm_pose_recovery_degraded` finding covers both.

**Output artifacts:**
- `.data/local_runs/model_run_advisor.json` — full structured output (findings, recommendations, recommended_env_updates, edge_deployment, sequential_vllm_graph_profile, recommended_ollama_pulls, recommended_rerun)
- `.data/local_runs/model_run_advisor.md` — sections: Hardware, Findings, Recommended `.env` Updates, Pull/Serve Models, Recommended Rerun, Rationale, Edge Deployment, Sequential VLLM Graph Profile

**Human focus:**
- Open `model_run_advisor.md` after every run. Read the Findings section first.
- If `qwen_structured_captioning_degraded` is present: apply the `recommended_env_updates` block — upgrade QWEN_MODEL and enable UNIDRIVE.
- If `sfm_pose_recovery_degraded` is present: read the `capture_guidance` bullets. A larger model does not fix zero SfM poses; the capture geometry must change.
- Locate the `Sequential VLLM Graph Profile` table and verify `OLLAMA_MAX_LOADED_MODELS=1` is set in your environment before re-running with multiple VLM steps enabled.
- Review the `recommended_rerun` command — it includes the correct flags (`--qwen --unidrive --world-model --rfdetr-model base --drone-detection`) for a full-quality run.

**Common failure modes:**

| Symptom | Cause | Fix |
|---|---|---|
| Advisor recommends `qwen2.5vl:3b` but you have 24 GB VRAM | `free_vram_gb` was near 0.0 at report time (GPU was full during the run) | Check `nvidia-smi` before re-running; pass post-run free VRAM to the advisor |
| `sfm_pose_recovery_degraded` persists after model upgrade | Map degradation is a capture problem, not a model problem | Follow `capture_guidance`: longer clip, lateral motion, higher frame overlap |
| Advisor writes no findings | All metrics were within bounds | Expected for a healthy run; the general recommendation is to keep the current plan |
| `recommended_ollama_pulls` lists fewer models than expected | Two roles mapped to the same model tier | `dict.fromkeys()` deduplicates automatically; this is correct behaviour |

---

<a id="step-31-onnx-export-and-gallery-build"></a>
## Step 31. ONNX export and gallery build

**What it does:**
Export the student model (or the fine-tuned teacher if no student exists) to ONNX format.
Build a reference gallery: embed all mission frames with the ONNX model and store the embeddings for fast search.

**Why it matters:**
A model that cannot be exported and served is still a research artifact.
ONNX (Open Neural Network Exchange) is the standard intermediate format for deploying PyTorch models on:
- Edge hardware with ONNX Runtime
- TensorRT for GPU-optimized inference
- Mobile devices via CoreML or Android NNAPI
- Any runtime that supports ONNX (most modern ML serving stacks)

The gallery is the pre-computed embedding index that allows real-time search without re-embedding every frame on each query.

**Implementation:**
- [`ssv_vdp/steps/distill.py`](../../packages/ss-fusion/src/ssv_vdp/steps/distill.py)

**Key concepts:**

*ONNX export process:*
1. PyTorch model is traced or scripted with a dummy input tensor.
2. The traced computation graph is exported to an `.onnx` file.
3. Dynamic axes are specified so the model accepts variable batch sizes.
4. The ONNX model is validated by loading it with onnxruntime and comparing outputs to the PyTorch model.

*Dynamic vs static axes:*
A static-axes ONNX model only accepts a fixed batch size and input resolution.
A dynamic-axes ONNX model accepts any batch size but may be slightly less optimizable.
The pipeline exports with dynamic batch axes; input resolution is fixed to the training resolution.

*Gallery build:*
The gallery is a `.npy` array of shape `(n_frames, embedding_dim)` produced by running the exported ONNX model over all mission frames.
Alongside it: a metadata JSON that maps each gallery row to a frame path and timestamp.
Search is then: embed query → cosine similarity against gallery → return top-K.

**Output artifact:**
`model_export.onnx` in the video output directory.
`gallery_embeddings.npy` and `gallery_metadata.json`.

**Human focus:**
- Understand why ONNX is the target: it separates model training (PyTorch) from model serving (any runtime).
- Learn the ONNX tracing vs scripting distinction: tracing captures one execution path; scripting handles control flow.
- Know the gallery as a pre-indexed search space: the trade-off is that adding a new frame requires re-running the gallery build.

**Common failure modes:**
- Dynamic control flow in the model (if statements depending on tensor values) → tracing fails; scripting required.
- Input normalization not included in the model → ONNX model requires the caller to pre-normalize; easy to forget.
- ONNX opset version mismatch → exported model uses ops not supported by the target runtime.
- Gallery size too large for RAM → `.npy` file is too big to load into memory; need chunked search or Qdrant.

---

<a id="step-32-fine-tuned-search-test"></a>
## Step 32. Fine-tuned search test

**What it does:**
Re-run the same test queries from Step 26 against the fine-tuned model's gallery.
Record top-K results and compute P@K, R@K.
Produce a side-by-side comparison: baseline (Step 26) vs fine-tuned (Step 31).

**Why it matters:**
Training a model does not mean it improved.
The fine-tuned model might:
- Improve retrieval for in-distribution mission content (the goal).
- Degrade retrieval for general content (acceptable trade-off if mission-specific is more important).
- Make no meaningful difference (fine-tuning was unnecessary for this mission type).
- Get worse overall (the SSL gate should have caught this; if it did not, the evaluation will).

Step 32 is the only honest measure of whether Steps 28-31 were worth doing.

**Implementation:**
- [`ssv_vdp/steps/embed.py`](../../packages/ss-fusion/src/ssv_vdp/steps/embed.py)

**Key concepts:**

*What counts as improvement:*
- Metric improvement: P@K or R@K increases for the target queries.
- Subjective improvement: retrieved frames are more relevant even if the metric is similar (edge cases near the threshold).
- Distribution shift: fine-tuning changed what the model finds similar, even if the metric looks the same.

*When the metric can lie:*
- P@K is sensitive to K: a model that ranks the one correct result at position 3 scores P@5=0.2, same as one that ranks 1 relevant result at position 5.
- R@K is sensitive to how many "correct" frames exist: if you only manually labeled 3 relevant frames but 30 exist, R@K is underestimated.
- Always combine metric comparison with human inspection of the results.

**Output artifact:**
Search comparison report: per-query table of baseline vs fine-tuned top-K results with scores and delta.
Summary: `{baseline_p_at_k, finetuned_p_at_k, delta, improved_queries, degraded_queries}`.

**Human focus:**
- Run the comparison yourself, not just read the summary table.
- Pick two queries: one where fine-tuning clearly helped and one where it clearly did not.
- Understand why mission-specific adaptation might degrade general queries: the model is less general after fine-tuning.

**Common failure modes:**
- Test queries were not designed before fine-tuning → the queries may now be biased toward what the fine-tuning emphasized.
- Only a few test queries → results are not statistically meaningful.
- No baseline recorded from Step 26 → comparison is impossible; rerun Step 26 with the baseline model first.

---

<a id="step-33-model-comparison-and-video-description"></a>
## Step 33. Model comparison and video description

**What it does:**
Produce a side-by-side comparison of baseline vs fine-tuned model behavior across a sample of frames.
Derive a clip-level video description: a readable natural-language summary of the full mission based on all accumulated evidence.

**Why it matters:**
Two separate outputs:
1. The model comparison gives a quantitative and qualitative picture of what changed between baseline and fine-tuned.
2. The video description is the human-facing mission summary: what happened, what was observed, what matters.

The video description is built from the union of all evidence in `VideoKnowledge`:
Gemma scene analysis, Florence captions, ASR text, Qwen structured observations, and UniDriveVLA expert analysis.
This is where the pipeline becomes a product rather than a tool.

**Implementation:**
- [`ssv_vdp/pipeline/runner.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/runner.py)
- [`ssv_vdp/steps/report.py`](../../packages/ss-fusion/src/ssv_vdp/steps/report.py)

**Key concepts:**

*Comparison dimensions:*
- Embedding distance: for the same frame, how far are the baseline and fine-tuned embeddings from each other?
  Large distance = adaptation changed the representation significantly.
  Near-zero distance = adaptation had no effect.
- Retrieval rank shift: did the same query return different neighbors?
  Which improved, which degraded?
- Visual inspection: for the same query, do the fine-tuned top-5 look more relevant than the baseline top-5?

*Video description quality:*
The video description combines evidence across all modalities and all steps.
Its quality is limited by the weakest upstream step: if Gemma failed, scene context is missing; if ASR was noisy, the transcript adds noise.
The description should be read critically: check which claims are supported by which evidence.

**Output artifact:**
`comparison.md` in the video output directory: side-by-side baseline vs fine-tuned retrieval results.
`video_description.md`: the full mission narrative summary.

**Human focus:**
- Read the video description and evaluate whether it matches your understanding of the mission.
- Identify which claims in the description are well-supported by evidence vs speculative.
- Check the comparison table: find the query with the largest rank improvement and the query with the largest regression.

**Common failure modes:**
- SSL fine-tuning failed → comparison shows no difference; fine-tuned results are identical to baseline.
- Video description is too generic ("a vehicle moved across a road") → domain-specific evidence from UniDriveVLA or sensors is absent.
- Description includes hallucinated claims from Qwen that no other modality supports.

---

<a id="step-34-multi-model-comparison"></a>
## Step 34. Multi-model comparison

**What it does:**
For a sample of key frames, collect outputs from multiple multimodal analyzers:
- Florence (caption)
- Gemma (scene analysis)
- Qwen (detailed structured observation)
- UniDriveVLA (domain expert analysis)

Identify where they agree and where they disagree.
Compute a cross-model agreement score for each frame.

**Why it matters:**
Agreement across independent models increases confidence in an observation.
Disagreement is often more informative than agreement:
- If Qwen says "empty road" and Florence says "a vehicle approaching an intersection", one of them is wrong.
- If Gemma says "industrial terrain" and UniDriveVLA says "highway interchange", the scene is ambiguous and warrants human inspection.
- Systematic disagreement between Florence and Qwen across many frames may indicate that one model's prompting is wrong for this domain.

**Implementation:**
- [`ssv_vdp/pipeline/runner.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/runner.py)

**Key concepts:**

*Sources of disagreement:*
1. Different input: Florence sees a single frame; Qwen sees the frame plus context. They should disagree slightly by design.
2. Different training distribution: Florence and Qwen were trained on different corpora; they use different feature biases.
3. Genuine ambiguity: the scene truly has multiple valid interpretations.
4. Error: one model is simply wrong.

*Agreement metric:*
The pipeline computes a Jaccard-style token overlap between captions from different models.
High overlap = agreement. Low overlap = disagreement.
This is a proxy, not a semantic agreement measure: two models can disagree in meaning with high word overlap, or agree in meaning with low overlap.

**Output artifact:**
`multi_model_comparison.md` in the video output directory: per-frame side-by-side outputs with agreement score.
Summary: list of frames with lowest agreement (most uncertain or ambiguous).

**Human focus:**
- Pick the three frames with the lowest cross-model agreement and inspect them manually.
- Determine whether disagreement is due to genuine ambiguity, different input, or outright error.
- Use the disagreement list as an attention signal: these frames are where the pipeline is least confident.

**Common failure modes:**
- Only one model is available → "comparison" has nothing to compare; the step produces a single-model report.
- All models agree on a wrong interpretation → ensemble agreement gives false confidence; always inspect high-confidence failures.
- Agreement metric treats paraphrase as disagreement ("vehicle" vs "car") → score underestimates true agreement.

---

<a id="step-35-video-synthesis"></a>
## Step 35. Video synthesis

**What it does:**
Collect all intermediate artifacts from the full pipeline run and produce a single structured HTML report:
- Mission metadata (timestamp, duration, GPS bounding box)
- Scene summary from Gemma and Florence
- Key frame gallery with captions, detection overlays, and depth labels
- ASR transcript timeline
- Multi-model comparison table for selected frames
- Model adaptation summary (baseline vs fine-tuned metrics)
- Active learning tags: which frames need human annotation?
- Change detection summary if this is a repeat mission

**Why it matters:**
This is where the pipeline becomes a human-facing product.
All previous steps produce machine-readable intermediate artifacts.
Step 34 assembles them into something a person can read in 10-15 minutes and use to make operational decisions.

**Implementation:**
- [`ssv_vdp/pipeline/runner.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/runner.py)
- [`ssv_vdp/steps/report.py`](../../packages/ss-fusion/src/ssv_vdp/steps/report.py)

**Key concepts:**

*What belongs in the synthesis:*
Include: every claim that is supported by at least two independent sources.
Flag: claims supported by only one source.
Exclude: model outputs that directly contradicted another model or failed quality checks.

*Active learning tags:*
Frames tagged `needs_annotation` (high `active_learning_score`) are surfaced in the report as priority labeling candidates.
The score formula depends on whether the RSSM ran in Step 23:

- **With RSSM** (default, `DREAMER_ENABLED=true`): `0.35 × DINOv3_dist + 0.25 × (1 - caption_confidence) + 0.40 × rssm_surprise`
- **Without RSSM**: `0.60 × DINOv3_dist + 0.40 × (1 - caption_confidence)`

High score = the model was uncertain or temporally surprised by this frame; human labeling here will improve future fine-tuning the most.

*Change detection summary:*
If the same GPS area was covered in a prior mission, the change detection table shows which objects or regions changed.
This is the key cross-mission feature: it makes the pipeline a persistent monitoring system, not a one-time analysis.

**Output artifact:**
`reports/{mission_id}/summary.html` — the full mission report rendered as HTML.
Linked: frame gallery images, ASR transcript, GPS track overlay.

**Human focus:**
- Read one full report end-to-end on a real mission and identify one error or gap in each major section.
- Compare the active learning tags to your own sense of which frames are most unusual.
- Evaluate the report as a product: could an operator make a decision based on this without reading the raw artifacts?

**Common failure modes:**
- Some intermediate artifacts are missing (step failed silently) → report has empty sections with no explanation.
- Report is too long → operator skips most of it; the synthesis failed at its own purpose.
- HTML rendering broken due to missing frame paths or relative path errors.

---

<a id="step-36-agentic-flow-audit"></a>
## Step 36. Agentic flow audit

**What it does:**
Produce a provenance-style audit document that traces how context moved through the pipeline:
- Which step produced which artifact?
- Which artifact was consumed by which later step?
- At which steps did context propagation break (missing input, API failure, silent skip)?
- Which claims in the final report are traceable back to raw sensor data?

**Why it matters:**
This is the inspection layer for debugging, trust calibration, and failure analysis.

Without an audit trail:
- You cannot determine which step caused a wrong output.
- You cannot tell whether a claim in the report is well-evidenced or hallucinated.
- You cannot detect silent failures (a step that appeared to run but produced garbage).
- You cannot reproduce the analysis with different settings.

With an audit trail:
- Every claim in the synthesis traces back to at least one source.
- Every step is timestamped; failures have clear upstream causes.
- An engineer can diagnose failures by following the provenance chain backward.

**Implementation:**
- [`ssv_vdp/pipeline/runner.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/runner.py)
- [`docs/pipeline.md`](../reference/pipeline.md)

**Key concepts:**

*Provenance:*
The record of where a piece of information came from, which transformations it went through, and which later outputs it influenced.
In this pipeline: a Qwen claim in the synthesis traces back to → Qwen input context → which came from VideoKnowledge → which was populated by Florence, ASR, OCR, depth, and detection steps.

*Silent failure:*
A step that returns a default or empty result without raising an exception.
Example: Gemma API is unavailable; the step logs a warning and returns `{}`.
`VideoKnowledge.scene_type` stays empty; all later domain hints are blank.
The synthesis does not know this happened; it just receives an empty domain hint and produces weaker output.
The audit catches this: it lists which steps produced non-empty output and which produced default fallbacks.

*Context contamination:*
Wrong context propagated forward through `VideoKnowledge` corrupts downstream steps.
Example: Gemma misclassifies the scene as "coastal" instead of "desert"; this wrong domain hint propagates into every Qwen call.
The audit traces which Gemma call produced the wrong classification and when it was injected.

**Output artifact:**
`agentic_flow.md` in the video output directory:
- Step-by-step timing table: step name, start time, elapsed, status (success / skipped / failed).
- Context propagation graph: which `VideoKnowledge` fields were populated by which step.
- Silent failure log: steps that returned empty output or default fallbacks.
- Claim provenance: for each section of the synthesis, which steps contributed evidence.

**Human focus:**
- Read `agentic_flow.md` first after a new run, before reading the synthesis.
- Identify any step with status `skipped` or `failed` and determine what downstream effects it had.
- Trace one claim from the synthesis backward to its raw source. Can you do it?
- Know the difference between a hard failure (exception logged) and a silent failure (empty output, no exception).

**Common failure modes:**
- Audit itself fails to run (runner crash in final cleanup) → no provenance exists; debugging is manual.
- Step timing is wrong because CUDA events are asynchronous → GPU steps appear instantaneous; use CPU timing for wall-clock.
- Silent failures not logged → audit is incomplete; some gaps are invisible.
- Context contamination is undetected → audit shows "success" for every step but the synthesis is wrong.

---

## What A Human Should Learn In This Phase

Learn to separate these four questions and ask them independently:

1. **Did the representation get better?**
   Compare P@K from Steps 26 and 32. Not just the numbers, but which queries improved.

2. **Did the smaller model preserve the right structure?**
   Compare the teacher's embeddings to the student's embeddings on held-out frames.
   Low cosine distance means the student learned well; high distance means it lost something.

3. **Did the outputs become more useful for humans?**
   Read the synthesis report. Could an operator use it? What is missing?
   This is a different question from (1) and (2).

4. **Can I trace how the final conclusion was formed?**
   Use the agentic flow audit. Can you follow the provenance chain from any synthesis claim back to raw data?

A system can optimize one of these and fail the others.
Good engineering requires all four.

## Related Docs

- [Tracking and mapping: Steps 21-27](05_tracking_mapping_steps_21_27.md)
- [Agentic knowledge flow](07_agentic_knowledge_flow.md)
- [Runtime and study guide](01_runtime_and_study_guide.md)
- [Pipeline architecture](../reference/pipeline.md)
- [Drone detection runbook](../runbooks/drone-detection.md)

---

## Learning Resources — Adaptation, Distillation, and Evaluation (Steps 28-36)

The central theme of this phase is the feedback loop: observation → representation → improvement. Resources are ordered basics → deep dive.

---

### Step 28 — SSL DINOv3 Fine-Tuning

**Why it matters:** Fine-tuning without labels is the key property that allows this pipeline to adapt to any new domain (underwater, Arctic, industrial) with zero human annotation cost. The RSSM-guided frame selection means the training set consists of the most temporally informative frames, not random samples.

**Basics — Self-supervised learning**
- Ericsson et al., "Self-Supervised Representation Learning: Introduction, Advances, and Challenges" (IEEE Signal Processing Magazine, 2022). The most accessible survey of the SSL landscape: contrastive methods (SimCLR, MoCo), self-distillation (BYOL, DINO), and masked image modelling (MAE). Read this before the individual papers. [arxiv.org/abs/2110.09327](https://arxiv.org/abs/2110.09327)

**Core papers — DINO and DINOv2**
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO, 2021). The student-teacher EMA architecture used in `pipeline/training/ssl.py`. Section 3 (self-distillation with no labels) and the appendix (multi-crop augmentation details) are required reading before modifying any SSL hyperparameter. [arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (Meta AI, 2023). Section 3.1 (curated data pipeline) is the key difference from DINO v1 — the importance of data curation for SSL quality has direct implications for what the pipeline's RSSM-selected frames provide. [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)

**Context — related self-supervised methods worth knowing**
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR, 2020). The contrastive learning baseline. Understanding SimCLR's collapse problem (without stop-gradient or momentum) makes DINO's design choices principled. [arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE, 2021). Masked image modelling — the alternative SSL paradigm to self-distillation. MAE trains faster on large datasets; DINO produces better dense features at small dataset sizes like a single mission. [arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)

**HuggingFace**
- HuggingFace Trainer documentation: [huggingface.co/docs/transformers/main_classes/trainer](https://huggingface.co/docs/transformers/main_classes/trainer) — the API used by `steps_ssl.py`.

---

### Step 29 — Knowledge Distillation

**Why it matters:** Distillation is how the pipeline converts a large fine-tuned teacher into a small student deployable on edge hardware. Soft targets — the teacher's full probability distribution over embeddings — carry "dark knowledge" about inter-class similarity that hard labels discard.

**Basics**
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015). The original paper. Five pages. The key insight is that the teacher's softmax output at temperature T > 1 reveals relative similarity between incorrect classes — information that a hard label (argmax) destroys. Read this before any distillation framework. [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

**Survey**
- Gou et al., "Knowledge Distillation: A Survey" (IJCV, 2021). Classifies 40+ distillation methods by what is distilled (logits, features, relations, structure) and how (offline, online, self-distillation). Section 4 (feature-based distillation) is most relevant to the pipeline's embedding-space distillation. [arxiv.org/abs/2006.05525](https://arxiv.org/abs/2006.05525)

**Deep dive**
- Romero et al., "FitNets: Hints for Thin Deep Nets" (2014). Feature-matching distillation — the basis for hint regression used when the student and teacher have mismatched architectures. [arxiv.org/abs/1412.6550](https://arxiv.org/abs/1412.6550)
- Tian et al., "Contrastive Representation Distillation" (CRD, 2019). Distillation as mutual information maximisation — outperforms standard logit-matching for visual representations. [arxiv.org/abs/1910.10699](https://arxiv.org/abs/1910.10699)

---

### Step 30 — Drone Detection Edge Training

**Why it matters:** Edge object detection combines three engineering disciplines at once: small-model training (YOLOv8n), quantization-aware export (ONNX int8), and hardware-specific compilation (RKNN NPU). Hard negative injection from the mission directly addresses the deployment environment's false-positive profile, a practice with measurable impact on precision that generic datasets cannot replicate.

**Basics — Object detection and YOLO**
- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (YOLOv1, 2016). The original YOLO paper — read Sections 2-3 to understand why single-stage detectors trade some accuracy for dramatically higher inference speed. [arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
- Ultralytics YOLOv8 documentation: [docs.ultralytics.com](https://docs.ultralytics.com). The API used in `steps_drone_detection.py`. Focus on the `train()` function signature and augmentation hyperparameters.

**Core reference — Quantization**
- Nagel et al., "A White Paper on Neural Network Quantization" (Qualcomm AI Research, 2021). The most rigorous treatment of INT8 quantization: symmetric vs asymmetric, per-tensor vs per-channel, post-training vs quantization-aware training, calibration methods. Directly relevant to the int8 export for the RV1106G3. [arxiv.org/abs/2106.08295](https://arxiv.org/abs/2106.08295)
- ONNX Runtime quantization documentation: [onnxruntime.ai/docs/performance/quantization.html](https://onnxruntime.ai/docs/performance/quantization.html). Explains `quantize_dynamic` vs `quantize_static` and when each is appropriate.

**Deep dive — Edge deployment**
- Rockchip RV1106G3 NPU documentation: [github.com/airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2). The RKNN toolkit converts ONNX to a hardware-optimized binary for the Rockchip NPU. Read the README and the quantization section before attempting the RKNN export.
- Howard et al., "MobileNets" (2017). The architectural trade-offs that make small models deployable — directly applicable to understanding why YOLOv8n (nano) fits the RV1106G3 budget while larger YOLO variants do not. [arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)

**Runbook**
- [`docs/runbooks/drone-detection.md`](../runbooks/drone-detection.md): complete operational guide covering dataset expansion, hard negative tuning, inference on Cortex-A76 and RV1106G3, RKNN offline conversion, and troubleshooting.

---

### Model Run Advisor

**Why it matters:** The advisor is the feedback loop closer — it converts run-health signals into concrete `.env` changes and a ready-to-paste rerun command. The sequential VLLM graph profile it emits encodes the correct multi-model orchestration strategy for a single-GPU machine: load one model, infer, evict (`KEEP_ALIVE=0`), load the next. Without this, running gemma + qwen + reasoning concurrently on a 12 GB card causes OOM or KV-cache fragmentation that silently degrades output quality.

**Core reference — Ollama concurrency model**
- Ollama documentation on concurrency: [github.com/ollama/ollama](https://github.com/ollama/ollama). The `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_PARALLEL`, and `OLLAMA_KEEP_ALIVE` environment variables control VRAM multiplexing. Read the README concurrency section before adjusting these values — the interactions between them are non-obvious.

**Deep dive — Graph-based step orchestration**
- LangGraph documentation: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph). The graph-based orchestration model that `SELFSUVIS_USE_GRAPH=1` enables. The concepts of nodes, edges, and conditional routing map directly to the advisor's `sequential_vllm_graph_profile` `recommended_order` — each entry is a node; the sequential constraint is an edge ordering.

---

### Step 31 — ONNX Export and Gallery Build

**Why it matters:** ONNX is the portability layer between PyTorch research code and any edge runtime (TensorRT, ONNX Runtime, CoreML, TFLite). Getting the dynamic axes right — so that batch size and image size are not hardcoded — is the difference between an ONNX export that deploys and one that silently fails on non-standard input sizes.

**Basics**
- ONNX specification and operator set: [onnx.ai/onnx/intro](https://onnx.ai/onnx/intro). Start with the operator set reference — every PyTorch op must map to one or more ONNX ops. Ops that don't have ONNX equivalents (or have conditional implementations) are the source of export failures.
- HuggingFace Optimum documentation: [huggingface.co/docs/optimum](https://huggingface.co/docs/optimum). The recommended way to export HuggingFace models to ONNX. Handles dynamic axes, input validation, and numerical equivalence checks automatically.

**Core reference**
- ONNX Runtime documentation: [onnxruntime.ai/docs](https://onnxruntime.ai/docs). The primary inference runtime. Pay attention to execution providers (CPU, CUDA, TensorRT, DirectML) and the InferenceSession API — directly used in the gallery search path.

**Deep dive**
- Han et al., "A Survey on Model Compression and Acceleration for Deep Learning" (2015). Reviews pruning, quantization, and factorization in addition to distillation — maps the full space of model compression that ONNX export enables. [arxiv.org/abs/1710.09282](https://arxiv.org/abs/1710.09282)

---

### Steps 32-34 — Retrieval Evaluation and Model Comparison

**Why it matters:** Precision@K and Recall@K are only honest if the query set was designed before looking at the retrieval results. Post-hoc query design — choosing queries that the fine-tuned model happens to get right — produces metrics that look like improvement but measure nothing.

**Basics — Information retrieval metrics**
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (Cambridge, 2008). Chapter 8 (evaluation in information retrieval): Precision@K, Mean Average Precision (MAP), nDCG. Freely at [nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book).
- Musgrave et al., "A Metric Learning Reality Check" (ECCV, 2020). The paper that empirically showed most reported metric learning improvements vanish under standardized evaluation. Directly relevant to interpreting Precision@K deltas. [arxiv.org/abs/2003.08505](https://arxiv.org/abs/2003.08505)

**Core paper**
- Babenko et al., "Neural Codes for Image Retrieval" (2014). The foundational paper on using deep features for image retrieval — establishes the baseline expectation for how embedding quality maps to retrieval performance. [arxiv.org/abs/1404.1777](https://arxiv.org/abs/1404.1777)

**Deep dive**
- Johnson et al., "Billion-scale Similarity Search with GPUs" (FAISS, 2017). Beyond correctness: the computational tradeoffs in approximate nearest-neighbour search. Understanding HNSW vs IVF vs flat index performance curves is required before tuning Qdrant collection parameters. [arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734)

---

### Steps 35-36 — Synthesis, Reporting, and Agentic Audit

**Why it matters:** The synthesis report is the human-readable output of the entire pipeline. Its quality determines whether an operator can make decisions from it. The audit step is the provenance chain — without it, a hallucinated synthesis claim is indistinguishable from a grounded one.

**Basics — LLM output evaluation**
- Bubeck et al., "Sparks of Artificial General Intelligence: Early Experiments with GPT-4" (Microsoft Research, 2023). Section 2.5 (evaluation methodology) discusses how to distinguish genuine capability from pattern-matched output — directly applicable to evaluating synthesis quality. [arxiv.org/abs/2303.12528](https://arxiv.org/abs/2303.12528)

**Core paper — Provenance and attribution**
- Gao et al., "Enabling Large Language Models to Generate Text with Citations" (ALCE, 2023). Attribution of generated claims to source documents — the theoretical framework for the agentic audit's claim-tracing design. [arxiv.org/abs/2305.14627](https://arxiv.org/abs/2305.14627)

**Deep dive — Agentic systems**
- Weng, "LLM-powered Autonomous Agents" (Lilian Weng's blog, 2023). The clearest system-level description of tool-use, memory, and planning in LLM agents — directly maps to `VideoKnowledge` (memory), the step runner (planning), and the audit step (provenance). [lilianweng.github.io/posts/2023-06-23-agent](https://lilianweng.github.io/posts/2023-06-23-agent)
- Mialon et al., "Augmented Language Models: a Survey" (Meta AI, 2023). Comprehensive survey of tool use, retrieval augmentation, and grounding — the design space the pipeline's agentic flow occupies. [arxiv.org/abs/2302.07842](https://arxiv.org/abs/2302.07842)

---

## Perspective Directions For Self-Supervised Vision

After you understand Steps 28-36 as they exist today, the next useful question is not
"which larger model should I add?" It is:

- what structure is still missing from the representation?

The most promising next directions are:

1. **Temporal SSL**
   Move from frame-only adaptation toward track-aware and clip-aware learning.
   Learn invariance across motion, viewpoint, occlusion, and short temporal gaps.
2. **Cross-modal SSL**
   Use agreement between RGB, depth, thermal, radar, IMU, and audio as a self-supervised signal.
   This is especially valuable when labels are scarce but synchronization exists.
3. **Geometry-aware SSL**
   Use multiview consistency, map consistency, and object permanence as training constraints, not only as downstream evaluation artifacts.
4. **Anomaly-aware SSL**
   Use RSSM surprise, track break statistics, and cross-modal contradiction as sampling signals for adaptation, not only retrieval uncertainty.

The practical human recommendation is:

- study temporal and cross-modal SSL before adding more prompt-heavy reasoning layers

That path improves retrieval, tracking, anomaly detection, and later global-threat and sensor-mesh extensions at the same time.
