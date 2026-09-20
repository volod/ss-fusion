# Agentic Knowledge Flow

The local pipeline is not a bag of isolated model calls.
Later steps reuse earlier evidence through an accumulated context object called `VideoKnowledge`.
Understanding this object is the key to understanding why the pipeline produces different outputs depending on which steps ran and in what order.

---

## Core Idea

Earlier steps produce structured evidence.
That evidence is stored in `VideoKnowledge` via `add_*` methods.
Later steps query `context_for_frame(t_sec)` or `domain_hint()` to receive all prior knowledge relevant to a specific moment.

The main accumulator is implemented in:

- [`ssv_vdp/steps/common.py`](../../packages/ss-fusion/src/ssv_vdp/steps/common.py) — `VideoKnowledge` class

---

## VideoKnowledge: Structure And Lifecycle

```python
class VideoKnowledge:
    video_name: str
    duration_sec: float
    frame_count: int

    # Domain-level knowledge (from Gemma, Step 3)
    scene_type: str  # dominant zero-shot category ("road", "urban", etc.)
    n_transitions: int  # how many scene changes Gemma detected
    n_clusters: int  # how many visually distinct segments
    gemma_mnn_dino: float  # mean nearest-neighbour DINOv3 distance

    # Per-frame evidence (keyed by t_sec)
    _captions: Dict[float, str]  # Florence caption per frame (Step 4)
    _asr: Dict[float, str]  # ASR speech text per timestamp (Step 5)
    _ocr: Dict[float, str]  # OCR visible text per frame (Step 6)
    _depth: Dict[float, Dict]  # depth statistics per frame (Step 7)
    _detections: Dict[float, List[str]]  # detected object labels per frame (Step 8)

    # Temporal structure
    _segments: List[Dict]  # scene segments derived from caption Jaccard overlap

    # Cross-frame entity inventory (from Step 8)
    known_entities: List[str]  # top-15 most frequent detected labels across video

    # Qwen rolling state (Step 24)
    _last_qwen: Dict[str, Any]  # most recent Qwen JSON output
```

The object is created once per video at the start of the run and passed through every step.
Each step either deposits data into it or queries it.

---

## Deposit Methods: What Gets Stored Where

| Step | Method | Data deposited |
|------|---------|---------------|
| 3 (Gemma) | `add_gemma()` | `scene_type`, `n_transitions`, `n_clusters` |
| 4 (Florence) | `add_captions()` | `_captions`, `_segments` |
| 5 (ASR) | `add_asr()` | `_asr` |
| 6 (OCR) | `add_ocr()` | `_ocr` |
| 7 (Depth) | `add_depth()` | `_depth` |
| 8 (Detection) | `add_detections()` | `_detections`, `known_entities` |
| 12 (Qwen) | `update_qwen_state()` | `_last_qwen` |

All per-frame data is keyed by `t_sec` (timestamp in seconds from video start).
Retrieval uses nearest-timestamp lookup within a configurable window (default ±2 seconds).

---

## Query Methods: How Later Steps Read Evidence

### `domain_hint()` → str

Used by Florence (Step 4) to condition its captioning prompt.

Returns a pipe-separated summary like:
```
Dominant scene: road | Known objects: truck, car, person | Visual transitions: 3
```

If Gemma was skipped or failed, this string is empty.
Florence then operates without domain context and produces more generic captions.

### `context_for_frame(t_sec)` → str

Used by Qwen (Step 24) to build its per-frame reasoning prompt.

Assembles all evidence available near `t_sec` into a multi-line string:

```
[Prior scene description]: Two vehicles on a highway visible from above.
[Scene segment 2, 12.3s–18.7s]: Industrial site with large structures.
[Audio context]: "Sector seven, all clear"
[Visible text]: STOP 40
[Depth profile]: near_ratio=0.18  mean=0.65
[Detected objects]: truck, car, traffic-sign
[Prior frame state]: vehicles=2×truck; road=highway; condition=clear
```

Each line is sourced from a different step:
- `[Prior scene description]` → Florence caption, nearest frame within ±2 s.
- `[Scene segment ...]` → Jaccard-based segment analysis from Step 4.
- `[Audio context]` → ASR text within ±2 s window.
- `[Visible text]` → OCR text within ±1 s window.
- `[Depth profile]` → depth statistics, nearest frame within ±2 s.
- `[Detected objects]` → detection labels, nearest frame within ±2 s.
- `[Prior frame state]` → previous Qwen JSON result, if it was successfully parsed.

If any upstream step was skipped or failed, that line is simply absent.
Qwen never sees a `null` or empty placeholder — it sees no line at all.

---

## The Rolling State Pattern

The `_last_qwen` field enables temporal continuity in Qwen's reasoning.

Flow:
1. Qwen processes frame at `t=5.0 s` and returns a JSON with `vehicle_groups: [{type: truck, count: 2}]`.
2. `update_qwen_state()` stores this in `_last_qwen`.
3. Qwen processes frame at `t=7.0 s`. `context_for_frame(7.0)` includes:
   `[Prior frame state]: vehicles=2×truck; road=highway; condition=clear`
4. Qwen can now observe continuity: "the prior frame had 2 trucks; this frame has 3" is a valid observation.

What can go wrong:
- If the Qwen output at `t=5.0 s` was a JSON parse error, `update_qwen_state()` skips storing it.
  The prior state is either stale (from an earlier successful frame) or empty.
  This is the correct behavior: do not propagate a known-bad state.
- If Gemma misclassified the scene, the wrong `scene_type` is stored at initialization and propagates into every `domain_hint()` call for the entire video.
  This is the most impactful failure mode in the whole system.

---

## Why This Matters For Debugging

When the final synthesis or a Qwen output is wrong, trace backward through `VideoKnowledge`:

1. **Check the domain hint**: open `gemma_analysis.md`. Is `scene_type` correct?
   If not, every Florence caption and every Qwen system prompt was off-domain.

2. **Check the context string for the suspect frame**: reconstruct `context_for_frame(t_sec)`.
   Which lines are present? Which are absent?
   Absent lines = the upstream step failed or was skipped.

3. **Check the rolling state**: look at the Qwen output for `t_sec - Δt` (the previous frame).
   Is the prior state plausible? A wrong prior state propagates forward.

4. **Check timestamps**: if ASR or OCR lines appear at the wrong timestamps, the `t_sec` alignment is off.
   This is usually a video container timestamp issue (see Step 1).

---

## What Gets Accumulated: Full Inventory

Evidence accumulated by the end of Step 24, available in `VideoKnowledge`:

| Field | Source step | Time alignment | Max age |
|-------|------------|----------------|---------|
| `scene_type` | Gemma (Step 3) | Video-global | Entire video |
| `n_transitions` | Gemma (Step 3) | Video-global | Entire video |
| `known_entities` | Detection (Step 8) | Video-global | Entire video |
| `_captions[t]` | Florence (Step 4) | Per-frame | ±2 s lookup |
| `_segments[...]` | Florence (Step 4) | Temporal range | By segment |
| `_asr[t]` | ASR (Step 5) | Per-segment | ±2 s lookup |
| `_ocr[t]` | OCR (Step 6) | Per-frame | ±1 s lookup |
| `_depth[t]` | Depth (Step 7) | Per-frame | ±2 s lookup |
| `_detections[t]` | Detection (Step 8) | Per-frame | ±2 s lookup |
| `_last_qwen` | Qwen (Step 12) | Rolling | Previous frame only |

---

## Context Contamination: The Main Risk

Agentic context flow creates a compounding risk: a wrong observation injected early propagates into all later steps.

The most impactful contamination sources, ranked by severity:

1. **Wrong Gemma scene type** (Step 3): contaminates `domain_hint()` → every Florence caption is off-domain → every Qwen system prompt is wrong for the entire video.

2. **Wrong prior Qwen state** (Step 12): a Qwen hallucination at frame N becomes a "fact" in frame N+1's context. Errors can compound over many frames before a scene change resets the state. The LangGraph path mitigates this with a parse-error retry and by skipping `update_qwen_state()` for confirmed-bad frames.

3. **Wrong OCR text** (Step 6): OCR noise is injected into `[Visible text]` lines. Qwen may incorporate garbled characters as "evidence".

4. **Stale ASR** (Step 5): if the ASR window is too wide (±5 s instead of ±2 s), speech from one scene contaminates frames in an adjacent scene.

The audit step (conceptual Step 36, runtime step 30) is designed to surface these contamination paths.
If you see a wrong synthesis output, start with the audit document.

---

## LangGraph Orchestration Layer

The pipeline has a LangGraph-based orchestrator (`runner_graph.py`) that replaces the
monolithic `run_video_pipeline()` function. Activate it with `SELFSUVIS_USE_GRAPH=1`.

### PipelineState — the graph's state schema

`graph_state.py` defines `PipelineState`, a `TypedDict` that is the single container
for all data flowing through the graph. Each node receives the full state and returns
only the keys it writes; LangGraph merges them with last-writer-wins.

Selected fields and their relation to `VideoKnowledge`:

| State field | Type | Source node | Relation to VideoKnowledge |
|-------------|------|-------------|---------------------------|
| `knowledge` | `VideoKnowledge` | `p1_extract_frames` | The same `VideoKnowledge` instance — mutated in-place and returned |
| `video_context` | `Dict` | Every node | Serialisable mirror of VideoKnowledge for LLM synthesis prompts |
| `agentic_trace` | `List[Dict]` | Every node | The same trace list passed to `step_agentic_flow_artifact` |
| `gemma_result` | `Dict` | `p2_gemma_analysis` | Feeds `knowledge.add_gemma()` in `p2_merge_parallel` |
| `caption_results` | `List` | `p2_florence_caption` | Feeds `knowledge.add_captions()` in `p2_merge_parallel` |
| `qwen_result` | `Dict` | `p2_qwen_caption` | Result of step that calls `knowledge.update_qwen_state()` |

The `knowledge` field is a Python object reference — not serialised — so it survives
in-process across all 24 nodes with zero copying overhead. For `SqliteSaver` persistence
across process restarts it is reconstructed from `video_context` via the
`_reconstruct_knowledge` helper in `graph_state.py`.

### Graph topology

```
Phase 1:  init_state → extract_frames → index_vectors
Phase 2:  gemma_analysis
            ↓ [fan-out: runs concurrently]
          florence_caption  asr  ocr  depth  detection
            ↓ [fan-in]
          merge_parallel → platform_fusion → yolo_sam → gemma_tracking
          → map_3d_submit → world_model → qwen_caption → unidrive
          → scenetok → base_search → map_3d_join → full_fusion
Phase 3:  ssl_finetune → [ssl_gate] → distill → onnx_export → ft_search → compare
                                    ↘ (gate fails) ─────────────────────────────┐
Phase 4:  multi_model_compare → synthesis → audit → emit_analytics → END ◄──────┘
```

Steps 4–8 (Florence, ASR, OCR, depth, detection) run concurrently — they write
distinct state keys (`caption_results`, `asr_result`, `ocr_result`, `depth_result`,
`det_result`) so there is no reducer conflict. `merge_parallel` is the fan-in barrier
that calls all the `knowledge.add_*()` methods in dependency order.

The 3D map (step 16) is still submitted to a background `ThreadPoolExecutor` via
`map_3d_submit` and joined at `map_3d_join` — the same pattern as in the monolith,
preserving overlap with the slower VLM steps that follow.

### Agentic node improvements

The LangGraph path upgrades the six LLM nodes. Key design patterns used:

**Claim verification (step 3 — `node_p2_gemma_analysis`)**

After `step_gemma_analysis()` returns, every `fact_verification` claim is scored against
the already-computed CLIP frame embeddings in the in-memory store. No new model load;
cost is a numpy dot product per claim. Claims below cosine similarity 0.25 are marked
`clip_verified=false`. This makes the Gemma domain-hint failure mode visible before it
propagates to Florence and Qwen.

**JSON-guard fallback (step 10 — `node_p2_gemma_tracking`)**

The existing step already retries three times on JSON parse failure. The graph wrapper
detects the post-retry state where `n_tracked_objects == 0` and `target_labels == []`
and injects `DEFAULT_TRACKING_TARGETS = ["person", "vehicle", "sign"]`. RF-DETR now
always has something to track when Gemma fails, rather than silently skipping the step.

**Parse-error retry with prior-state preservation (step 12 — `node_p2_qwen_caption`)**

Frames with `parse_error=True` are retried once with a simplified prompt (no extra
context, domain hint only). If the retry succeeds, the result replaces the error entry.
Critically, `knowledge.update_qwen_state()` is only called on confirmed-good results —
the prior-state chain now skips bad frames rather than anchoring on them.

**MoE consensus scoring (step 13 — `node_p2_unidrive`)**

Per-frame Jaccard similarity across expert `recommended_action` fields gives a consensus
score. Frames below 0.5 are flagged `low_moe_agreement=true` and logged as warnings.
The `mean_moe_agreement` and `low_agreement_frame_count` fields are surfaced in the
agentic trace, making disagreement explicit for the audit step.

**Draft → critique → conditional regeneration (runtime step 29 — `node_p4_synthesis`)**

1. `step_video_synthesis()` generates the ontology and narrative (unchanged logic).
2. A critique prompt asks the LLM to compare the generation against a factual evidence
   summary built from CLIP-grounded detections, captions, and ASR — not another LLM call.
3. If the verdict is `MAJOR_CONTRADICTION`, synthesis is re-run with the critique note
   prepended to `video_context`. The regenerated file overwrites the original.

**Reflection sub-loop (runtime step 30 — `node_p4_audit`)**

After `step_agentic_flow_artifact()` produces `agentic_flow.md`, a reflection prompt
checks whether every step ID in `agentic_trace` appears in the audit text, and whether
cross-step context propagation risk is mentioned. If the verdict is `HAS_GAPS`, the gap
list is appended as a `## Reflection Gaps` section. This is the minimum viable version
of a ReAct loop — one self-check round, deterministic fallback on failure.

### Shared helpers (`pipeline/nodes/helpers.py`)

| Helper | Used by |
|--------|---------|
| `json_guard(raw, required_keys)` | Step 10 — validates Gemma JSON before acting on it |
| `llm_call_with_retry(endpoint, payload, *, max_attempts, backoff_base)` | Runtime steps 29, 30 |
| `critique_pass(endpoint, model, generation, evidence_summary)` | Runtime step 29 |
| `moe_consensus_score(expert_outputs, field)` | Step 13 |
| `low_agreement_frames(results, threshold)` | Step 13 |
| `build_evidence_summary(state)` | Runtime steps 29, 30 — builds CLIP-grounded evidence string |

---

## Human Reading Strategy

To understand the agentic part of the pipeline, read these in order:

1. [`ssv_vdp/steps/common.py`](../../packages/ss-fusion/src/ssv_vdp/steps/common.py) — `VideoKnowledge` class and `context_for_frame()`
2. [`ssv_vdp/steps/caption/`](../../packages/ss-fusion/src/ssv_vdp/steps/caption) — how Florence, ASR, OCR, depth, and detection deposit into `VideoKnowledge`
3. [`ssv_vdp/steps/caption/`](../../packages/ss-fusion/src/ssv_vdp/steps/caption) — how Qwen queries `context_for_frame()` and updates rolling state
4. [`ssv_vdp/pipeline/runner.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/runner.py) — the top-level orchestration and the `VideoKnowledge` lifecycle
5. [`ssv_vdp/pipeline/state.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/state.py) — `PipelineState` TypedDict and how it wraps `VideoKnowledge`
6. [`ssv_vdp/pipeline/graph.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/graph.py) — graph topology, node wiring, and `run_graph_pipeline()`
7. [`ssv_vdp/pipeline/nodes/helpers.py`](../../packages/ss-fusion/src/ssv_vdp/pipeline/nodes/helpers.py) — shared agentic primitives

## Questions To Ask While Reading

- What evidence is added at this step?
- What later step consumes it?
- Is the evidence time-aligned (per-frame) or video-global?
- What happens if this step is skipped — which downstream lines go silent?
- Can stale or wrong context propagate from this step onward?

## Related Docs

- [Perception core: Steps 1-8](02_perception_core_steps_01_08.md)
- [Tracking and mapping: Steps 21-27](05_tracking_mapping_steps_21_27.md)
- [Adaptation and audit: Steps 28-35](06_adaptation_eval_steps_28_35.md)
- [Pipeline architecture](../reference/pipeline.md)

---

## Learning Resources — Agentic Knowledge Flow

The `VideoKnowledge` class is an episodic memory store with a rolling state mechanism — a specific implementation of the broader problem of how AI agents accumulate and consume structured evidence. Resources below go from cognitive-science foundations to current engineering practice.

---

### Memory and context management in language models

**Basics**
- Weng, "LLM-powered Autonomous Agents" (Lilian Weng's blog, 2023). The best non-paper introduction to the four components of an AI agent: memory, planning, tools, and action. The "memory" section maps directly to `VideoKnowledge`'s per-frame evidence store. [lilianweng.github.io/posts/2023-06-23-agent](https://lilianweng.github.io/posts/2023-06-23-agent)
- Vaswani et al., "Attention Is All You Need" (2017). The transformer attention mechanism is fundamentally a learned read/write to a key-value memory. Understanding attention as memory access clarifies why context window length limits are a hard engineering constraint. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

**Core paper**
- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (Stanford, 2023). The most concrete implementation of an agent memory system with retrieve, reflect, and plan operations. `VideoKnowledge.context_for_frame()` is a simpler version of their retrieval-augmented context construction. [arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)

**Deep dive**
- Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023). Treats the LLM context window as a CPU register file with explicit page-in/page-out of external memory — the architecture that would replace `VideoKnowledge`'s flat dict if the pipeline scaled to very long missions. [arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)

---

### Rolling state and temporal context propagation

**Basics**
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (Neural Computation, 1997). The LSTM architecture is the classical solution to the problem that `_last_qwen` solves manually: propagating relevant state forward while forgetting irrelevant state. Understanding LSTM gates makes the risks of `VideoKnowledge`'s unbounded error propagation explicit. [doi: 10.1162/neco.1997.9.8.1735]

**Core paper**
- Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (2019). Segment-level recurrence with cached hidden states — the mechanism that `_last_qwen` approximates with a dictionary. The paper's analysis of why fixed-window attention fails on long sequences directly applies to long-mission Qwen runs. [arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860)

---

### Context contamination: grounding and attribution

**Why it matters:** The five contamination risks in `VideoKnowledge` (wrong domain hint, stale Qwen state, garbled OCR, misaligned ASR, wrong depth) all share a common structure: a wrong input at step N poisons every downstream consumer without any error signal. This is the core reliability problem in any pipeline that chains LLM outputs.

**Core paper**
- Maynez et al., "On Faithfulness and Factuality in Abstractive Summarization" (ACL, 2020). Defines the distinction between intrinsic hallucination (contradicts source) and extrinsic hallucination (adds information not in source). The audit step detects both types. [arxiv.org/abs/2005.00661](https://arxiv.org/abs/2005.00661)

**Deep dive**
- Min et al., "FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long-form Text Generation" (2023). Per-claim factuality scoring — the automated equivalent of what the agentic audit does manually. [arxiv.org/abs/2305.14251](https://arxiv.org/abs/2305.14251)

---

### Practical tooling for agentic pipelines

- LangChain documentation: [python.langchain.com](https://python.langchain.com). The most widely used framework for chaining LLM calls with tools and memory. Understanding LangChain's `ConversationBufferWindowMemory` and `VectorStoreRetrieverMemory` clarifies the design space that `VideoKnowledge` occupies.
- LlamaIndex documentation: [docs.llamaindex.ai](https://docs.llamaindex.ai). Focused on retrieval-augmented generation — the approach most relevant to the query path in this pipeline.
- Mialon et al., "Augmented Language Models: a Survey" (Meta AI, 2023). Unified framework for tools, retrieval, and memory in LLMs. [arxiv.org/abs/2302.07842](https://arxiv.org/abs/2302.07842)
