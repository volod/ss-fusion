"""Phase 2 serial nodes: gemma_analysis, merge_parallel, platform_fusion,
world_model, qwen_caption (agentic), unidrive (agentic), scenetok, cosmos3,
base_search, full_fusion.
"""

import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.config import settings
from selfsuvis.pipeline.core.logging import get_logger

from ..state import PipelineState
from ..runner import _append_agentic_step
from .helpers import (
    GEMMA_CLAIM_MIN_SIM,
    low_agreement_frames,
    moe_consensus_score,
)

_log = get_logger(__name__)


# -- Step 03: Gemma multimodal analysis (with claim verification) --------------


def node_p2_gemma_analysis(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import _unload_ollama_model, step_gemma_analysis

    args = state["args"]
    gemma_api_url = getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL
    gemma_api_model = getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL

    t0 = time.monotonic()
    j = step_gemma_analysis(
        Path(state["video_path"]),
        state["video_id"],
        state["video_name"],
        Path(state["video_dir"]),
        state["frame_list"],
        state["models"],
        gemma_api_url=gemma_api_url,
        gemma_api_model=gemma_api_model,
    )

    # Agentic improvement: verify Gemma claims against CLIP frame embeddings.
    if not j.get("skipped") and state["models"].get("clip"):
        j = _verify_gemma_claims(j, state)

    knowledge = state.get("knowledge")
    if knowledge and not j.get("skipped"):
        knowledge.add_gemma(
            j.get("task_results", {}),
            mnn_dino=j.get("dino_comparison", {}).get("mnn_rate") or 0.0,
        )

    video_context = dict(state.get("video_context", {}))
    if not j.get("skipped"):
        video_context["gemma_analysis"] = {
            "n_frames": j.get("n_frames", 0),
            "n_tasks": len(j.get("task_results", {})),
            "task_results": j.get("task_results", {}),
            "mnn_rate_dino": j.get("dino_comparison", {}).get("mnn_rate"),
            "mnn_rate_clip": j.get("clip_comparison", {}).get("mnn_rate"),
        }
        _precomp = j.get("structured_scene_summary") or j.get("structured_scene")
        if _precomp:
            video_context["gemma_structured_scene"] = _precomp

    if gemma_api_url and gemma_api_model and state["device"] == "cuda":
        _unload_ollama_model(gemma_api_url, gemma_api_model)

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="03",
        title="Gemma multimodal analysis",
        description="Run coarse video-level reasoning to infer dominant scene type, transitions, and clusters.",
        status="skipped" if j.get("skipped") else "ok",
        context_inputs=["sampled video frames", "existing embeddings"],
        context_outputs=[
            f"scene type {knowledge.scene_type if knowledge else 'unknown'}",
            f"unverified_claims={j.get('unverified_claims', 0)}",
        ]
        if not j.get("skipped")
        else ["no persistent Gemma context"],
        risks=[
            "scene classification can over-generalize from sparse samples",
            "wrong domain hint can bias Florence and Qwen toward the wrong narrative",
        ],
        artifacts=["gemma_analysis.md"] if not j.get("skipped") else [],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["J_gemma"] = time.monotonic() - t0

    return {
        "gemma_result": j,
        "knowledge": knowledge,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


def _verify_gemma_claims(j: dict[str, Any], state: PipelineState) -> dict[str, Any]:
    """Post-hoc claim verification using CLIP cosine similarity against frame embeddings.

    For each claim in task_results["fact_verification"]["claims"], compute cosine
    similarity between claim text and the nearest frame embedding.  Claims below
    GEMMA_CLAIM_MIN_SIM are marked unverified.  Uses only the already-computed
    in-memory store — no new model load.
    """
    try:
        clip_model = state["models"].get("clip")
        store = state.get("store")
        if clip_model is None or store is None:
            return j

        claims = j.get("task_results", {}).get("fact_verification", {}).get("claims", [])
        if not claims:
            return j

        unverified = 0
        verified_claims = []
        for claim in claims:
            text = claim.get("statement") or claim.get("claim") or str(claim)
            try:
                text_emb = clip_model.encode_texts([text])  # (1, D)
                # In-memory store: retrieve top-1 by cosine similarity.
                hits = store.search(text_emb[0], top_k=1)
                top_score = hits[0]["score"] if hits else 0.0
            except Exception:
                top_score = 0.0
            passed = float(top_score) >= GEMMA_CLAIM_MIN_SIM
            if not passed:
                unverified += 1
            verified_claims.append(
                {**claim, "clip_verified": passed, "clip_score": float(top_score)}
            )

        j = {**j}
        j.setdefault("task_results", {}).setdefault("fact_verification", {})
        j["task_results"]["fact_verification"]["claims"] = verified_claims
        j["unverified_claims"] = unverified
        if unverified:
            _log.info(
                "  Gemma claim verification: %d/%d claims not frame-supported (score < %.2f)",
                unverified,
                len(claims),
                GEMMA_CLAIM_MIN_SIM,
            )
    except Exception as exc:
        _log.debug("Gemma claim verification skipped: %s", exc)
    return j


# -- Fan-in merge: commit all parallel results to knowledge & video_context ----


def node_p2_merge_parallel(state: PipelineState) -> dict[str, Any]:
    knowledge = state.get("knowledge")
    caption_results = state.get("caption_results", [])
    asr_result = state.get("asr_result", {"skipped": True, "subtitle_map": {}, "segments": []})
    ocr_result = state.get("ocr_result", {"skipped": True, "ocr_results": []})
    depth_result = state.get("depth_result", {"skipped": True, "depth_results": []})
    det_result = state.get("det_result", {"skipped": True, "detection_results": []})

    if knowledge:
        if caption_results:
            knowledge.add_captions(caption_results)
        knowledge.add_asr(asr_result.get("subtitle_map", {}))
        if not ocr_result.get("skipped"):
            knowledge.add_ocr(ocr_result.get("ocr_results", []))
        if not depth_result.get("skipped"):
            knowledge.add_depth(depth_result.get("depth_results", []))
        if not det_result.get("skipped"):
            knowledge.add_detections(det_result.get("detection_results", []))

    video_context = dict(state.get("video_context", {}))
    video_context["captions"] = caption_results
    video_context["caption_segments"] = len(getattr(knowledge, "_segments", [])) if knowledge else 0
    video_context["asr_segments"] = asr_result.get("segments", [])
    video_context["ocr"] = ocr_result.get("ocr_results", [])
    if not det_result.get("skipped"):
        obj_counts: dict[str, int] = {}
        for r in det_result.get("detection_results", []):
            for d in r.get("detections", []):
                lbl = d.get("label", "unknown")
                obj_counts[lbl] = obj_counts.get(lbl, 0) + 1
        video_context["detections"] = obj_counts

    agentic_trace = list(state.get("agentic_trace", []))
    for step_id, title, description, skipped, outputs, risks, artifacts in [
        (
            "04",
            "Scene captioning",
            "Generate per-frame scene captions and coarse temporal segments.",
            not caption_results,
            [f"{len(caption_results)} scene captions"],
            ["caption hallucinations can create false scene priors"],
            ["scene_captions.md"] if caption_results else [],
        ),
        (
            "05",
            "ASR transcription",
            "Transcribe audio and align subtitles to frames.",
            asr_result.get("skipped", True),
            [f"{len(asr_result.get('segments', []))} ASR segments"],
            ["transcription errors can inject false entities"],
            ["asr_subtitles.md"] if not asr_result.get("skipped") else [],
        ),
        (
            "06",
            "OCR extraction",
            "Extract visible text from frames.",
            ocr_result.get("skipped", True),
            [f"{ocr_result.get('non_empty', 0)} frames with OCR text"],
            ["false OCR tokens can create wrong named-entity context"],
            [],
        ),
        (
            "07",
            "Depth estimation",
            "Estimate relative scene geometry.",
            depth_result.get("skipped", True),
            [f"{depth_result.get('ok_count', 0)} depth-estimated frames"],
            ["monocular depth can confuse scale and elevation"],
            [],
        ),
        (
            "08",
            "Object detection",
            "Detect frame-level entities.",
            det_result.get("skipped", True),
            [f"{det_result.get('total_objects', 0)} detected objects"],
            ["class confusion can misidentify critical objects"],
            [],
        ),
    ]:
        _append_agentic_step(
            agentic_trace,
            step_id=step_id,
            title=title,
            description=description,
            status="skipped" if skipped else "ok",
            context_inputs=["frames"],
            context_outputs=outputs if not skipped else [f"no {title.lower()} context"],
            risks=risks,
            artifacts=artifacts,
        )

    return {
        "knowledge": knowledge,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
    }


# -- Platform state fusion -----------------------------------------------------


def node_p2_platform_fusion(state: PipelineState) -> dict[str, Any]:
    from ..steps_fusion import step_platform_state_fusion

    result = step_platform_state_fusion(
        Path(state["video_path"]),
        state["frame_list"],
        state["video_name"],
        Path(state["video_dir"]),
    )
    knowledge = state.get("knowledge")
    if knowledge:
        knowledge.add_state_fusion(result.get("posterior_samples", []))

    video_context = dict(state.get("video_context", {}))
    video_context["platform_state_fusion"] = result.get("summary", {})

    return {
        "platform_fusion_result": result,
        "knowledge": knowledge,
        "video_context": video_context,
    }


# -- Step 11: World model ------------------------------------------------------


def node_p2_world_model(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import _prep_vram_for_step, step_world_model_pass

    args = state["args"]
    world_result: dict[str, Any] = {"skipped": True, "world_results": []}
    t0 = time.monotonic()

    if args.world_model:
        _prep_vram_for_step(state["models"], state["device"])
        world_result = step_world_model_pass(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            models=state["models"],
        )

    video_context = dict(state.get("video_context", {}))
    if not world_result.get("skipped"):
        video_context["world_model_clips"] = world_result.get("ok_count", 0)

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="11",
        title="World model pass",
        description="Compress clips into temporal embeddings to capture motion-level context.",
        status="skipped" if world_result.get("skipped") else "ok",
        context_inputs=["ordered frame clips"],
        context_outputs=[f"{world_result.get('ok_count', 0)} temporal clip embeddings"]
        if not world_result.get("skipped")
        else ["no temporal clip context"],
        risks=["temporal embeddings are hard to interpret and easy to overtrust"],
        artifacts=[],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["Q_world"] = time.monotonic() - t0
    return {
        "world_result": world_result,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


# -- Step 12: Qwen captioning (agentic: retry on parse_error) -----------------


def node_p2_qwen_caption(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import step_qwen_captioning

    args = state["args"]
    asr_result = state.get("asr_result", {})
    ocr_result = state.get("ocr_result", {})
    knowledge = state.get("knowledge")
    qwen_result: dict[str, Any] = {"skipped": True, "results": []}
    t0 = time.monotonic()

    if args.qwen:
        qwen_result = step_qwen_captioning(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            subtitle_map=asr_result.get("subtitle_map", {}),
            ocr_results=ocr_result.get("ocr_results", []),
            clip_prescreen_fn=lambda _img: True,
            knowledge=knowledge,
        )
        # Agentic improvement: retry frames with parse_error using simplified prompt.
        if not qwen_result.get("skipped"):
            qwen_result = _qwen_retry_parse_errors(qwen_result, state, knowledge)

    video_context = dict(state.get("video_context", {}))
    if not qwen_result.get("skipped"):
        video_context["qwen_captions"] = qwen_result.get("results", [])

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="12",
        title="Qwen detailed captioning",
        description="Fuse visual frames with accumulated context for structured per-frame reasoning.",
        status="skipped" if qwen_result.get("skipped") else "ok",
        context_inputs=[
            "frame image",
            "Florence priors",
            "ASR/OCR/depth/detection cues",
            "prior Qwen state",
        ],
        context_outputs=[
            f"{qwen_result.get('ok_count', 0)} detailed captions",
            f"retry_successes={qwen_result.get('retry_success_count', 0)}",
        ]
        if not qwen_result.get("skipped")
        else ["no detailed reasoning context"],
        risks=[
            "upstream misidentification compounds inside one prompt",
            "previous-frame state can anchor model to stale context",
        ],
        artifacts=["detailed_captions.md"] if not qwen_result.get("skipped") else [],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["R_qwen"] = time.monotonic() - t0
    return {
        "qwen_result": qwen_result,
        "knowledge": knowledge,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


def _qwen_retry_parse_errors(
    qwen_result: dict[str, Any],
    state: PipelineState,
    knowledge: Any,
) -> dict[str, Any]:
    """Retry individual Qwen results that had parse_error=True with a simplified prompt."""
    from ..steps_caption import step_qwen_captioning

    results = qwen_result.get("results", [])
    error_indices = [i for i, r in enumerate(results) if r.get("parse_error")]
    if not error_indices:
        return qwen_result

    retry_successes = 0
    retry_frame_list = [
        state["frame_list"][i] for i in error_indices if i < len(state["frame_list"])
    ]
    if not retry_frame_list:
        return qwen_result

    try:
        retry_out = step_qwen_captioning(
            retry_frame_list,
            state["video_name"],
            Path(state["video_dir"]),
            subtitle_map={},
            ocr_results=[],
            clip_prescreen_fn=lambda _img: True,
            knowledge=None,  # no prior-state chain on retry
        )
        retry_results = retry_out.get("results", [])
        for local_idx, orig_idx in enumerate(error_indices):
            if local_idx < len(retry_results) and not retry_results[local_idx].get("parse_error"):
                results[orig_idx] = {**retry_results[local_idx], "retry_used": True}
                retry_successes += 1
    except Exception as exc:
        _log.debug("Qwen retry pass failed: %s", exc)

    qwen_result = dict(qwen_result)
    qwen_result["results"] = results
    qwen_result["parse_error_count"] = len(error_indices)
    qwen_result["retry_success_count"] = retry_successes
    if retry_successes:
        _log.info("  Qwen retry: recovered %d/%d parse errors", retry_successes, len(error_indices))
    return qwen_result


# -- Step 13: UniDriveVLA (agentic: MoE consensus scoring) --------------------


def node_p2_unidrive(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import step_unidrive_analysis

    args = state["args"]
    asr_result = state.get("asr_result", {})
    ocr_result = state.get("ocr_result", {})
    knowledge = state.get("knowledge")
    unidrive_result: dict[str, Any] = {"skipped": True, "results": []}
    t0 = time.monotonic()

    if getattr(args, "unidrive", False):
        unidrive_result = step_unidrive_analysis(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            subtitle_map=asr_result.get("subtitle_map", {}),
            ocr_results=ocr_result.get("ocr_results", []),
            knowledge=knowledge,
        )
        # Agentic improvement: compute MoE consensus scores per frame.
        if not unidrive_result.get("skipped"):
            unidrive_result = _score_unidrive_consensus(unidrive_result, state)

    video_context = dict(state.get("video_context", {}))
    if not unidrive_result.get("skipped"):
        video_context["unidrive_analysis"] = unidrive_result.get("results", [])

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="13",
        title="UniDriveVLA expert analysis",
        description="Run UniDriveVLA for understanding/perception/planning with MoE consensus scoring.",
        status="skipped" if unidrive_result.get("skipped") else "ok",
        context_inputs=["sampled frames", "ASR/OCR context", "agentic knowledge"],
        context_outputs=[
            f"{unidrive_result.get('ok_count', 0)} UniDrive analyses",
            f"mean_moe_agreement={unidrive_result.get('mean_moe_agreement', 1.0):.3f}",
            f"low_agreement_frames={unidrive_result.get('low_agreement_frame_count', 0)}",
        ]
        if not unidrive_result.get("skipped")
        else ["no UniDrive context"],
        risks=[
            "planning advice may be overconfident for non-driving footage",
            "expert consensus can hide meaningful disagreement if prompts are too generic",
        ],
        artifacts=["unidrive_analysis.md"] if not unidrive_result.get("skipped") else [],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["S_unidrive"] = time.monotonic() - t0
    return {
        "unidrive_result": unidrive_result,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


def _score_unidrive_consensus(
    unidrive_result: dict[str, Any], state: PipelineState
) -> dict[str, Any]:
    results = unidrive_result.get("results", [])
    if not results:
        return unidrive_result

    low_idxs = low_agreement_frames(results)
    scores = []
    for idx, frame_result in enumerate(results):
        experts = frame_result.get("experts", [])
        score = moe_consensus_score(experts) if len(experts) >= 2 else 1.0
        scores.append(score)
        if idx in low_idxs:
            results[idx] = {**frame_result, "low_moe_agreement": True, "moe_score": score}
            t_sec = frame_result.get("t_sec", 0.0)
            _log.warning(
                "  UniDrive step 13: low MoE agreement at t=%.1fs (score=%.3f)", t_sec, score
            )

    mean_score = sum(scores) / len(scores) if scores else 1.0
    unidrive_result = dict(unidrive_result)
    unidrive_result["results"] = results
    unidrive_result["low_agreement_frame_count"] = len(low_idxs)
    unidrive_result["mean_moe_agreement"] = mean_score
    return unidrive_result


# -- Step 14: SceneTok ---------------------------------------------------------


def node_p2_scenetok(state: PipelineState) -> dict[str, Any]:
    from ..steps_scenetok import step_scenetok

    args = state["args"]
    scenetok_result: dict[str, Any] = {"skipped": True}
    t0 = time.monotonic()

    if getattr(args, "scenetok", False):
        import os

        _api_url = getattr(args, "scenetok_api_url", "") or settings.SCENETOK_API_URL
        _checkpoint = getattr(args, "scenetok_checkpoint", "") or settings.SCENETOK_CHECKPOINT
        if _api_url:
            os.environ.setdefault("SCENETOK_API_URL", _api_url)
        if _checkpoint:
            os.environ.setdefault("SCENETOK_CHECKPOINT", _checkpoint)
        scenetok_result = step_scenetok(
            state["frame_list"],
            Path(state["video_dir"]),
            checkpoint=_checkpoint,
            mode=settings.SCENETOK_MODE,
        )

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="14",
        title="SceneTok scene compression + segmentation",
        description="Encode frame sequence into compact scene tokens via SceneTok encoder.",
        status="skipped" if scenetok_result.get("skipped") else "ok",
        context_inputs=["sampled keyframes"],
        context_outputs=[f"{scenetok_result.get('n_tokens', 0)} scene tokens"]
        if not scenetok_result.get("skipped")
        else ["no SceneTok context"],
        risks=["~24 GB VRAM required for local inference"],
        artifacts=["scenetok_tokens.npz"] if not scenetok_result.get("skipped") else [],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["S_scenetok"] = time.monotonic() - t0
    return {
        "scenetok_result": scenetok_result,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


# -- Step 15: Cosmos3 world-model inference ------------------------------------


def node_p2_cosmos3(state: PipelineState) -> dict[str, Any]:
    from ...steps.cosmos3 import step_cosmos3_inference

    args = state["args"]
    cosmos3_result: dict[str, Any] = {"skipped": True, "clips": [], "n_clips": 0}
    t0 = time.monotonic()

    if getattr(args, "cosmos3", None):
        from ..steps_caption import _prep_vram_for_step  # type: ignore[attr-defined]

        _prep_vram_for_step(state["models"], state["device"])
        cosmos3_result = step_cosmos3_inference(
            state["frame_list"],
            state["video_name"],
            Path(state["video_dir"]),
            device=state["device"],
        )

    video_context = dict(state.get("video_context", {}))
    if not cosmos3_result.get("skipped"):
        video_context["cosmos3_analysis"] = cosmos3_result.get("clips", [])

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="15",
        title="Cosmos3 world-model inference",
        description=(
            "Run nvidia/Cosmos3-Nano (or vLLM-Omni sidecar) on sampled video clips to produce "
            "an omnimodal scene understanding narrative."
        ),
        status="skipped" if cosmos3_result.get("skipped") else "ok",
        context_inputs=["sampled frame clips", "physical-AI analysis prompt"],
        context_outputs=[
            f"{cosmos3_result.get('n_clips', 0)} clip analyses",
            f"scene_type={cosmos3_result.get('scene_type', '')}",
        ]
        if not cosmos3_result.get("skipped")
        else ["no Cosmos3 world-model context"],
        risks=[
            "omnimodal model may hallucinate entity interactions not present in frames",
            "~32 GB VRAM required for full local load; layerwise offload degrades throughput",
        ],
        artifacts=["cosmos3_inference.json"] if not cosmos3_result.get("skipped") else [],
    )

    stats = dict(state.get("stats", {}))
    stats.setdefault("timings", {})["S_cosmos3"] = time.monotonic() - t0
    return {
        "cosmos3_result": cosmos3_result,
        "video_context": video_context,
        "agentic_trace": agentic_trace,
        "stats": stats,
    }


# -- Step 16: Base model search ------------------------------------------------


def node_p2_base_search(state: PipelineState) -> dict[str, Any]:
    from ..steps_caption import (
        _models_on_device,
        _prep_vram_for_step,
        _restore_models_to_gpu,
    )
    from ..steps_embed import step_base_model_search_test

    args = state["args"]
    models = state["models"]
    device = state["device"]
    clip_dino_on_gpu = state.get("clip_dino_on_gpu", False)

    if device == "cuda" and not clip_dino_on_gpu:
        _qwen_url = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
        _qwen_model_name = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
        _prep_vram_for_step(
            models,
            device,
            extra_sidecars=[(_qwen_url, _qwen_model_name)],
            label="base-search restore",
        )
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)

    t0 = time.monotonic()
    c = step_base_model_search_test(
        state["frame_list"],
        state["store"],
        state["is_qdrant"],
        models,
        state["video_id"],
        state["video_name"],
        Path(state["video_dir"]),
        top_k=args.top_k,
    )

    base_results = c["results"]
    query_frame = c["query_frame"]
    query_t_sec = c["query_t_sec"]

    stats = dict(state.get("stats", {}))
    stats["base_top_score"] = base_results[0]["score"] if base_results else 0.0
    stats.setdefault("timings", {})["C_base_search"] = time.monotonic() - t0

    agentic_trace = list(state.get("agentic_trace", []))
    _append_agentic_step(
        agentic_trace,
        step_id="15",
        title="Base search test",
        description="Measure retrieval behavior of the base model as the control reference for adaptation steps.",
        status="ok",
        context_inputs=["retrieval index", "query frame"],
        context_outputs=[
            f"top-{len(base_results)} baseline matches",
            f"query at {query_t_sec:.1f}s",
        ],
        risks=["one query frame can underrepresent broader retrieval behavior"],
        artifacts=["base_search.md"],
    )

    return {
        "base_results": base_results,
        "query_frame": query_frame,
        "query_t_sec": query_t_sec,
        "clip_dino_on_gpu": clip_dino_on_gpu,
        "stats": stats,
        "agentic_trace": agentic_trace,
    }


# -- Step 16 close + full state fusion ----------------------------------------


def node_p2_full_fusion(state: PipelineState) -> dict[str, Any]:
    from ..steps_fusion import step_full_state_fusion

    h = state.get("map_result", {})
    gemma_result = state.get("gemma_result", {})
    gemma_tracking_result = state.get("gemma_tracking_result", {})
    world_result = state.get("world_result", {})
    qwen_result = state.get("qwen_result", {})

    _rssm_mean = None
    if world_result and not world_result.get("skipped"):
        rssm_scores = world_result.get("rssm_scores") or []
        if rssm_scores:
            _rssm_mean = float(sum(rssm_scores) / len(rssm_scores))

    _qwen_captions = (
        qwen_result.get("structured_captions") or [] if not qwen_result.get("skipped") else []
    )

    video_context = state.get("video_context", {})
    _gemma_info = gemma_result if not gemma_result.get("skipped") else None
    _structured_scene = (
        (
            video_context.get("gemma_structured_scene")
            or (gemma_result.get("task_results", {}) or {}).get("structured_scene_summary")
            or gemma_result.get("structured_scene_summary")
        )
        if not gemma_result.get("skipped")
        else None
    )

    if isinstance(_structured_scene, dict):
        _gemma_info = {**(_gemma_info or {}), **_structured_scene}
    if not gemma_tracking_result.get("skipped") and gemma_tracking_result.get("scene_type"):
        _gemma_info = {
            **(_gemma_info or {}),
            "scene_type": gemma_tracking_result.get("scene_type"),
            "tracking_priority": gemma_tracking_result.get("tracking_priority", []),
        }

    t0 = time.monotonic()
    full_fusion_result = step_full_state_fusion(
        video_path=Path(state["video_path"]),
        frame_list=state["frame_list"],
        video_name=state["video_name"],
        video_dir=Path(state["video_dir"]),
        sfm_frame_positions=h.get("frame_positions") or [],
        tracking_results=(
            gemma_tracking_result.get("tracking_results") or []
            if not gemma_tracking_result.get("skipped")
            else []
        ),
        gemma_analysis=_gemma_info,
        qwen_captions=_qwen_captions or None,
        rssm_surprise_mean=_rssm_mean,
    )

    stats = dict(state.get("stats", {}))
    stats["full_fusion_tracks"] = full_fusion_result.get("track_count", 0)
    stats["full_fusion_scene"] = full_fusion_result.get("scene_type", "unknown")
    stats.setdefault("timings", {})["PS_full_fusion"] = time.monotonic() - t0

    video_context = dict(video_context)
    video_context["full_state_fusion"] = full_fusion_result.get("summary", {})

    return {
        "full_fusion_result": full_fusion_result,
        "video_context": video_context,
        "stats": stats,
    }
