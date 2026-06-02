"""Per-video pipeline orchestrator (monolithic path).

Contains run_video_pipeline() and its safe wrapper.
Activate the LangGraph path instead via SELFSUVIS_USE_GRAPH=1.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from selfsuvis.models.openclip_model import OpenCLIPEmbedder
from selfsuvis.pipeline.core import resolve_device, settings
from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.storage import InMemoryStore

try:
    from selfsuvis.models.dino_model import DINOEmbedder
    _HAS_DINO = True
except Exception:
    _HAS_DINO = False

try:
    from selfsuvis.models.gemma_model import GemmaEmbedder
    _HAS_GEMMA = True
except Exception:
    _HAS_GEMMA = False

from ...steps.common import (
    _TEXT_PROMPTS,
    VideoKnowledge,
    _banner,
    _step,
    _Timer,
)
from ._agentic import _append_agentic_step, _build_context_prompt
from ._analytics import _emit_local_run_analytics
from ._compare import step_compare_and_describe, step_multi_model_compare
from ._synthesis import step_agentic_flow_artifact, step_video_synthesis

_log = get_logger(__name__)

_TOTAL_STEPS = 35
_SSL_GATE_MAX_LOSS = 10.0


def run_video_pipeline(
    args: Any,
    video_path: Path,
    output_dir: Path,
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
    _out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all pipeline steps for a single video. Returns per-video stats dict.

    When ``SELFSUVIS_USE_GRAPH=1`` is set this function delegates to the
    LangGraph-based orchestrator in ``runner_graph.py`` and returns its result
    directly.  All existing callers remain unaffected.

    *_out* is an optional external dict that is used as the stats container.
    When provided, callers can inspect it for partial results if an exception
    escapes — the timings and frame counts recorded up to the failure point
    are preserved.
    """
    import os as _os

    if _os.getenv("SELFSUVIS_USE_GRAPH", "").lower() in ("1", "true", "yes"):
        from .graph import run_graph_pipeline

        return run_graph_pipeline(args, video_path, output_dir, models, store, is_qdrant, device)

    import concurrent.futures as _cf

    from ...steps.caption import (
        _guard_min_free_vram,
        _models_on_device,
        _offload_models_to_cpu,
        _prep_vram_for_step,
        _restore_models_to_gpu,
        _unload_known_sidecars,
        _unload_ollama_model,
        get_runtime_telemetry,
        reset_runtime_telemetry,
        step_asr_transcription,
        step_depth_estimation,
        step_gemma_analysis,
        step_gemma_segment_captions,
        step_object_detection,
        step_ocr_extraction,
        step_qwen_captioning,
        step_scene_captioning,
        step_unidrive_analysis,
        step_world_model_pass,
    )
    from ...steps.distill import step_distill, step_distill_stage2, step_export_model
    from ...steps.embed import (
        step_base_model_search_test,
        step_extract_frames,
        step_finetuned_model_search_test,
        step_index_to_store,
    )
    from ...steps.gemma_tracking import step_gemma_directed_tracking
    from ...steps.map import step_advise_3d_map_quality, step_create_3d_map
    from ...steps.report import write_multimodal_md
    from ...steps.scenetok import step_scenetok
    from ...steps.semantic_graph import step_build_semantic_environment_graph
    from ...steps.ssl import step_ssl_finetune
    from ...steps.yolo_sam import step_yolo_sam_detection

    video_name = video_path.stem
    video_id = video_name.replace(" ", "_").lower()
    video_dir = output_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    reset_runtime_telemetry()

    _banner(f"Processing video: {video_path.name}")
    _log.info("Output directory: %s", video_dir)

    # Use the shared container when provided so partial state is visible outside.
    if _out is None:
        _out = {}
    _out.update({"name": video_name, "video_path": str(video_path), "timings": {}})
    stats: dict[str, Any] = _out
    T = stats["timings"]

    # Accumulated context passed through the pipeline; enriches synthesis at step 22.
    video_context: dict[str, Any] = {"video_name": video_name}
    agentic_trace: list[dict[str, Any]] = []
    video_context["agentic_trace"] = agentic_trace

    # Tracks whether CLIP+DINO backbones are on GPU (relevant only when device=="cuda").
    clip_dino_on_gpu = device == "cuda" and _models_on_device(models, "cuda")

    # -- Phase 1: Foundational ingestion (no gate) -----------------------------
    _banner("Phase 1 — Foundational ingestion")

    # Step 01: Extract frames
    _step(1, _TOTAL_STEPS, "Frame extraction")
    with _Timer(T, "A_extract"):
        a = step_extract_frames(video_path, video_id, video_dir, fps=args.fps)
    frame_list: list[tuple[str, float]] = a["frame_list"]
    stats["frames"] = a["meta"]["frame_count"]
    stats["duration_sec"] = a["meta"]["duration_sec"]
    video_context["meta"] = {
        "frame_count": stats["frames"],
        "duration_sec": stats["duration_sec"],
    }
    _append_agentic_step(
        agentic_trace,
        step_id="01",
        title="Frame extraction",
        description="Decode the source video into a timestamped frame sequence that every later step reuses.",
        status="ok" if frame_list else "empty",
        context_inputs=["raw video bytes"],
        context_outputs=[
            f"{len(frame_list)} timestamped frames",
            f"duration {stats['duration_sec']:.1f}s",
            "frame timeline for all downstream alignment",
        ],
        risks=[
            "sampling can miss short-lived objects or events",
            "timestamp drift can misalign later ASR/OCR/detection context",
            "wrong extraction rate biases all downstream context",
        ],
        artifacts=["frames_metadata.json"],
    )
    if not frame_list:
        _log.error("No frames extracted — skipping video %s", video_path.name)
        return stats

    # Agentic knowledge accumulator — enriches downstream steps as each completes.
    knowledge = VideoKnowledge(
        video_name=video_name,
        duration_sec=stats["duration_sec"],
        frame_count=stats["frames"],
    )

    # Step 02: Index — needs CLIP+DINO on GPU
    if device == "cuda" and not clip_dino_on_gpu:
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(2, _TOTAL_STEPS, "Vector store indexing")
    with _Timer(T, "B_index"):
        b = step_index_to_store(video_path, video_id, store, is_qdrant, models, frame_list)
    if device == "cuda":
        clip_dino_on_gpu = _models_on_device(models, device)
    stats["index_sec"] = b["elapsed_sec"]
    _append_agentic_step(
        agentic_trace,
        step_id="02",
        title="Vector store indexing",
        description="Embed frames for retrieval and establish the baseline semantic memory used by search steps.",
        status="ok",
        context_inputs=["timestamped frames", "base CLIP/DINO embeddings"],
        context_outputs=[
            "retrieval index populated",
            f"index latency {b['elapsed_sec']:.1f}s",
            "baseline visual neighborhoods",
        ],
        risks=[
            "embedding collisions can mix semantically different frames",
            "duplicate-heavy footage can distort nearest-neighbor context",
            "wrong baseline neighborhoods affect later search comparisons",
        ],
        artifacts=[],
    )

    # -- Phase 2: Multimodal analysis (no gate, parallel where feasible) -------
    # The 3D-map step is CPU-bound (pycolmap SfM + fallback enrichment) and is
    # submitted to a background thread after depth/detection/tracking cues are
    # available. All GPU steps remain serialised on the main thread.
    _banner("Phase 2 — Multimodal analysis (parallel)")
    _map_executor = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sfm-bg")
    _map_future: _cf.Future | None = None

    # Step 03: Gemma open-weight multimodal analysis
    _step(3, _TOTAL_STEPS, "Gemma multimodal analysis → gemma_analysis.md")
    with _Timer(T, "J_gemma"):
        j = step_gemma_analysis(
            video_path,
            video_id,
            video_name,
            video_dir,
            frame_list,
            models,
            gemma_api_url=getattr(args, "gemma_api_url", ""),
            gemma_api_model=getattr(args, "gemma_api_model", ""),
        )
    if not j.get("skipped"):
        video_context["gemma_analysis"] = {
            "n_frames": j.get("n_frames", 0),
            "n_tasks": len(j.get("task_results", {})),
            "task_results": j.get("task_results", {}),
            "mnn_rate_dino": j.get("dino_comparison", {}).get("mnn_rate"),
            "mnn_rate_clip": j.get("clip_comparison", {}).get("mnn_rate"),
        }
        # step_gemma_analysis stores the scene under "structured_scene_summary";
        # fall back to "structured_scene" in case the schema was changed.
        _precomp = j.get("structured_scene_summary") or j.get("structured_scene")
        if _precomp:
            video_context["gemma_structured_scene"] = _precomp
        knowledge.add_gemma(
            j.get("task_results", {}),
            mnn_dino=j.get("dino_comparison", {}).get("mnn_rate") or 0.0,
        )
    _append_agentic_step(
        agentic_trace,
        step_id="03",
        title="Gemma multimodal analysis",
        description="Run coarse video-level reasoning to infer dominant scene type, transitions, clusters, and teacher-signal compatibility.",
        status="skipped" if j.get("skipped") else "ok",
        context_inputs=["sampled video frames", "existing embeddings"],
        context_outputs=[
            f"scene type {knowledge.scene_type or 'unknown'}",
            f"{knowledge.n_transitions} transitions",
            f"{knowledge.n_clusters} semantic clusters",
            "domain hint for captioning and later reasoning",
        ]
        if not j.get("skipped")
        else ["no persistent Gemma context"],
        risks=[
            "scene classification can over-generalize from sparse samples",
            "wrong domain hint can bias Florence and Qwen toward the wrong narrative",
            "teacher-similarity judgments can be mistaken for semantic truth",
        ],
        artifacts=["gemma_analysis.md"] if not j.get("skipped") else [],
    )
    # Unload Gemma from Ollama immediately after analysis — frees ~12+ GiB for Florence.
    _gemma_api_url_j = settings.GEMMA_API_URL or getattr(args, "gemma_api_url", "")
    _gemma_api_model_j = settings.GEMMA_API_MODEL or getattr(args, "gemma_api_model", "")
    if _gemma_api_url_j and _gemma_api_model_j and device == "cuda":
        _unload_ollama_model(_gemma_api_url_j, _gemma_api_model_j)

    # Step 04: Scene captioning — offloads CLIP+DINO internally, does NOT restore them
    caption_results: list[dict[str, Any]] = []
    if not args.no_caption:
        _step(4, _TOTAL_STEPS, "Florence-2 scene captioning → scene_captions.md")
        with _Timer(T, "L_caption"):
            l_cap = step_scene_captioning(
                frame_list,
                video_name,
                video_dir,
                device,
                models=models,
                qwen_api_url=getattr(args, "qwen_api_url", ""),
                qwen_model=getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
                florence_api_url=getattr(args, "florence_api_url", ""),
                florence_model=getattr(args, "florence_model", ""),
                domain_hint=knowledge.domain_hint(),
            )
        caption_results = l_cap.get("captions", [])
        knowledge.add_captions(caption_results)
        if device == "cuda":
            clip_dino_on_gpu = False  # Florence offloaded them; we keep them off for M–Q
    else:
        T["L_caption"] = 0.0
        _step(4, _TOTAL_STEPS, "Scene captioning (skipped — --no-caption)")
    video_context["captions"] = caption_results
    video_context["caption_segments"] = len(getattr(knowledge, "_segments", []))
    _append_agentic_step(
        agentic_trace,
        step_id="04",
        title="Scene captioning",
        description="Generate per-frame scene captions and coarse temporal segments to seed later context-aware reasoning.",
        status="skipped" if args.no_caption else "ok",
        context_inputs=[
            "timestamped frames",
            knowledge.domain_hint() or "no domain hint",
        ],
        context_outputs=[
            f"{len(caption_results)} scene captions",
            f"{len(getattr(knowledge, '_segments', []))} caption segments",
            "frame-level prior scene descriptions",
        ]
        if caption_results
        else ["no caption context"],
        risks=[
            "caption hallucinations can create false scene priors",
            "repeated captions may hide real transitions",
            "wrong segment boundaries can contaminate later frame context",
        ],
        artifacts=["scene_captions.md"] if caption_results else [],
    )

    # Step 4b: Gemma segment-boundary diff — runs only when GEMMA_API_URL is set
    # and caption results are available; no model loading required.
    seg_cap_result: dict[str, Any] = {"skipped": True, "boundary_diffs": []}
    _gemma_url_4b = getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL
    if _gemma_url_4b and caption_results:
        _log.info(
            "--- Step 4b/%d: Gemma 4 segment-boundary diffs → gemma_segment_captions.md",
            _TOTAL_STEPS,
        )
        with _Timer(T, "L_seg_caps"):
            seg_cap_result = step_gemma_segment_captions(
                frame_list,
                caption_results,
                video_name,
                video_dir,
                gemma_api_url=_gemma_url_4b,
                gemma_api_model=getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL,
            )
    else:
        T["L_seg_caps"] = 0.0
    if not seg_cap_result.get("skipped"):
        video_context["segment_diffs"] = seg_cap_result.get("boundary_diffs", [])
    _append_agentic_step(
        agentic_trace,
        step_id="04b",
        title="Gemma segment-boundary diffs",
        description="Identify scene transitions from caption segments and describe what changed between the last frame of segment N and the first frame of segment N+1.",
        status="skipped" if seg_cap_result.get("skipped") else "ok",
        context_inputs=["caption segments", "frame images at boundaries"],
        context_outputs=[
            f"{seg_cap_result.get('described_count', 0)}/{seg_cap_result.get('boundary_count', 0)} boundaries described",
        ]
        if not seg_cap_result.get("skipped")
        else ["no segment diff context"],
        risks=["two-image prompts increase per-call latency"],
        artifacts=["gemma_segment_captions.md"] if not seg_cap_result.get("skipped") else [],
    )

    # Step 05: ASR — evict any Ollama model that may still be resident from the
    # Florence-2 Qwen fallback (or a normal Qwen caption pass) before loading
    # Whisper, which needs ~1.6 GB VRAM.
    asr_result: dict[str, Any] = {"skipped": True, "subtitle_map": {}, "segments": []}
    if args.asr:
        _step(5, _TOTAL_STEPS, "ASR transcription → asr_subtitles.md")
        _prep_vram_for_step(models, device)
        with _Timer(T, "M_asr"):
            asr_result = step_asr_transcription(video_path, frame_list, video_name, video_dir)
    else:
        T["M_asr"] = 0.0
    video_context["asr_segments"] = asr_result.get("segments", [])
    knowledge.add_asr(asr_result.get("subtitle_map", {}))
    _append_agentic_step(
        agentic_trace,
        step_id="05",
        title="ASR transcription",
        description="Transcribe audio and align subtitles to frames so later reasoning can use speech context.",
        status="skipped" if asr_result.get("skipped") else "ok",
        context_inputs=["video audio stream", "frame timestamps"],
        context_outputs=[
            f"{len(asr_result.get('segments', []))} ASR segments",
            f"{asr_result.get('covered_frames', 0)} subtitle-covered frames",
            "audio context aligned to timestamps",
        ]
        if not asr_result.get("skipped")
        else ["no audio context"],
        risks=[
            "transcription errors can inject false entities or actions",
            "language mismatch can produce wrong context with high confidence",
            "subtitle-frame misalignment can contaminate visual reasoning",
        ],
        artifacts=["asr_subtitles.md"] if not asr_result.get("skipped") else [],
    )

    from ...steps.fusion import step_platform_state_fusion

    platform_fusion_result = step_platform_state_fusion(
        video_path,
        frame_list,
        video_name,
        video_dir,
    )
    knowledge.add_state_fusion(platform_fusion_result.get("posterior_samples", []))
    video_context["platform_state_fusion"] = platform_fusion_result.get("summary", {})

    # Step 06: OCR
    ocr_result: dict[str, Any] = {"skipped": True, "ocr_results": []}
    if args.ocr:
        _step(6, _TOTAL_STEPS, "OCR text extraction")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "N_ocr"):
            ocr_result = step_ocr_extraction(
                frame_list,
                video_name,
                video_dir,
                caption_results=caption_results,
            )
    else:
        T["N_ocr"] = 0.0
    video_context["ocr"] = ocr_result.get("ocr_results", [])
    knowledge.add_ocr(ocr_result.get("ocr_results", []))
    _append_agentic_step(
        agentic_trace,
        step_id="06",
        title="OCR extraction",
        description="Extract visible text from frames to enrich object and scene interpretation.",
        status="skipped" if ocr_result.get("skipped") else "ok",
        context_inputs=["frames", "caption-confidence prescreen when available"],
        context_outputs=[
            f"{ocr_result.get('non_empty', 0)} frames with OCR text",
            "visible-text evidence for Qwen and final synthesis",
        ]
        if not ocr_result.get("skipped")
        else ["no OCR context"],
        risks=[
            "small or low-contrast text can be missed",
            "false OCR tokens can create wrong named-entity context",
            "prescreen skips may discard frames with useful text",
        ],
        artifacts=[],
    )

    # Step 07: Depth
    depth_result: dict[str, Any] = {"skipped": True, "depth_results": []}
    if args.depth:
        _step(7, _TOTAL_STEPS, "Depth estimation")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "O_depth"):
            depth_result = step_depth_estimation(frame_list, video_name, video_dir)
        knowledge.add_depth(depth_result.get("depth_results", []))
    else:
        T["O_depth"] = 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="07",
        title="Depth estimation",
        description="Estimate relative scene geometry for near/far reasoning and scene-structure cues.",
        status="skipped" if depth_result.get("skipped") else "ok",
        context_inputs=["frames"],
        context_outputs=[
            f"{depth_result.get('ok_count', 0)} depth-estimated frames",
            "relative geometry cues for later prompts",
        ]
        if not depth_result.get("skipped")
        else ["no depth context"],
        risks=[
            "monocular depth can confuse scale and elevation",
            "depth failure in low-texture scenes can misstate geometry",
            "wrong depth priors can bias later scene explanations",
        ],
        artifacts=[],
    )

    # Step 08: Detection — accumulate per-label object counts into context
    det_result: dict[str, Any] = {"skipped": True, "detection_results": []}
    if args.detection:
        _step(8, _TOTAL_STEPS, "Object detection")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "P_detection"):
            det_result = step_object_detection(frame_list, video_name, video_dir)
        knowledge.add_detections(det_result.get("detection_results", []))
    else:
        T["P_detection"] = 0.0
    if not det_result.get("skipped"):
        obj_counts: dict[str, int] = {}
        for _r in det_result.get("detection_results", []):
            for _d in _r.get("detections", []):
                lbl = _d.get("label", "unknown")
                obj_counts[lbl] = obj_counts.get(lbl, 0) + 1
        video_context["detections"] = obj_counts
    _append_agentic_step(
        agentic_trace,
        step_id="08",
        title="Object detection",
        description="Detect frame-level entities so later reasoning can reference concrete objects instead of only global scene text.",
        status="skipped" if det_result.get("skipped") else "ok",
        context_inputs=["frames"],
        context_outputs=[
            f"{det_result.get('total_objects', 0)} detected objects",
            f"top entities: {', '.join(knowledge.known_entities[:5]) or 'none'}",
        ]
        if not det_result.get("skipped")
        else ["no detection context"],
        risks=[
            "class confusion can misidentify critical objects",
            "open-vocabulary labels can drift semantically across frames",
            "false positives can become persistent agentic context",
        ],
        artifacts=[],
    )

    # Step 09: YOLO11 + SAM2/3 detection and segmentation
    yolo_sam_result: dict[str, Any] = {"skipped": True, "detection_results": []}
    if not getattr(args, "no_yolo", False):
        _step(9, _TOTAL_STEPS, "YOLO11 + SAM2/3 detection → yolo_sam/ + detection_comparison.md")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "P2_yolo_sam"):
            yolo_sam_result = step_yolo_sam_detection(
                frame_list,
                video_name,
                video_dir,
                device,
                det_result=det_result,
            )
        if not yolo_sam_result.get("skipped"):
            knowledge.add_detections(yolo_sam_result.get("detection_results", []))
    else:
        T["P2_yolo_sam"] = 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="09",
        title="YOLO11 + SAM2/3 detection and segmentation",
        description=(
            "Run YOLO11 for fast instance detection with priority-ordered output "
            "(human > vehicle > artificial > other), optionally refined with SAM2/3 "
            "segmentation masks. Produces annotated frames and a comparison artifact "
            "against the HF detector (step 08)."
        ),
        status="skipped" if yolo_sam_result.get("skipped") else "ok",
        context_inputs=["frames", "HF detection results from step 08"],
        context_outputs=[
            f"{yolo_sam_result.get('total_objects', 0)} YOLO detections",
            f"human={yolo_sam_result.get('human_count', 0)} vehicle={yolo_sam_result.get('vehicle_count', 0)} artificial={yolo_sam_result.get('artificial_count', 0)}",
            "annotated frames + JSON results + comparison.md",
        ]
        if not yolo_sam_result.get("skipped")
        else ["no YOLO context"],
        risks=[
            "YOLO class confusion can misidentify humans as objects (safety-critical)",
            "priority ordering treats all persons equally regardless of role",
            "SAM masks may bleed across object boundaries in cluttered frames",
            "comparison vs HF detector may hide YOLO-specific failure modes",
        ],
        artifacts=[
            "yolo_sam_results.json",
            "yolo_sam/frame_*_annotated.jpg",
            "detection_comparison.md",
        ]
        if not yolo_sam_result.get("skipped")
        else [],
    )

    # Step 10: Gemma 4 directed tracking — Gemma understands the scene, directs SAM to
    # segment named objects, RF-DETR tracks those objects across the full sequence.
    gemma_tracking_result: dict[str, Any] = {"skipped": True}
    _gemma_api_url_p3 = getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL
    _gemma_api_model_p3 = getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL
    if not getattr(args, "no_rfdetr", False) and _gemma_api_url_p3:
        _step(
            10,
            _TOTAL_STEPS,
            "Gemma 4 directed tracking → gemma_tracking/ + gemma_tracking_results.json",
        )
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "P3_gemma_tracking"):
            gemma_tracking_result = step_gemma_directed_tracking(
                frame_list,
                video_name,
                video_dir,
                device,
                models=models,
                gemma_api_url=_gemma_api_url_p3,
                gemma_api_model=_gemma_api_model_p3,
                precomputed_scene=video_context.get("gemma_structured_scene"),
            )
    else:
        T["P3_gemma_tracking"] = 0.0
        if not _gemma_api_url_p3:
            _log.info("  Step 10 skipped (no gemma_api_url configured)")
    _append_agentic_step(
        agentic_trace,
        step_id="10",
        title="Gemma 4 directed tracking",
        description=(
            "Gemma 4 watches sampled frames and produces structured JSON: scene type, "
            "dominant object categories with rough bounding boxes, and a priority-ordered "
            "tracking list. SAM uses Gemma's bboxes as direct box prompts (Path A) or "
            "falls back to CLIP-filtered auto-masks (Path B). RF-DETR then tracks "
            "Gemma-priority classes across the full frame sequence with persistent track IDs."
        ),
        status="skipped" if gemma_tracking_result.get("skipped") else "ok",
        context_inputs=[
            "sampled frames",
            "Gemma sidecar API output",
            "CLIP embeddings for SAM mask filtering",
        ],
        context_outputs=[
            f"scene_type={gemma_tracking_result.get('scene_type', 'n/a')}",
            f"{gemma_tracking_result.get('n_tracked_objects', 0)} unique track IDs",
            f"{gemma_tracking_result.get('sam_masks_total', 0)} SAM masks",
            "gemma_tracking_results.json + annotated frames + summary.md",
        ]
        if not gemma_tracking_result.get("skipped")
        else ["no Gemma tracking context"],
        risks=[
            "Gemma JSON parse failure silently falls back to no-op (empty target_labels)",
            "rough_bbox from Gemma may not align precisely — SAM mask may bleed",
            "CLIP-filtered auto-mask path adds latency; disable with --no-sam to skip",
            "RF-DETR tracking IDs reset per video; no cross-video identity",
            "Gemma object labels may not match RF-DETR COCO vocabulary exactly",
        ],
        artifacts=[
            "gemma_tracking_results.json",
            "gemma_tracking/frame_*_tracked.jpg",
            "gemma_tracking_summary.md",
        ]
        if not gemma_tracking_result.get("skipped")
        else [],
    )

    # Submit the 3D-map build once sparse geometry can be enriched with the
    # already-computed depth, detection, and tracking cues. The CPU-bound map
    # stage still overlaps with the slower VLM/API stages that follow.
    _sfm_min_dur = float(settings.SFM_MIN_DURATION_SEC)
    _clip_dur = float(stats.get("duration_sec", 0.0))
    _run_sfm = not args.no_sfm
    if _run_sfm and _sfm_min_dur > 0 and _clip_dur < _sfm_min_dur:
        _log.info(
            "  SfM skipped: clip %.1fs < SFM_MIN_DURATION_SEC=%.0fs — using pseudo-3D fallback",
            _clip_dur,
            _sfm_min_dur,
        )
        _run_sfm = False
    _log.info("  -> Submitting 3D-map step 16 to background thread (SfM+enrichment+Splat) …")
    _map_future = _map_executor.submit(
        step_create_3d_map,
        video_path,
        video_id,
        video_dir,
        frame_list,
        models,
        run_sfm_flag=_run_sfm,
        run_gsplat_flag=not getattr(args, "no_gsplat", False),
        device=device,
        depth_results=depth_result.get("depth_results", []),
        yolo_detection_results=yolo_sam_result.get("detection_results", []),
        tracking_results=gemma_tracking_result,
    )

    # Step 11: World model
    world_result: dict[str, Any] = {"skipped": True, "world_results": []}
    if args.world_model:
        _step(11, _TOTAL_STEPS, "World model video embeddings")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "Q_world"):
            world_result = step_world_model_pass(frame_list, video_name, video_dir, models=models)
    else:
        T["Q_world"] = 0.0
    if not world_result.get("skipped"):
        video_context["world_model_clips"] = world_result.get("ok_count", 0)
    _append_agentic_step(
        agentic_trace,
        step_id="11",
        title="World model pass",
        description="Compress clips into temporal embeddings to capture motion-level context not visible in single frames.",
        status="skipped" if world_result.get("skipped") else "ok",
        context_inputs=["ordered frame clips"],
        context_outputs=[
            f"{world_result.get('ok_count', 0)} temporal clip embeddings",
            "coarse motion-context signal",
        ]
        if not world_result.get("skipped")
        else ["no temporal clip context"],
        risks=[
            "clip pooling can smooth away rare but important events",
            "temporal embeddings are hard to interpret and easy to overtrust",
            "wrong clip-level context can bias synthesis without clear provenance",
        ],
        artifacts=[],
    )

    # Step 12: Qwen — uses ASR + OCR context from previous steps (already agentic)
    qwen_result: dict[str, Any] = {"skipped": True, "results": []}
    if args.qwen:
        _step(12, _TOTAL_STEPS, "Qwen VLM detailed captioning → detailed_captions.md")
        with _Timer(T, "R_qwen"):
            qwen_result = step_qwen_captioning(
                frame_list,
                video_name,
                video_dir,
                subtitle_map=asr_result.get("subtitle_map", {}),
                ocr_results=ocr_result.get("ocr_results", []),
                # Pass a passthrough so QwenModel never creates a second CLIP
                # embedder (OpenCLIPTagger) that competes for VRAM. In local mode
                # mode we want full coverage; prescreening is not needed.
                clip_prescreen_fn=lambda _img: True,
                knowledge=knowledge,
            )
    else:
        T["R_qwen"] = 0.0
    if not qwen_result.get("skipped"):
        video_context["qwen_captions"] = qwen_result.get("results", [])
    _append_agentic_step(
        agentic_trace,
        step_id="12",
        title="Qwen detailed captioning",
        description="Fuse visual frames with accumulated Florence, ASR, OCR, depth, detections, and prior-Qwen state for structured per-frame reasoning.",
        status="skipped" if qwen_result.get("skipped") else "ok",
        context_inputs=[
            "frame image",
            "Florence scene priors",
            "ASR-aligned subtitle context",
            "OCR/depth/detection cues",
            "previous Qwen structured state",
        ],
        context_outputs=[
            f"{qwen_result.get('ok_count', 0)} detailed captions",
            "structured scene facts for downstream synthesis",
            "updated prior-state chain across frames",
        ]
        if not qwen_result.get("skipped")
        else ["no detailed reasoning context"],
        risks=[
            "upstream misidentification compounds inside one prompt",
            "previous-frame state can anchor the model to stale or wrong context",
            "rich prompt context can make uncertain claims look internally consistent",
        ],
        artifacts=["detailed_captions.md"] if not qwen_result.get("skipped") else [],
    )

    # Step 13: UniDriveVLA expert analysis — compare understanding/perception/planning
    unidrive_result: dict[str, Any] = {"skipped": True, "results": []}
    if getattr(args, "unidrive", False):
        _step(13, _TOTAL_STEPS, "UniDriveVLA expert analysis → unidrive_analysis.md")
        with _Timer(T, "S_unidrive"):
            unidrive_result = step_unidrive_analysis(
                frame_list,
                video_name,
                video_dir,
                subtitle_map=asr_result.get("subtitle_map", {}),
                ocr_results=ocr_result.get("ocr_results", []),
                knowledge=knowledge,
            )
    else:
        _step(13, _TOTAL_STEPS, "UniDriveVLA expert analysis (skipped — pass --unidrive to enable)")
        T["S_unidrive"] = 0.0
    if not unidrive_result.get("skipped"):
        video_context["unidrive_analysis"] = unidrive_result.get("results", [])
    _append_agentic_step(
        agentic_trace,
        step_id="13",
        title="UniDriveVLA expert analysis",
        description="Run an external UniDriveVLA bridge for understanding, perception, planning, and mixture-of-experts consensus on sampled frames.",
        status="skipped" if unidrive_result.get("skipped") else "ok",
        context_inputs=[
            "sampled frames",
            "ASR/OCR context when available",
            "agentic context from earlier steps",
        ],
        context_outputs=[
            f"{unidrive_result.get('ok_count', 0)} UniDrive analyses",
            "understanding/perception/planning triplets",
            "mixture-of-experts consensus summaries",
        ]
        if not unidrive_result.get("skipped")
        else ["no UniDrive context"],
        risks=[
            "external bridge can expose a different ontology than existing steps",
            "planning advice may be overconfident for non-driving footage",
            "expert consensus can hide meaningful disagreement if prompts are too generic",
        ],
        artifacts=["unidrive_analysis.md"] if not unidrive_result.get("skipped") else [],
    )

    if any(
        [
            args.asr,
            args.ocr,
            args.depth,
            args.detection,
            args.world_model,
            args.qwen,
            getattr(args, "unidrive", False),
        ]
    ):
        _mm_md = video_dir / "multimodal_features.md"
        write_multimodal_md(
            _mm_md,
            video_name,
            asr_result,
            ocr_result,
            depth_result,
            det_result,
            world_result,
            platform_fusion_result,
            qwen_result,
            unidrive_result,
        )

    # Step 14: SceneTok streaming scene encoder + segmentation decoder
    scenetok_result: dict[str, Any] = {"skipped": True}
    if getattr(args, "scenetok", False):
        _step(
            14,
            _TOTAL_STEPS,
            "SceneTok streaming encoder + segmentation decoder → scenetok_tokens.npz",
        )
        _scenetok_api_url = getattr(args, "scenetok_api_url", "") or settings.SCENETOK_API_URL
        _scenetok_checkpoint = (
            getattr(args, "scenetok_checkpoint", "") or settings.SCENETOK_CHECKPOINT
        )
        if _scenetok_api_url:
            import os as _os

            _os.environ.setdefault("SCENETOK_API_URL", _scenetok_api_url)
        if _scenetok_checkpoint:
            import os as _os

            _os.environ.setdefault("SCENETOK_CHECKPOINT", _scenetok_checkpoint)
        with _Timer(T, "S_scenetok"):
            scenetok_result = step_scenetok(
                frame_list,
                video_dir,
                checkpoint=_scenetok_checkpoint,
                mode=settings.SCENETOK_MODE,
            )
    else:
        _step(14, _TOTAL_STEPS, "SceneTok (skipped — pass --scenetok to enable)")
        T["S_scenetok"] = 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="14",
        title="SceneTok scene compression + segmentation",
        description=(
            "Encode the sampled frame sequence into compact permutation-invariant scene tokens "
            "via the SceneTok multi-view encoder, then decode each frame to a segmentation mask "
            "or novel-view render using the rectified flow decoder."
        ),
        status="skipped" if scenetok_result.get("skipped") else "ok",
        context_inputs=["sampled keyframes"],
        context_outputs=[
            f"{scenetok_result.get('n_tokens', 0)} scene tokens",
            f"{scenetok_result.get('n_frames', 0)} decoded frames",
        ]
        if not scenetok_result.get("skipped")
        else ["no SceneTok context"],
        risks=[
            "base checkpoint outputs RGB novel views, not masks — requires a fine-tuned segmentation checkpoint for mask mode",
            "token compression may drop subtle or transient scene elements",
            "~24 GB VRAM required for local inference; sidecar mode recommended on single-GPU setups",
        ],
        artifacts=(
            ["scenetok_tokens.npz", "scenetok_masks/"]
            if not scenetok_result.get("skipped") and settings.SCENETOK_MODE == "masks"
            else (
                ["scenetok_tokens.npz", "scenetok_views/"]
                if not scenetok_result.get("skipped")
                else []
            )
        ),
    )

    # Step 15: Cosmos3 omnimodal world-model inference
    cosmos3_result: dict[str, Any] = {"skipped": True, "clips": [], "n_clips": 0}
    _cosmos3_enabled = getattr(args, "cosmos3", None)
    if _cosmos3_enabled:
        from ...steps.cosmos3 import step_cosmos3_inference

        _step(15, _TOTAL_STEPS, "Cosmos3 world-model inference → cosmos3_inference.json")
        _prep_vram_for_step(models, device)
        with _Timer(T, "S_cosmos3"):
            cosmos3_result = step_cosmos3_inference(
                frame_list,
                video_name,
                video_dir,
                device=device,
            )
    else:
        _step(15, _TOTAL_STEPS, "Cosmos3 inference (skipped — pass --cosmos3 to enable)")
        T["S_cosmos3"] = 0.0
    if not cosmos3_result.get("skipped"):
        video_context["cosmos3_analysis"] = cosmos3_result.get("clips", [])
    _append_agentic_step(
        agentic_trace,
        step_id="15",
        title="Cosmos3 world-model inference",
        description=(
            "Run nvidia/Cosmos3-Nano (or vLLM-Omni sidecar) on sampled video clips to produce "
            "an omnimodal scene understanding narrative: environment type, entity states, "
            "temporal dynamics, and safety observations."
        ),
        status="skipped" if cosmos3_result.get("skipped") else "ok",
        context_inputs=["sampled frame clips", "physical-AI analysis prompt"],
        context_outputs=[
            f"{cosmos3_result.get('n_clips', 0)} clip analyses",
            f"scene_type={cosmos3_result.get('scene_type', '')}",
            f"mode={cosmos3_result.get('mode', '')}",
        ]
        if not cosmos3_result.get("skipped")
        else ["no Cosmos3 world-model context"],
        risks=[
            "omnimodal model may hallucinate entity interactions not present in frames",
            "~32 GB VRAM required for full local load; layerwise offload degrades throughput",
            "sidecar vLLM-Omni latency can be high for large frame batches",
            "model was trained on curated physical-AI data — may generalize poorly to niche domains",
        ],
        artifacts=["cosmos3_inference.json"] if not cosmos3_result.get("skipped") else [],
    )

    # Step 16: Base model search — restore CLIP+DINO to GPU before joining 3D-map thread
    # (I must be joined first so the background thread no longer accesses models).
    # Evict Ollama (reloaded during step 12) before restoring CLIP+DINO.
    if device == "cuda" and not clip_dino_on_gpu:
        _qwen_url = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
        _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
        _unidrive_url = getattr(args, "unidrive_api_url", "") or settings.UNIDRIVE_API_URL
        _unidrive_model = getattr(args, "unidrive_model", "") or settings.UNIDRIVE_MODEL
        _prep_vram_for_step(
            models,
            device,
            extra_sidecars=[
                (_qwen_url, _qwen_model),
                (_unidrive_url, _unidrive_model),
            ],
            label="base-search restore",
        )
        _restore_models_to_gpu(models, device)
        clip_dino_on_gpu = _models_on_device(models, device)
    _step(16, _TOTAL_STEPS, "Base model transformation test → base_search.md")
    with _Timer(T, "C_base_search"):
        c = step_base_model_search_test(
            frame_list, store, is_qdrant, models, video_id, video_name, video_dir, top_k=args.top_k
        )
    base_results = c["results"]
    query_frame = c["query_frame"]
    query_t_sec = c["query_t_sec"]
    stats["base_top_score"] = base_results[0]["score"] if base_results else 0.0
    _append_agentic_step(
        agentic_trace,
        step_id="16",
        title="Base search test",
        description="Measure retrieval behavior of the base model as the control reference for adaptation steps.",
        status="ok",
        context_inputs=["retrieval index", "query frame"],
        context_outputs=[
            f"top-{len(base_results)} baseline matches",
            f"query at {query_t_sec:.1f}s",
        ],
        risks=[
            "search quality may favor visual similarity over semantic identity",
            "one query frame can underrepresent broader retrieval behavior",
            "baseline errors can distort later before/after comparisons",
        ],
        artifacts=["base_search.md"],
    )

    # Step 16: 3D map + Gaussian Splat — collect background-thread result (Phase 2 close)
    _step(17, _TOTAL_STEPS, "3D map + Gaussian Splat → 3d_map/ (joining background thread)")
    with _Timer(T, "I_3dmap"):
        if _map_future is not None:
            try:
                h = _map_future.result(timeout=600)  # up to 10 min for SfM
            except Exception as _map_exc:
                _log.warning("  3D-map background thread raised: %s", _map_exc, exc_info=True)
                h = {
                    "sfm_poses": 0,
                    "method": "failed",
                    "points": None,
                    "gsplat_method": "failed",
                    "splat_ply": None,
                    "viewer_html": "",
                }
            finally:
                _map_executor.shutdown(wait=False)
        else:
            _map_executor.shutdown(wait=False)
            h = {
                "sfm_poses": 0,
                "method": "skipped",
                "points": None,
                "gsplat_method": "skipped",
                "splat_ply": None,
                "viewer_html": "",
            }
    T["I_3dmap"] = float(h.get("elapsed_sec", T.get("I_3dmap", 0.0)) or 0.0)
    stats["sfm_poses"] = h["sfm_poses"]
    stats["map_method"] = h["method"]
    stats["map_points"] = int(h["points"].shape[0]) if h.get("points") is not None else 0
    stats["gsplat_method"] = h.get("gsplat_method", "skipped")
    stats["map_degraded"] = bool(
        h.get("quality_degraded", stats["map_points"] < 50 or stats["sfm_poses"] < 20)
    )
    if stats["map_degraded"]:
        _log.warning(
            "3D map quality is degraded: %d points, %d SfM poses%s",
            stats["map_points"],
            stats["sfm_poses"],
            (
                f", {int(len(h.get('frame_positions') or []))} total anchors"
                if len(h.get("frame_positions") or []) > stats["sfm_poses"]
                else ""
            ),
        )
    stats["splat_ply"] = h.get("splat_ply")
    semantic_graph_result: dict[str, Any] = {"skipped": True}
    if not getattr(args, "no_yolo", False) and settings.YOLO_SSG_ENABLED:
        semantic_graph_result = step_build_semantic_environment_graph(
            video_id=video_id,
            video_name=video_name,
            video_dir=video_dir,
            yolo_sam_result=yolo_sam_result,
            map_result=h,
        )
    stats["semantic_graph_nodes"] = (
        semantic_graph_result.get("graph", {}).get("summary", {}).get("node_count", 0)
        if not semantic_graph_result.get("skipped")
        else 0
    )
    stats["semantic_graph_edges"] = (
        semantic_graph_result.get("graph", {}).get("summary", {}).get("edge_count", 0)
        if not semantic_graph_result.get("skipped")
        else 0
    )
    map_quality_advisor = step_advise_3d_map_quality(
        video_path=video_path,
        video_dir=video_dir,
        frame_list=frame_list,
        map_result=h,
        caption_results=caption_results,
        tracking_results=gemma_tracking_result,
    )
    stats["map_quality_advisor"] = map_quality_advisor
    advisor_summary = map_quality_advisor.get("summary", {}) or {}
    advisor_issues = advisor_summary.get("issues", []) or []
    if advisor_issues:
        _log.info("  Map advisor: %s", advisor_issues[0])
    if h.get("splat_ply"):
        _log.info("  [ok] Gaussian Splat → %s", h["splat_ply"])
        _log.info("  [ok] Interactive viewer → %s", h.get("viewer_html", ""))
    video_context["map"] = {
        "method": h["method"],
        "points": stats["map_points"],
        "sfm_poses": h["sfm_poses"],
        "gsplat_method": stats["gsplat_method"],
        "splat_ply": stats["splat_ply"],
        "semantic_graph": semantic_graph_result.get("graph", {}).get("summary", {}),
    }
    _append_agentic_step(
        agentic_trace,
        step_id="17",
        title="3D map creation",
        description="Recover scene geometry and export sparse-map or splat artifacts for spatial interpretation (ran concurrently with steps M–R).",
        status="ok" if h["method"] not in ("failed", "skipped") else h["method"],
        context_inputs=["video frames", "camera-motion consistency"],
        context_outputs=[
            f"{stats['map_points']} map points",
            f"{stats['sfm_poses']} SfM poses",
            f"map method {stats['map_method']}",
            f"{stats['semantic_graph_nodes']} semantic nodes",
        ],
        risks=[
            "geometry failure can create confident but wrong spatial context",
            "SfM fallback outputs may look valid while lacking metric truth",
            "map artifacts can be overinterpreted as semantic evidence",
        ],
        artifacts=[
            "3d_map/sparse_map.npz",
            "3d_map/map_stats.json",
            "3d_map/semantic_environment_graph.json",
            "3d_map/semantic_environment_graph.md",
        ]
        if not semantic_graph_result.get("skipped")
        else ["3d_map/sparse_map.npz", "3d_map/map_stats.json"],
    )

    # -- Full probabilistic state fusion (all four layers) ---------------------
    from ...steps.fusion import step_full_state_fusion

    # Collect RSSM surprise mean (from world model result if available)
    _rssm_mean: float | None = None
    if world_result and not world_result.get("skipped"):
        _rssm_scores = world_result.get("rssm_scores") or []
        if _rssm_scores:
            _rssm_mean = float(sum(_rssm_scores) / len(_rssm_scores))

    # Collect Qwen structured captions
    _qwen_captions = (
        qwen_result.get("structured_captions") or [] if not qwen_result.get("skipped") else []
    )

    # Collect Gemma structured analysis. Step 03 nests the structured scene in
    # task_results; step 10 has the stronger tracking-specific scene if it reran
    # vision analysis. Use the strongest structured scene available so semantic
    # priors do not fall back to "unknown".
    _gemma_info = j if not j.get("skipped") else None
    _structured_scene = (
        (
            video_context.get("gemma_structured_scene")
            or (j.get("task_results", {}) or {}).get("structured_scene_summary")
            or j.get("structured_scene_summary")
            or j.get("structured_scene")
        )
        if not j.get("skipped")
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

    with _Timer(T, "PS_full_fusion"):
        full_fusion_result = step_full_state_fusion(
            video_path=video_path,
            frame_list=frame_list,
            video_name=video_name,
            video_dir=video_dir,
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
    T["PS_full_fusion"] = T.get("PS_full_fusion", 0.0)
    stats["full_fusion_tracks"] = full_fusion_result.get("track_count", 0)
    stats["full_fusion_scene"] = full_fusion_result.get("scene_type", "unknown")
    video_context["full_state_fusion"] = full_fusion_result.get("summary", {})

    _step(18, _TOTAL_STEPS, "Physical scene state summary → physical_state_summary.json")
    from ...steps.physical_state import step_physical_state as _step_physical_state

    with _Timer(T, "PS_physical_state"):
        physical_state_result = _step_physical_state(
            full_fusion_result=full_fusion_result,
            depth_result=depth_result,
            gemma_tracking_result=gemma_tracking_result,
            yolo_sam_result=yolo_sam_result,
            frame_list=frame_list,
            video_dir=video_dir,
            video_name=video_name,
        )
    knowledge.add_physical_state(physical_state_result)
    video_context["physical_state"] = physical_state_result

    _step(19, _TOTAL_STEPS, "Environmental field state → field_state_summary.json")
    from ...steps.field_state import step_field_state as _step_field_state

    with _Timer(T, "PS_field_state"):
        field_state_result = _step_field_state(
            video_path=video_path,
            video_dir=video_dir,
            video_name=video_name,
            frame_list=frame_list,
            depth_result=depth_result,
            physical_state_result=physical_state_result,
            caption_results=caption_results,
            unidrive_result=unidrive_result,
        )
    video_context["field_state"] = field_state_result

    _step(20, _TOTAL_STEPS, "Threat primitives → threat_primitives.json")
    from ...steps.threat_primitives import step_threat_primitives as _step_threat_primitives

    with _Timer(T, "PS_threat_primitives"):
        threat_primitives_result = _step_threat_primitives(
            physical_state_result=physical_state_result,
            field_state_result=field_state_result,
            depth_result=depth_result,
            caption_results=caption_results,
            unidrive_result=unidrive_result,
            gemma_tracking_result=gemma_tracking_result,
            full_fusion_result=full_fusion_result,
            frame_list=frame_list,
            sfm_poses=int(stats.get("sfm_poses", 0)),
            map_degraded=bool(stats.get("map_degraded", False)),
            video_dir=video_dir,
            video_name=video_name,
        )
    video_context["threat_primitives"] = threat_primitives_result

    # -- Phase 3: SSL-gated adaptation (SSL gate required) ---------------------
    # Step 16 (SSL fine-tuning) always runs to evaluate the gate.
    # Steps E, F, G, H only proceed when D produces a valid checkpoint with
    # best_loss < _SSL_GATE_MAX_LOSS. Steps 22 and 23 always run as finalization.
    _banner("Phase 3 — SSL-gated adaptation")

    # Step 16: SSL fine-tuning — DINOFineTuner loads its own separate DINO; offload ours first.
    # Skipped when using an API-based embedder — no local backbone to fine-tune.
    checkpoint_path = ""
    if models.get("uses_api_embedder"):
        T["D_finetune"] = 0.0
        T["E_distill"] = 0.0
        _step(21, _TOTAL_STEPS, "SSL DINOv3 fine-tuning (skipped — API embedder)")
        _step(22, _TOTAL_STEPS, "Knowledge distillation (skipped — API embedder)")
        student_backbone = None
        student_dim = 768
    else:
        if device == "cuda":
            _prep_vram_for_step(
                models,
                device,
                extra_sidecars=[
                    (
                        getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL,
                        getattr(args, "qwen_model", "") or settings.QWEN_MODEL,
                    ),
                    (
                        getattr(args, "unidrive_api_url", "") or settings.UNIDRIVE_API_URL,
                        getattr(args, "unidrive_model", "") or settings.UNIDRIVE_MODEL,
                    ),
                ],
                label="SSL fine-tuning",
            )
            _guard_min_free_vram("SSL fine-tuning")
            clip_dino_on_gpu = False
        _step(21, _TOTAL_STEPS, "SSL DINOv3 fine-tuning → finetune_stats.md")
        # Adaptive epoch count: scale up for short clips so the training sees
        # ~200 gradient steps regardless of frame count.  Short homogeneous
        # videos (< 100 frames) have few batches per epoch and under-train at
        # the default 3 epochs; longer videos converge faster and need fewer.
        # Hard ceiling of 20 prevents runaway training on very short clips.
        _n_batches_per_epoch = max(1, len(frame_list) // max(1, args.batch_size))
        _ssl_epochs = max(
            args.epochs, min(20, (200 + _n_batches_per_epoch - 1) // _n_batches_per_epoch)
        )
        if _ssl_epochs != args.epochs:
            _log.info(
                "  SSL adaptive epochs: %d (CLI default=%d) — %d frames, %d batches/epoch",
                _ssl_epochs,
                args.epochs,
                len(frame_list),
                _n_batches_per_epoch,
            )
        with _Timer(T, "D_finetune"):
            d = step_ssl_finetune(
                video_id,
                video_name,
                video_dir,
                frame_list,
                device,
                epochs=_ssl_epochs,
                batch_size=args.batch_size,
                tracking_results=(
                    gemma_tracking_result.get("tracking_results") or []
                    if not gemma_tracking_result.get("skipped")
                    else []
                ),
                depth_result=depth_result,
                platform_state_fusion=platform_fusion_result,
                full_fusion_result=full_fusion_result,
                physical_state_result=physical_state_result,
            )
        stats["best_loss"] = d["best_loss"]
        stats["ckpt_mb"] = d["ckpt_mb"]
        checkpoint_path = d["checkpoint"]
        _append_agentic_step(
            agentic_trace,
            step_id="18",
            title="SSL fine-tuning",
            description="Adapt the local DINO backbone to mission-specific footage so retrieval neighborhoods reflect this video domain more closely.",
            status="ok",
            context_inputs=["frame sequence", "base DINO initialization"],
            context_outputs=[
                f"best loss {d['best_loss']:.4f}",
                "mission-adapted backbone checkpoint",
            ],
            risks=[
                "small-video adaptation can overfit to accidental patterns",
                "temporal positives can encode wrong sameness assumptions",
                "adapted features can improve scores while harming semantics",
            ],
            artifacts=["finetune_stats.md", "checkpoints/dino_ssl_best.pt"],
        )

        # -- SSL gate: only proceed to E/F/G/H if D produced a usable checkpoint --
        import os as _os

        _ssl_best_loss = stats.get("best_loss", float("inf"))
        ssl_gate_passed = (
            bool(checkpoint_path)
            and _os.path.exists(checkpoint_path)
            and _ssl_best_loss < _SSL_GATE_MAX_LOSS
        )
        if ssl_gate_passed:
            _log.info(
                "  [ok] SSL gate passed (best_loss=%.4f < %.1f) — proceeding to distillation, "
                "ONNX export, and search comparison",
                _ssl_best_loss,
                _SSL_GATE_MAX_LOSS,
            )
        else:
            _log.warning(
                "  ✗ SSL gate did not pass (checkpoint=%r, best_loss=%.4f, threshold=%.1f) — "
                "skipping steps E/F/G/H (distillation, ONNX export, search comparison)",
                checkpoint_path,
                _ssl_best_loss,
                _SSL_GATE_MAX_LOSS,
            )

        # Step 17: Distillation — maximum-hydration chain (Gemma teacher + caption anchor when available)
        student_backbone = None
        student_dim = 768
        if ssl_gate_passed and not args.no_distill:
            # Build caption anchor embeddings from Florence captions via CLIP text encoder
            _cap_anchor_embs: np.ndarray | None = None
            _scene_captions = caption_results  # set by step_scene_captioning (step 04)
            if _scene_captions and models.get("clip"):
                try:
                    _cap_texts = [r.get("caption") or "" for r in _scene_captions]
                    _cap_texts = [t for t in _cap_texts if t.strip()]
                    if _cap_texts:
                        _clip_model = models["clip"]
                        # Only OpenCLIPEmbedder has encode_texts suitable for anchoring
                        if hasattr(_clip_model, "encode_texts") and not isinstance(
                            _clip_model, GemmaEmbedder if _HAS_GEMMA else type(None)
                        ):
                            _cap_anchor_embs = _clip_model.encode_texts(_cap_texts)
                            _log.info(
                                "  Distillation caption anchors: %d CLIP text embeddings from Florence captions",
                                len(_cap_anchor_embs),
                            )
                except Exception as _exc:
                    _log.debug(
                        "  Caption anchor prep failed (%s) — distilling without anchor", _exc
                    )

            # Use Gemma embedder as teacher when loaded and MODEL_NAME=gemma
            _gemma_teacher = None
            if _HAS_GEMMA and isinstance(models.get("clip"), GemmaEmbedder):
                _gemma_teacher = models["clip"]
                _log.info("  Using GemmaVisionTeacher for distillation (max hydration)")

            _step(22, _TOTAL_STEPS, "Knowledge distillation (max hydration) → ViT-S/14 student")
            with _Timer(T, "E_distill"):
                e_distill = step_distill(
                    checkpoint_path,
                    frame_list,
                    video_name,
                    video_dir,
                    device,
                    distill_epochs=args.distill_epochs,
                    batch_size=args.batch_size,
                    caption_embeddings=_cap_anchor_embs,
                    gemma_embedder=_gemma_teacher,
                )
            if not e_distill["skipped"]:
                student_backbone = e_distill["student_backbone"]
                student_dim = e_distill["student_dim"]
                stats["distill_loss"] = e_distill["best_loss"]
                stats["student_ckpt_mb"] = e_distill["ckpt_mb"]
                stats["student_dim"] = student_dim
                stats["teacher_dim"] = e_distill["teacher_dim"]
                stats["distill_compression_ratio"] = e_distill.get("compression_ratio", 0.0)
            _append_agentic_step(
                agentic_trace,
                step_id="19",
                title="Knowledge distillation",
                description="Compress teacher geometry and optional language anchors into a smaller student suitable for deployment.",
                status="skipped" if e_distill.get("skipped") else "ok",
                context_inputs=[
                    "fine-tuned teacher checkpoint",
                    "optional Gemma teacher alignment",
                    "optional Florence caption anchors",
                ],
                context_outputs=[
                    f"student dim {student_dim}",
                    f"best distill loss {e_distill.get('best_loss', float('nan')):.4f}",
                    "student deployment checkpoint",
                ]
                if not e_distill.get("skipped")
                else ["no distilled student"],
                risks=[
                    "teacher mistakes transfer into the student representation",
                    "caption anchors can inject wrong semantics into retrieval space",
                    "compression can erase rare but important distinctions",
                ],
                artifacts=["distill_stats.md", "checkpoints/student_best.pt"]
                if not e_distill.get("skipped")
                else [],
            )
        else:
            T["E_distill"] = 0.0
            _gate_reason = "SSL gate did not pass" if not ssl_gate_passed else "--no-distill"
            _step(22, _TOTAL_STEPS, f"Knowledge distillation (skipped — {_gate_reason})")
            _append_agentic_step(
                agentic_trace,
                step_id="19",
                title="Knowledge distillation",
                description="Compress teacher knowledge into a smaller deployable student.",
                status="skipped",
                context_inputs=["fine-tuned teacher checkpoint"],
                context_outputs=["no distilled student"],
                risks=[
                    "without this step, deployment relies on the larger teacher or ONNX export only",
                    "no compression audit is produced",
                ],
                artifacts=[],
            )
    if models.get("uses_api_embedder"):
        ssl_gate_passed = False
        student_backbone = None
        student_dim = 768
        _append_agentic_step(
            agentic_trace,
            step_id="18",
            title="SSL fine-tuning",
            description="Adapt the local DINO backbone to mission-specific footage.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no local fine-tuning checkpoint"],
            risks=[
                "no task-specific adaptation is learned in API-embedder mode",
            ],
            artifacts=[],
        )
        _append_agentic_step(
            agentic_trace,
            step_id="19",
            title="Knowledge distillation",
            description="Compress teacher knowledge into a smaller deployable student.",
            status="skipped",
            context_inputs=["API embedder mode"],
            context_outputs=["no distillation artifacts"],
            risks=[
                "no student compression path is available in API-embedder mode",
            ],
            artifacts=[],
        )

    # Step 18b: Stage 2 distillation — ViT-S/14 → EfficientViT-B1 (RKD-D + KoLeo only)
    # Runs only when Stage 1 produced a student backbone and --no-distill is not set.
    e_distill_stage2: dict[str, Any] = {"skipped": True, "onnx_exported": False}
    if ssl_gate_passed and not args.no_distill and not e_distill.get("skipped"):
        _step(23, _TOTAL_STEPS, "Stage 2 distillation: ViT-S/14 → EfficientViT-B1 student")
        with _Timer(T, "E_distill_stage2"):
            e_distill_stage2 = step_distill_stage2(
                student_backbone,
                frame_list,
                video_name,
                video_dir,
                device,
                distill_epochs=args.distill_epochs,
                batch_size=args.batch_size,
            )
        if not e_distill_stage2["skipped"]:
            stats["distill_stage2_loss"] = e_distill_stage2.get("best_loss", float("nan"))
            stats["efficientvit_ckpt_mb"] = e_distill_stage2.get("ckpt_mb", 0.0)
            stats["efficientvit_onnx_mb"] = e_distill_stage2.get("onnx_mb", 0.0)
        _append_agentic_step(
            agentic_trace,
            step_id="19b",
            title="Stage 2 distillation (EfficientViT-B1)",
            description="Compress the Stage 1 ViT-S/14 student into a 384-dim EfficientViT-B1 student using RKD-D + KoLeo. Requires only ~2 GB VRAM.",
            status="skipped" if e_distill_stage2.get("skipped") else "ok",
            context_inputs=["Stage 1 ViT-S/14 student backbone", "frame paths"],
            context_outputs=[
                "EfficientViT-B1 dim=384",
                f"best_loss={e_distill_stage2.get('best_loss', float('nan')):.4f}",
                f"onnx_exported={e_distill_stage2.get('onnx_exported', False)}",
            ]
            if not e_distill_stage2.get("skipped")
            else ["no Stage 2 student"],
            risks=[
                "Stage 2 adds a second compression hop — errors from Stage 1 compound",
                "RKD-D-only may underfit when teacher/student topologies diverge",
            ],
            artifacts=[
                "distill_stage2_stats.md",
                "checkpoints_stage2/student_best.pt",
                "edge_models/efficientvit_local.onnx",
            ]
            if not e_distill_stage2.get("skipped")
            else [],
        )
    else:
        T["E_distill_stage2"] = 0.0
        _gate_reason_s2 = (
            "SSL gate did not pass"
            if not ssl_gate_passed
            else "--no-distill"
            if args.no_distill
            else "no Stage 1 student"
        )
        _step(23, _TOTAL_STEPS, f"Stage 2 distillation (skipped — {_gate_reason_s2})")
        _append_agentic_step(
            agentic_trace,
            step_id="19b",
            title="Stage 2 distillation (EfficientViT-B1)",
            description="Compress Stage 1 student into EfficientViT-B1 using RKD-D + KoLeo.",
            status="skipped",
            context_inputs=["Stage 1 student backbone"],
            context_outputs=["no EfficientViT student"],
            risks=["no ultra-lightweight deployment artifact produced"],
            artifacts=[],
        )

    # Step 19: ONNX export + gallery — restore CLIP+DINO (export uses models["dino"])
    # Skipped when SSL gate did not pass (no valid checkpoint to package).
    if ssl_gate_passed:
        if device == "cuda" and not clip_dino_on_gpu:
            _restore_models_to_gpu(models, device)
            clip_dino_on_gpu = _models_on_device(models, device)
        _step(24, _TOTAL_STEPS, "ONNX export + gallery build → edge_models/")
        with _Timer(T, "F_export"):
            e = step_export_model(
                checkpoint_path,
                frame_list,
                video_dir,
                device,
                models,
                no_onnx=args.no_onnx,
                student_backbone=student_backbone,
                student_dim=student_dim,
            )
        stats["onnx_mb"] = e.get("onnx_mb", 0.0)
        stats["onnx_exported"] = e.get("exported", False)
        _append_agentic_step(
            agentic_trace,
            step_id="20",
            title="ONNX export",
            description="Package the best available backbone and gallery into deployment artifacts.",
            status="ok",
            context_inputs=["teacher or student backbone", "retrieval gallery frames"],
            context_outputs=[
                f"onnx exported={e.get('exported', False)}",
                "gallery.npz for edge classification",
            ],
            risks=[
                "export mismatches can change runtime behavior versus training",
                "gallery coverage can be too narrow for field use",
                "deployment artifacts can hide upstream semantic errors behind good latency",
            ],
            artifacts=["edge_models/dino_local.onnx", "edge_models/gallery.npz"],
        )
    else:
        T["F_export"] = 0.0
        stats.setdefault("onnx_mb", 0.0)
        stats.setdefault("onnx_exported", False)
        _step(24, _TOTAL_STEPS, "ONNX export (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="20",
            title="ONNX export",
            description="Package the best available backbone and gallery into deployment artifacts.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no deployment artifacts"],
            risks=["no edge deployment artifacts produced"],
            artifacts=[],
        )

    # Step 19: Fine-tuned search — only if SSL gate passed (needs fine-tuned or distilled backbone)
    ft_results: list[dict] = []
    if ssl_gate_passed:
        _step(25, _TOTAL_STEPS, "Fine-tuned model transformation test → finetuned_search.md")
        with _Timer(T, "G_ft_search"):
            f = step_finetuned_model_search_test(
                frame_list,
                store,
                is_qdrant,
                models,
                query_frame,
                query_t_sec,
                video_id,
                video_name,
                video_dir,
                top_k=args.top_k,
            )
        ft_results = f["results"]
        stats["ft_top_score"] = ft_results[0]["score"] if ft_results else 0.0
        _append_agentic_step(
            agentic_trace,
            step_id="21",
            title="Fine-tuned search test",
            description="Re-run retrieval after adaptation to quantify search-space changes.",
            status="ok",
            context_inputs=["fine-tuned or distilled backbone", "same query frame as baseline"],
            context_outputs=[
                f"top-{len(ft_results)} adapted matches",
                f"top score {stats['ft_top_score']:.4f}",
            ],
            risks=[
                "score improvements can hide semantic regressions",
                "query reuse can overstate adaptation gains",
                "retrieval differences may reflect memorization rather than better context",
            ],
            artifacts=["finetuned_search.md"],
        )
    else:
        T["G_ft_search"] = 0.0
        stats.setdefault("ft_top_score", 0.0)
        _step(25, _TOTAL_STEPS, "Fine-tuned search (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="21",
            title="Fine-tuned search test",
            description="Re-run retrieval after adaptation to quantify search-space changes.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no fine-tuned retrieval results"],
            risks=["no before/after retrieval comparison available"],
            artifacts=[],
        )

    # Step 20: Comparison + description — only runs when ssl_gate_passed (needs ft_results)
    if ssl_gate_passed:
        _step(
            25, _TOTAL_STEPS, "Model comparison + video description → comparison.md, description.md"
        )
        with _Timer(T, "H_compare"):
            g = step_compare_and_describe(
                frame_list,
                store,
                is_qdrant,
                base_results,
                ft_results,
                models,
                video_id,
                video_name,
                video_dir,
                stats.get("ckpt_mb", 0.0),
                stats.get("onnx_mb", 0.0),
            )
        if g:
            stats["base_infer_ms"] = g.get("base_infer_ms", 0.0)
            stats["ft_infer_ms"] = g.get("ft_infer_ms", 0.0)
            stats["top_description"] = g.get("top_description", "")
            video_context["top_descriptions"] = g.get("text_descriptions", [])
        _append_agentic_step(
            agentic_trace,
            step_id="22",
            title="Comparison and description",
            description="Summarize retrieval changes and derive a CLIP-based coarse natural-language description of the video.",
            status="ok",
            context_inputs=["baseline and adapted retrieval outputs", "sampled frame embeddings"],
            context_outputs=[
                f"top description: {stats.get('top_description', 'unknown')}",
                "comparison summary across model variants",
            ],
            risks=[
                "top text prompt may sound plausible but be too coarse or wrong",
                "comparison metrics can privilege ranking stability over semantics",
                "narrative labels can bias the final synthesis context",
            ],
            artifacts=["comparison.md", "description.md"],
        )
    else:
        T["H_compare"] = 0.0
        _step(26, _TOTAL_STEPS, "Model comparison (skipped — SSL gate did not pass)")
        _append_agentic_step(
            agentic_trace,
            step_id="22",
            title="Comparison and description",
            description="Summarize retrieval changes and derive a CLIP-based coarse natural-language description of the video.",
            status="skipped",
            context_inputs=["SSL gate did not pass"],
            context_outputs=["no comparison or description artifacts"],
            risks=["no adaptation quality signal produced"],
            artifacts=[],
        )

    # Step 21: Multi-model comparison — Gemma vs Qwen vs UniDriveVLA
    if not qwen_result.get("skipped") and not unidrive_result.get("skipped"):
        _step(27, _TOTAL_STEPS, "Multi-model comparison → multi_model_comparison.md")
        with _Timer(T, "T_multimodel"):
            mm = step_multi_model_compare(video_name, video_dir, j, qwen_result, unidrive_result)
        video_context["multi_model_comparison"] = mm
        _append_agentic_step(
            agentic_trace,
            step_id="23",
            title="Multi-model comparison",
            description="Compare Gemma, Qwen, and UniDriveVLA outputs and expose UniDrive mixture-of-experts agreement signals.",
            status="ok",
            context_inputs=[
                "Gemma summary",
                "Qwen structured scene facts",
                "UniDrive expert outputs",
            ],
            context_outputs=[
                f"{mm.get('matched_frames', 0)} matched comparison frames",
                f"Qwen/UniDrive agreement {mm.get('mean_qwen_unidrive_agreement', 0.0):.3f}",
                f"{mm.get('high_risk_frames', 0)} high-risk UniDrive frames",
            ],
            risks=[
                "timestamp-nearest matching can compare slightly different moments",
                "token-overlap agreement is a coarse proxy for semantic agreement",
                "expert consensus may under-report minority expert concerns",
            ],
            artifacts=["multi_model_comparison.md"],
        )
    else:
        T["T_multimodel"] = 0.0
        _step(27, _TOTAL_STEPS, "Multi-model comparison (skipped — requires Qwen and UniDrive)")
        _append_agentic_step(
            agentic_trace,
            step_id="23",
            title="Multi-model comparison",
            description="Compare Gemma, Qwen, and UniDriveVLA outputs and expose UniDrive mixture-of-experts agreement signals.",
            status="skipped",
            context_inputs=["Qwen and UniDrive outputs"],
            context_outputs=["no cross-model comparison artifact"],
            risks=["cross-model disagreement remains implicit"],
            artifacts=[],
        )

    # Step 26: Local threat aggregation — collapse persisted primitives to clip level.
    _step(28, _TOTAL_STEPS, "Local threat inference → local_threat_assessment.json")
    from ...steps.local_threat import step_local_threat

    with _Timer(T, "PS_local_threat"):
        local_threat_result = step_local_threat(
            threat_primitives_result=threat_primitives_result,
            video_dir=video_dir,
            video_name=video_name,
            unidrive_rows=video_context.get("unidrive_analysis", []),
            physical_state=physical_state_result,
        )
    video_context["local_threat"] = local_threat_result
    _append_agentic_step(
        agentic_trace,
        step_id="27",
        title="Local threat inference",
        description="Aggregate persisted threat primitives across the full video window into a policy-free threat estimate.",
        status="ok" if not local_threat_result.get("skipped") else "skipped",
        context_inputs=["threat primitives", "temporal persistence threshold"],
        context_outputs=[
            f"local threat score {float(local_threat_result.get('local_threat_score', 0.0)):.3f}",
            f"automation confidence {float(local_threat_result.get('automation_confidence', 1.0)):.3f}",
        ]
        if not local_threat_result.get("skipped")
        else ["no active local threat output"],
        risks=[
            "persistence threshold can suppress short but real hazards",
            "clip-level aggregation can hide when a threat is localized to a brief segment",
            "threat estimate can be over-trusted if policy and sensor-health checks are skipped downstream",
        ],
        artifacts=["local_threat_assessment.json"]
        if not local_threat_result.get("skipped")
        else [],
    )

    _step(29, _TOTAL_STEPS, "Action policy → policy_decision.json")
    from ...steps.policy import step_policy

    with _Timer(T, "PS_policy"):
        policy_result = step_policy(
            local_threat_result,
            video_dir,
            video_name,
            sensor_health={
                "degraded": float(local_threat_result.get("trust_penalty", 0.0) or 0.0) >= 0.30,
                "health_warnings": [
                    conflict.get("pattern", "unknown")
                    for conflict in (local_threat_result.get("source_pair_conflicts") or [])[:3]
                ],
                "missing_sensors": [],
            },
        )
    video_context["policy_decision"] = policy_result
    _append_agentic_step(
        agentic_trace,
        step_id="28",
        title="Action policy",
        description="Map the threat estimate, confidence, and sensor-health context into a fixed action vocabulary without changing the threat score semantics.",
        status="ok" if not policy_result.get("skipped") else "skipped",
        context_inputs=[
            "local threat estimate",
            "automation confidence",
            "sensor-health indicators",
        ],
        context_outputs=[
            f"recommended action {policy_result.get('recommended_action', 'continue')}",
            f"policy reason {policy_result.get('policy_reason', 'n/a')}",
        ]
        if not policy_result.get("skipped")
        else ["no policy decision"],
        risks=[
            "policy defaults may not match mission-specific objectives",
            "sensor-health heuristics can over-trigger inspect-sensor in noisy environments",
        ],
        artifacts=["policy_decision.json"] if not policy_result.get("skipped") else [],
    )

    # -- Finalization (always runs regardless of SSL gate) ----------------------
    # Step 28: Video synthesis — offload CLIP+DINO; Ollama API call only (no local model)
    if device == "cuda" and clip_dino_on_gpu:
        _offload_models_to_cpu(models)
        clip_dino_on_gpu = False  # noqa: F841
    _step(30, _TOTAL_STEPS, "Video synthesis (ontology + narrative) → video_synthesis.md")
    _qwen_url = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
    _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
    with _Timer(T, "Z_synthesis"):
        step_video_synthesis(
            video_name,
            video_dir,
            video_context,
            api_url=_qwen_url,
            model=_qwen_model,
        )
    _append_agentic_step(
        agentic_trace,
        step_id="29",
        title="Video synthesis",
        description="Use accumulated multimodal context to generate a structured ontology and narrative summary of the whole video.",
        status="ok" if _qwen_url else "skipped",
        context_inputs=[
            "Gemma summary",
            "captions, ASR, OCR, detections, Qwen frame reasoning",
            "local threat assessment",
            "retrieval description and map summary",
        ],
        context_outputs=[
            "video ontology",
            "global narrative summary",
        ]
        if _qwen_url
        else ["no synthesis output"],
        risks=[
            "final narrative can collapse uncertain evidence into a single confident story",
            "contradictions across modalities may be hidden in the synthesized summary",
            "wrong high-level framing can mask the original source of context errors",
        ],
        artifacts=["video_synthesis.md", "video_ontology.json"] if _qwen_url else [],
    )

    # Step 29: Agentic flow artifact — prefer Gemma reasoning, fall back to Qwen, then deterministic text.
    _step(31, _TOTAL_STEPS, "Agentic flow audit → agentic_flow.md")
    _agentic_url = (
        getattr(args, "reasoning_api_url", "")
        or getattr(settings, "REASONING_API_URL", "")
        or getattr(args, "gemma_api_url", "")
        or settings.GEMMA_API_URL
        or _qwen_url
    )
    _agentic_model = (
        getattr(args, "reasoning_model", "")
        or getattr(settings, "REASONING_MODEL", "")
        or getattr(args, "gemma_api_model", "")
        or settings.GEMMA_API_MODEL
        or _qwen_model
    )
    _append_agentic_step(
        agentic_trace,
        step_id="30",
        title="Agentic flow audit",
        description="Audit the full context chain, explain step-to-step reasoning state, and register per-step risks of misidentification and wrong context.",
        status="ok",
        context_inputs=["complete pipeline trace", "all accumulated artifacts and summaries"],
        context_outputs=["agentic_flow.md audit report"],
        risks=[
            "reasoning model can restate upstream errors coherently",
            "audit quality depends on provenance captured from earlier steps",
            "fallback deterministic summary is less nuanced than the LLM audit",
        ],
        artifacts=["agentic_flow.md"],
    )
    with _Timer(T, "AA_agentic"):
        step_agentic_flow_artifact(
            video_name,
            video_dir,
            video_context,
            api_url=_agentic_url,
            model=_agentic_model,
        )
    if device == "cuda":
        _unload_known_sidecars(
            [
                (_agentic_url, _agentic_model),
                (_qwen_url, _qwen_model),
                (
                    getattr(args, "unidrive_api_url", "") or settings.UNIDRIVE_API_URL,
                    getattr(args, "unidrive_model", "") or settings.UNIDRIVE_MODEL,
                ),
                (
                    getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL,
                    getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL,
                ),
            ]
        )

    # -- Step 30: Drone-detection edge model training --------------------------
    _drone_enabled = getattr(args, "drone_detection", None)
    if _drone_enabled is None:
        _drone_enabled = True  # on by default when not explicitly disabled
    if _drone_enabled:
        from ...steps.drone_detection import step_drone_detection_training

        _step(32, _TOTAL_STEPS, "Drone detection training → drone_detection/")
        _append_agentic_step(
            agentic_trace,
            step_id="31",
            title="Drone detection edge training",
            description=(
                "Train YOLOv8n on seraphim-drone-detection-dataset + mission hard negatives; "
                "export ONNX fp32 (Cortex-A76) and int8 (RV1106G3 NPU)."
            ),
            status="ok",
            context_inputs=["extracted mission frames", "seraphim HF dataset batch_001"],
            context_outputs=[
                "drone_yolo8n_a76.onnx",
                "drone_yolo8n_rv1106_int8.onnx",
                "drone_detection_report.md",
            ],
            risks=[
                "small dataset subset limits generalisation",
                "false positives increase without sufficient hard negatives",
                "rknn-toolkit2 required for full NPU deployment on RV1106G3",
            ],
            artifacts=["drone_detection/drone_detection_report.md"],
        )
        with _Timer(T, "AC_drone_detection"):
            drone_result = step_drone_detection_training(
                frame_list, video_name, video_dir, output_dir, device, args
            )
        stats["drone_detection"] = drone_result
        _log.info(
            "  Drone detection: map50=%.4f | fp32=%s | int8=%s | rknn=%s",
            drone_result.get("map50", float("nan")),
            "[ok]" if drone_result.get("model_fp32") else "✗",
            "[ok]" if drone_result.get("model_int8") else "✗",
            "[ok]" if drone_result.get("model_rknn") else "skipped",
        )
    else:
        _step(
            31,
            _TOTAL_STEPS,
            "Drone detection training (skipped — pass --drone-detection to enable)",
        )

    # -- Step 32: Drone audio detection model training -------------------------
    _audio_enabled = getattr(args, "drone_audio", None)
    if _audio_enabled is None:
        _audio_enabled = True  # on by default when not explicitly disabled
    if _audio_enabled:
        from ...steps.drone_audio import step_drone_audio_training

        _step(33, _TOTAL_STEPS, "Drone audio training → drone_audio/")
        _append_agentic_step(
            agentic_trace,
            step_id="33",
            title="Drone audio detection model training",
            description=(
                "Train DroneAudioCNN (small 2-D CNN on MFCC features) on "
                "geronimobasso/drone-audio-detection-samples cached in "
                ".data/drone-audio-data/; export ONNX for edge inference."
            ),
            status="ok",
            context_inputs=[
                ".data/drone-audio-data/train/drone/*.wav",
                ".data/drone-audio-data/train/no_drone/*.wav",
            ],
            context_outputs=[
                "drone_audio_cnn.pt",
                "drone_audio_cnn.onnx",
                "drone_audio_report.md",
            ],
            risks=[
                "datasets library required for first-time HF download",
                "small dataset may limit generalisation to novel drone types",
                "run ssv-prepare-audio first for best results",
            ],
            artifacts=["drone_audio/drone_audio_report.md"],
        )
        with _Timer(T, "AC_drone_audio"):
            audio_result = step_drone_audio_training(video_dir, output_dir, device, args)
        stats["drone_audio"] = audio_result
        _log.info(
            "  Drone audio: val_acc=%.3f  val_f1=%.3f  onnx=%s",
            audio_result.get("val_acc", float("nan")),
            audio_result.get("val_f1", float("nan")),
            "[ok]" if audio_result.get("model_onnx") else "✗",
        )
    else:
        _step(33, _TOTAL_STEPS, "Drone audio training (skipped — pass --drone-audio to enable)")

    # -- Step 33: drau range-detection evaluation (github.com/volod/drau) ------
    _drau_enabled = getattr(args, "drau_eval", None)
    if _drau_enabled is None:
        # Auto: run if the ONNX from step 32 exists; skip otherwise.
        _drau_onnx = video_dir / "drone_audio" / "drone_audio_cnn.onnx"
        _drau_enabled = _drau_onnx.exists()
    if _drau_enabled:
        from ...steps.drau_eval import step_drau_range_eval

        _step(34, _TOTAL_STEPS, "drau range eval → drone_audio/drau_range_report.md")
        _append_agentic_step(
            agentic_trace,
            step_id="34",
            title="drau range-detection evaluation",
            description=(
                "Evaluate DroneAudioCNN ONNX model at simulated distances using "
                "the drau physics model (github.com/volod/drau): inverse-square "
                "amplitude scaling + ISO 9613-1 atmospheric absorption. "
                "Exports drau_edge_test.py for numpy+scipy+onnxruntime-only inference."
            ),
            status="ok",
            context_inputs=[
                "drone_audio/drone_audio_cnn.onnx (from step 32)",
                "synthetic quadcopter audio at 9 distances (1-200 m)",
            ],
            context_outputs=[
                "drau_range_report.md (detection probability vs distance)",
                "drau_edge_test.py (standalone edge script, no PyTorch)",
            ],
            risks=[
                "synthetic signal is a simplification; real drone audio varies by model and rotor configuration",
                "onnxruntime must be installed for inference; skips gracefully if absent",
                "detection range estimate is a model characteristic, not a deployment guarantee",
            ],
            artifacts=[
                "drone_audio/drau_range_report.md",
                "drone_audio/drau_edge_test.py",
            ],
        )
        with _Timer(T, "AC_drau_eval"):
            drau_result = step_drau_range_eval(video_dir, output_dir, args)
        stats["drau_eval"] = drau_result
        if drau_result.get("skipped"):
            _log.info("  drau eval: skipped (%s)", drau_result.get("reason", ""))
        else:
            _log.info(
                "  drau eval: detection_range=%s m  elapsed=%.1fs",
                drau_result.get("detection_range_m", "n/a"),
                drau_result.get("elapsed_sec", 0.0),
            )
    else:
        _step(34, _TOTAL_STEPS, "drau eval (skipped — ONNX not found; pass --drau-eval to force)")

    stats["pipeline_sec"] = sum(T.values())
    runtime_metrics = get_runtime_telemetry()
    stats["vram_wait_time_sec"] = runtime_metrics.get("vram_wait_time_sec", 0.0)
    stats["restore_failures"] = int(runtime_metrics.get("restore_failures", 0.0))
    (video_dir / "runtime_metrics.json").write_text(
        json.dumps(runtime_metrics, indent=2),
        encoding="utf-8",
    )
    stats["analysis_summary"] = _emit_local_run_analytics(video_dir) or {}
    stats["video_dir"] = str(video_dir)

    _banner(f"[ok] Video complete: {video_path.name}")
    _log.info("  Output dir: %s", video_dir)
    return stats


def _run_video_pipeline_safe(
    args: Any,
    video_path: "Path",
    output_dir: "Path",
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
) -> dict[str, Any]:
    """Wrapper around :func:`run_video_pipeline` that always returns a stats dict.

    On exception, returns the partial stats dict with timings recorded up to
    the failure point so step times and frame counts are not lost.
    """
    _out: dict[str, Any] = {}
    try:
        result = run_video_pipeline(
            args, video_path, output_dir, models, store, is_qdrant, device, _out=_out
        )
        # The graph-pipeline path returns a new dict without mutating _out; merge it back.
        if result and result is not _out:
            _out.update(result)
    except Exception as exc:
        _log.error("Pipeline failed for %s: %s", video_path.name, exc, exc_info=True)
        _out.setdefault("name", video_path.stem)
        _out.setdefault("video_dir", str(output_dir / video_path.stem))
        _out["error"] = str(exc)
        _out.setdefault("timings", {})
        _out.setdefault("frames", 0)
        _out.setdefault("duration_sec", 0.0)
        timings = _out.get("timings", {})
        _out.setdefault("pipeline_sec", sum(timings.values()))
    _out.setdefault("name", video_path.stem)
    return _out


# -- Main entry point ----------------------------------------------------------
