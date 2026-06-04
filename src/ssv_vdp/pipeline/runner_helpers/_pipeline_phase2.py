"""Phase 2 — Multimodal analysis: Steps 03-20.

Covers VLM/sensing, 3D map, full state fusion, physical/field state, and threat primitives.
"""

import concurrent.futures as _cf
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

from ...steps.common import _banner, _step, _Timer
from ._agentic import _append_agentic_step

_log = get_logger(__name__)

_TOTAL_STEPS = 35


def run_phase2(
    *,
    args: Any,
    video_path: Path,
    video_dir: Path,
    video_name: str,
    video_id: str,
    models: dict[str, Any],
    store: Any,
    is_qdrant: bool,
    device: str,
    frame_list: list[tuple[str, float]],
    clip_dino_on_gpu: bool,
    # shared mutable state
    stats: dict[str, Any],
    T: dict[str, Any],
    video_context: dict[str, Any],
    agentic_trace: list[dict[str, Any]],
    knowledge: Any,
) -> dict[str, Any]:
    """Run Steps 03-20. Returns dict of step results needed by later phases."""
    from ...steps.caption import (
        _models_on_device,
        _prep_vram_for_step,
        _restore_models_to_gpu,
        _unload_ollama_model,
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
    from ...steps.perception.embed import step_base_model_search_test
    from ...steps.perception.gemma_tracking import step_gemma_directed_tracking
    from ...steps.perception.map import step_advise_3d_map_quality, step_create_3d_map
    from ...steps.perception.scenetok import step_scenetok
    from ...steps.perception.semantic_graph import step_build_semantic_environment_graph
    from ...steps.perception.yolo_sam import step_yolo_sam_detection
    from ...steps.report import write_multimodal_md
    from ...steps.state.fusion import step_full_state_fusion, step_platform_state_fusion

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
    # Unload Gemma immediately after analysis — frees ~12+ GiB for Florence.
    _gemma_api_url_j = settings.GEMMA_API_URL or getattr(args, "gemma_api_url", "")
    _gemma_api_model_j = settings.GEMMA_API_MODEL or getattr(args, "gemma_api_model", "")
    if _gemma_api_url_j and _gemma_api_model_j and device == "cuda":
        _unload_ollama_model(_gemma_api_url_j, _gemma_api_model_j)

    # Step 04: Scene captioning
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
            clip_dino_on_gpu = False  # Florence offloaded them
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
        context_inputs=["timestamped frames", knowledge.domain_hint() or "no domain hint"],
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

    # Step 04b: Gemma segment-boundary diff
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
                gemma_api_backend=getattr(args, "gemma_api_backend", "")
                or settings.GEMMA_API_BACKEND,
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

    # Step 05: ASR
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

    platform_fusion_result = step_platform_state_fusion(
        video_path, frame_list, video_name, video_dir
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
                frame_list, video_name, video_dir, caption_results=caption_results
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

    # Step 08: Detection
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

    # Step 09: YOLO11 + SAM2/3
    yolo_sam_result: dict[str, Any] = {"skipped": True, "detection_results": []}
    if not getattr(args, "no_yolo", False):
        _step(9, _TOTAL_STEPS, "YOLO11 + SAM2/3 detection → yolo_sam/ + detection_comparison.md")
        _prep_vram_for_step(models, device)
        clip_dino_on_gpu = False
        with _Timer(T, "P2_yolo_sam"):
            yolo_sam_result = step_yolo_sam_detection(
                frame_list, video_name, video_dir, device, det_result=det_result
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
        artifacts=["yolo_sam_results.json", "yolo_sam/frame_*_annotated.jpg", "detection_comparison.md"]
        if not yolo_sam_result.get("skipped")
        else [],
    )

    # Step 10: Gemma 4 directed tracking
    gemma_tracking_result: dict[str, Any] = {"skipped": True}
    _gemma_api_url_p3 = getattr(args, "gemma_api_url", "") or settings.GEMMA_API_URL
    _gemma_api_model_p3 = getattr(args, "gemma_api_model", "") or settings.GEMMA_API_MODEL
    if not getattr(args, "no_rfdetr", False) and _gemma_api_url_p3:
        _step(10, _TOTAL_STEPS, "Gemma 4 directed tracking → gemma_tracking/ + gemma_tracking_results.json")
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
        context_inputs=["sampled frames", "Gemma sidecar API output", "CLIP embeddings for SAM mask filtering"],
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
        artifacts=["gemma_tracking_results.json", "gemma_tracking/frame_*_tracked.jpg", "gemma_tracking_summary.md"]
        if not gemma_tracking_result.get("skipped")
        else [],
    )

    # Submit 3D map to background thread now that depth/detection/tracking cues are ready.
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

    # Step 12: Qwen
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
            "frame image", "Florence scene priors", "ASR-aligned subtitle context",
            "OCR/depth/detection cues", "previous Qwen structured state",
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

    # Step 13: UniDriveVLA
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
        context_inputs=["sampled frames", "ASR/OCR context when available", "agentic context from earlier steps"],
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

    if any([args.asr, args.ocr, args.depth, args.detection, args.world_model, args.qwen, getattr(args, "unidrive", False)]):
        write_multimodal_md(
            video_dir / "multimodal_features.md",
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

    # Step 14: SceneTok
    scenetok_result: dict[str, Any] = {"skipped": True}
    if getattr(args, "scenetok", False):
        _step(14, _TOTAL_STEPS, "SceneTok streaming encoder + segmentation decoder → scenetok_tokens.npz")
        _scenetok_api_url = getattr(args, "scenetok_api_url", "") or settings.SCENETOK_API_URL
        _scenetok_checkpoint = getattr(args, "scenetok_checkpoint", "") or settings.SCENETOK_CHECKPOINT
        if _scenetok_api_url:
            import os as _os
            _os.environ.setdefault("SCENETOK_API_URL", _scenetok_api_url)
        if _scenetok_checkpoint:
            import os as _os
            _os.environ.setdefault("SCENETOK_CHECKPOINT", _scenetok_checkpoint)
        with _Timer(T, "S_scenetok"):
            scenetok_result = step_scenetok(
                frame_list, video_dir, checkpoint=_scenetok_checkpoint, mode=settings.SCENETOK_MODE
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
            else (["scenetok_tokens.npz", "scenetok_views/"] if not scenetok_result.get("skipped") else [])
        ),
    )

    # Step 15: Cosmos3
    cosmos3_result: dict[str, Any] = {"skipped": True, "clips": [], "n_clips": 0}
    if getattr(args, "cosmos3", None):
        from ...steps.perception.cosmos3 import step_cosmos3_inference
        _step(15, _TOTAL_STEPS, "Cosmos3 world-model inference → cosmos3_inference.json")
        _prep_vram_for_step(models, device)
        with _Timer(T, "S_cosmos3"):
            cosmos3_result = step_cosmos3_inference(frame_list, video_name, video_dir, device=device)
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

    # Step 16: Base model search — restore CLIP+DINO before joining 3D-map thread.
    if device == "cuda" and not clip_dino_on_gpu:
        _qwen_url = getattr(args, "qwen_api_url", "") or settings.QWEN_API_URL
        _qwen_model = getattr(args, "qwen_model", "") or settings.QWEN_MODEL
        _unidrive_url = getattr(args, "unidrive_api_url", "") or settings.UNIDRIVE_API_URL
        _unidrive_model = getattr(args, "unidrive_model", "") or settings.UNIDRIVE_MODEL
        _prep_vram_for_step(
            models,
            device,
            extra_sidecars=[(_qwen_url, _qwen_model), (_unidrive_url, _unidrive_model)],
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
        context_outputs=[f"top-{len(base_results)} baseline matches", f"query at {query_t_sec:.1f}s"],
        risks=[
            "search quality may favor visual similarity over semantic identity",
            "one query frame can underrepresent broader retrieval behavior",
            "baseline errors can distort later before/after comparisons",
        ],
        artifacts=["base_search.md"],
    )

    # Step 17: Collect 3D map background thread result
    _step(17, _TOTAL_STEPS, "3D map + Gaussian Splat → 3d_map/ (joining background thread)")
    with _Timer(T, "I_3dmap"):
        if _map_future is not None:
            try:
                h = _map_future.result(timeout=600)
            except Exception as _map_exc:
                _log.warning("  3D-map background thread raised: %s", _map_exc, exc_info=True)
                h = {"sfm_poses": 0, "method": "failed", "points": None, "gsplat_method": "failed", "splat_ply": None, "viewer_html": ""}
            finally:
                _map_executor.shutdown(wait=False)
        else:
            _map_executor.shutdown(wait=False)
            h = {"sfm_poses": 0, "method": "skipped", "points": None, "gsplat_method": "skipped", "splat_ply": None, "viewer_html": ""}
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
    advisor_issues = (map_quality_advisor.get("summary", {}) or {}).get("issues", []) or []
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

    # Full probabilistic state fusion

    _rssm_mean: float | None = None
    if world_result and not world_result.get("skipped"):
        _rssm_scores = world_result.get("rssm_scores") or []
        if _rssm_scores:
            _rssm_mean = float(sum(_rssm_scores) / len(_rssm_scores))

    _qwen_captions = (
        qwen_result.get("structured_captions") or [] if not qwen_result.get("skipped") else []
    )

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

    # Step 18: Physical state
    _step(18, _TOTAL_STEPS, "Physical scene state summary → physical_state_summary.json")
    from ...steps.state.physical_state import step_physical_state as _step_physical_state

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

    # Step 19: Environmental field state
    _step(19, _TOTAL_STEPS, "Environmental field state → field_state_summary.json")
    from ...steps.state.field_state import step_field_state as _step_field_state

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

    # Step 20: Threat primitives
    _step(20, _TOTAL_STEPS, "Threat primitives → threat_primitives.json")
    from ...steps.threat.threat_primitives import step_threat_primitives as _step_threat_primitives

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

    return {
        "j": j,
        "caption_results": caption_results,
        "asr_result": asr_result,
        "ocr_result": ocr_result,
        "depth_result": depth_result,
        "det_result": det_result,
        "yolo_sam_result": yolo_sam_result,
        "gemma_tracking_result": gemma_tracking_result,
        "world_result": world_result,
        "qwen_result": qwen_result,
        "unidrive_result": unidrive_result,
        "base_results": base_results,
        "query_frame": query_frame,
        "query_t_sec": query_t_sec,
        "h": h,
        "full_fusion_result": full_fusion_result,
        "platform_fusion_result": platform_fusion_result,
        "physical_state_result": physical_state_result,
        "field_state_result": field_state_result,
        "threat_primitives_result": threat_primitives_result,
        "clip_dino_on_gpu": clip_dino_on_gpu,
    }
