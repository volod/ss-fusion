"""Captioning steps: Gemma, Florence, Qwen, ASR, OCR, depth, detection, world model.

All public names from the original caption.py are re-exported here so that every
existing ``from ssv_vdp.steps.caption import X`` import continues to work unchanged.
"""

from selfsuvis.pipeline.core import settings  # noqa: F401 — accessed as caption.settings by tests

# -- Step functions -----------------------------------------------------------
from ._florence import step_gemma_segment_captions, step_scene_captioning  # noqa: F401
from ._gemma_analysis import step_gemma_analysis  # noqa: F401
from ._sensing import (  # noqa: F401
    step_asr_transcription,
    step_depth_estimation,
    step_object_detection,
    step_ocr_extraction,
)
from ._vlm import step_qwen_captioning, step_unidrive_analysis  # noqa: F401
from ._world_model import step_world_model_pass  # noqa: F401

# -- caption_helpers re-exports (used directly by pipeline code) --------------
from ..caption_helpers.frame_selection import (  # noqa: F401
    _SEGMENT_DIFF_MAX_BOUNDARIES_DEFAULT,
    _adaptive_sparse_budget,
    _reduce_llm_sample_frames,
    _select_qwen_frames,
    _select_segment_boundary_pairs,
)
from ..caption_helpers.gemma_api import (  # noqa: F401
    _gemma_analyse_frame_via_api,
    _gemma_diff_two_frames_via_api,
    _summarise_gemma_captions_to_structured_scene,
    step_qwen_captioning_gemma_fallback,
)
from ..caption_helpers.ocr import (  # noqa: F401
    _fallback_ocr_frame_sample,
    _select_ocr_candidate_frames,
)
from ..caption_helpers.ollama import (  # noqa: F401
    _compute_sidecar_timeout,
    _list_ollama_models,
    _recommend_gemma_sidecar_models,
    _resolve_ollama_gemma_model,
    _resolve_ollama_reasoning_model,
    _unload_known_sidecars,
    _unload_ollama_model,
)
from ..caption_helpers.vlm_api import caption_via_florence_api, caption_via_qwen_api  # noqa: F401
from ..caption_helpers.vram import (  # noqa: F401
    _detect_free_vram_gb,
    _flush_cuda_allocator,
    _guard_min_free_vram,
    _log_vram_snapshot,
    _models_on_device,
    _offload_models_to_cpu,
    _prep_vram_for_step,
    _restore_models_to_gpu,
    get_runtime_telemetry,
    reset_runtime_telemetry,
)

# -- common re-exports used by pipeline code ----------------------------------
from ..common import (  # noqa: F401
    _GEMMA_ANALYSIS_SAMPLE_N,
    _GEMMA_TEXT_PROBES,
    _SCENE_CHANGE_THRESH,
    VideoKnowledge,
    _open_frame_batch,
    _open_frame_image,
    _run_batched_frame_inference,
    write_json_artifact,
    write_markdown_artifact,
)
