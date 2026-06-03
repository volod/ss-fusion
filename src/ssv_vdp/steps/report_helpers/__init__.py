"""Report helpers subpackage — all write_*_md functions and print_run_stats."""

from ._captions import (
    _write_gemma_captions_md,
    write_gemma_segment_captions_md,
    write_scene_captions_md,
)
from ._search import write_comparison_md, write_description_md, write_search_md
from ._analysis import write_gemma_analysis_md
from ._adaptation import write_distill_stats_md, write_finetune_stats_md
from ._multimodal import (
    write_detailed_captions_md,
    write_multi_model_comparison_md,
    write_multimodal_md,
    write_state_fusion_md,
    write_unidrive_analysis_md,
)
from ._synthesis import (
    _normalise_threat_rows,
    write_agentic_flow_md,
    write_video_synthesis_md,
)
from ._stats import (
    _STEP_LABELS,
    _fmt_analytics_coverage,
    _fmt_analytics_detections,
    _fmt_analytics_map,
    _fmt_analytics_temporal,
    _fmt_analytics_warnings,
    _fmt_analytics_world_tracking,
    _fmt_sec,
    print_run_stats,
    write_final_stats_md,
)

__all__ = [
    # captions
    "_write_gemma_captions_md",
    "write_scene_captions_md",
    "write_gemma_segment_captions_md",
    # search / comparison
    "write_search_md",
    "write_comparison_md",
    "write_description_md",
    # analysis
    "write_gemma_analysis_md",
    # adaptation
    "write_finetune_stats_md",
    "write_distill_stats_md",
    # multimodal
    "write_multimodal_md",
    "write_state_fusion_md",
    "write_detailed_captions_md",
    "write_unidrive_analysis_md",
    "write_multi_model_comparison_md",
    # synthesis / threat
    "_normalise_threat_rows",
    "write_video_synthesis_md",
    "write_agentic_flow_md",
    # stats
    "_STEP_LABELS",
    "_fmt_sec",
    "write_final_stats_md",
    "print_run_stats",
    "_fmt_analytics_coverage",
    "_fmt_analytics_detections",
    "_fmt_analytics_temporal",
    "_fmt_analytics_world_tracking",
    "_fmt_analytics_map",
    "_fmt_analytics_warnings",
]
