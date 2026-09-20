"""Shared logging helpers, constants, and VideoKnowledge for the local subpackage."""

import hashlib
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any

from PIL import Image

# -- Logging helpers ------------------------------------------------------------

_LOG_FMT = "%(asctime)s  %(levelname)-7s  %(message)s"
_DATE_FMT = "%H:%M:%S"

_NOISY_LOGGERS = (
    "urllib3",
    "PIL",
    "filelock",
    "torch",
    "timm",
    "httpx",
    "httpcore",
    "transformers",
)

# Logger namespaces that should stay at INFO level.
# Setting root to WARNING silences SAM2/ultralytics spam; these overrides
# restore INFO for our own code so progress messages are not lost.
_PIPELINE_NAMESPACES = ("pipeline", "models", "dinov2")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_DATE_FMT)
    # Suppress noisy named third-party loggers
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Raise root to WARNING — suppresses SAM2/ultralytics verbose INFO ("root" in output)
    logging.getLogger().setLevel(logging.WARNING)
    # Re-pin our own namespaces to INFO so they are not silenced by root's level
    for ns in _PIPELINE_NAMESPACES:
        logging.getLogger(ns).setLevel(logging.INFO)


def _configure_warnings() -> None:
    warnings.filterwarnings("ignore", message="xFormers is available", category=UserWarning)
    warnings.filterwarnings("ignore", message="xFormers is not available", category=UserWarning)
    warnings.filterwarnings(
        "ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message="The image_processor_class argument is deprecated", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Florence2Processor.*image_processor_class = 'CLIPImageProcessor'.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The following generation flags are not valid and may be ignored: .*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You seem to be using the pipelines sequentially on GPU.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Constant folding.*",
        category=UserWarning,
    )


# Apply timm FutureWarning filter at import time so it takes effect before
# timm is imported anywhere in the process (calling _configure_warnings()
# later would be too late — the warning fires at timm import time).
warnings.filterwarnings(
    "ignore", message="Importing from timm.models.layers is deprecated", category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message=r".*Florence2Processor.*image_processor_class = 'CLIPImageProcessor'.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The following generation flags are not valid and may be ignored: .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You seem to be using the pipelines sequentially on GPU.*",
    category=UserWarning,
)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


# Apply at import time — pipeline/core/logging may have already called
# basicConfig, making the _configure_logging() basicConfig call a no-op.
logging.getLogger().setLevel(logging.WARNING)
for _ns in _PIPELINE_NAMESPACES:
    logging.getLogger(_ns).setLevel(logging.INFO)

_log = logging.getLogger("pipeline.local")


def _banner(msg: str) -> None:
    width = 72
    _log.info("=" * width)
    _log.info("  %s", msg)
    _log.info("=" * width)


def _step(n: int, total: int, name: str) -> None:
    _log.info("--- Step %d/%d: %s", n, total, name)


def write_json_artifact(path: Path, payload: Any, *, ensure_ascii: bool = True) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )


def write_markdown_artifact(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gemma_cache_file(video_dir: Path) -> Path:
    return video_dir / "runtime_cache" / "gemma_responses.json"


def load_gemma_cache(video_dir: Path, *, enabled: bool) -> dict[str, Any]:
    path = gemma_cache_file(video_dir)
    if not enabled or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_gemma_cache(video_dir: Path, cache: dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    path = gemma_cache_file(video_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(path, cache, ensure_ascii=False)


def gemma_frame_cache_key(frame_path: str, *, model: str, prompt_tag: str) -> str:
    digest = hashlib.sha256(Path(frame_path).read_bytes()).hexdigest()
    return f"{prompt_tag}:{model}:{digest}"


class _Timer:
    """Context manager that records elapsed seconds into a dict under *key*."""

    def __init__(self, store: dict[str, float], key: str) -> None:
        self._store = store
        self._key = key
        self._t0 = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.time()
        return self

    def __exit__(self, *_: Any) -> None:
        self._store[self._key] = time.time() - self._t0


def _open_frame_image(frame_path: str) -> Image.Image:
    try:
        return Image.open(frame_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))


def _open_frame_batch(batch: list[tuple[str, float]]) -> list[Image.Image]:
    return [_open_frame_image(fp) for fp, _t in batch]


def _run_batched_frame_inference(
    frame_list: list[tuple[str, float]],
    *,
    batch_size: int,
    batch_fn,
    warning_label: str,
    error_result: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(frame_list), batch_size):
        batch = frame_list[batch_start : batch_start + batch_size]
        imgs = _open_frame_batch(batch)
        try:
            batch_out = batch_fn(batch, imgs)
        except Exception as exc:
            _log.warning("  %s batch %d failed: %s", warning_label, batch_start, exc)
            batch_out = [dict(error_result) for _ in batch]
        if len(batch_out) != len(batch):
            _log.warning(
                "  %s batch %d returned %d results for %d frames; padding/truncating",
                warning_label,
                batch_start,
                len(batch_out),
                len(batch),
            )
            padded = list(batch_out[: len(batch)])
            while len(padded) < len(batch):
                padded.append(dict(error_result))
            batch_out = padded
        for (fp, t_sec), r in zip(batch, batch_out):
            results.append({"frame_path": fp, "t_sec": t_sec, **r})
    return results


# -- Text prompts for CLIP video-to-text description ---------------------------

_TEXT_PROMPTS: list[str] = [
    "aerial footage of a road or highway",
    "outdoor terrain with green vegetation",
    "urban environment with buildings and streets",
    "industrial site or construction area",
    "rural landscape viewed from above",
    "dense forest or woodland area",
    "coastal area or water body",
    "agricultural field or farmland",
    "mountain or rocky terrain",
    "open desert or arid landscape",
    "parking lot or vehicle depot",
    "residential neighbourhood from above",
    "radar antenna or rotating radar dish on a rooftop or tower",
    "military radar installation in open terrain",
    "phased array radar or sensor array on a vehicle or structure",
    "surveillance radar dome or radome on a building",
    "weather radar tower in a field",
    "radar site with large parabolic antenna",
    "electronic warfare sensor mast on a ship or vehicle",
    "panoramic wide-angle view of vehicles on a road",
    "multiple cars and trucks visible in a wide scene",
    "convoy of military vehicles on a road viewed from above",
    "vehicles moving along a highway in a panoramic shot",
    "armoured vehicles or tanks in an open field",
    "trucks and heavy transport vehicles at an industrial site",
    "emergency vehicles with lights visible from aerial view",
    "vehicles parked in an open area viewed from a drone",
    "mobile radar unit mounted on a truck in a field",
    "radar vehicle or electronic warfare truck in a convoy",
    "surveillance vehicle with antenna array on a road",
    "small vehicles weaving in a serpentine pattern along a road",
    "tiny cars following a zigzag slalom course on a wide road",
    "overhead view of vehicles navigating obstacles in a serpentine layout",
    "small objects moving in curved paths on a straight road from above",
    "miniature vehicles visible as small dots arranged in a winding line",
    "drone view of traffic slowing and weaving around road obstacles",
    "serpentine convoy of small vehicles on an open road from altitude",
    "simple portable radar unit on a tripod in a field",
    "small ground surveillance radar deployed on the roadside",
    "handheld or man-portable radar device in open terrain",
    "compact radar sensor on a pole or mast near a road",
    "short-range radar unit with small dish antenna on the ground",
    "mobile radar system on a lightweight trailer or cart",
    "radar detector or traffic speed radar on a road",
]

# -- Gemma analysis constants --------------------------------------------------

_GEMMA_ANALYSIS_SAMPLE_N = 30  # max frames sampled per video for Gemma analysis
_SCENE_CHANGE_THRESH = 0.25  # cosine distance threshold for scene change detection

# Text probes for cross-modal search and zero-shot classification
_GEMMA_TEXT_PROBES: list[str] = [
    "aerial view of open terrain",
    "military vehicle or equipment",
    "road or highway from above",
    "buildings and urban infrastructure",
    "natural landscape or vegetation",
    "radar or surveillance equipment",
    "convoy or vehicle formation",
    "industrial or construction site",
    "open field or farmland",
    "coastal or water feature",
]

# -- Runner label --------------------------------------------------------------

_RUNNER_LABEL = "local full-analysis pipeline (`main.py --mode local`)"

# -- VideoKnowledge — agentic knowledge accumulator ----------------------------


def _analyze_caption_sequence(
    caption_results: list[dict[str, Any]],
    new_segment_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Annotate caption results with temporal segment info.

    Adds to each result:
        segment_id       — integer, increments when caption content changes
        is_new_segment   — True for first frame of each segment
        similarity       — Jaccard similarity to previous frame's caption (None for first)
        segment_start_t  — t_sec of the first frame in this segment
    """
    enriched: list[dict[str, Any]] = []
    seg_id = 0
    prev_caption = ""
    seg_start_t = 0.0

    for i, r in enumerate(caption_results):
        cap = (r.get("caption") or "").strip()
        if i == 0:
            sim = None
            is_new = True
            seg_start_t = r.get("t_sec", 0.0)
        else:
            sim = _jaccard(prev_caption, cap)
            if sim < new_segment_threshold:
                seg_id += 1
                is_new = True
                seg_start_t = r.get("t_sec", 0.0)
            else:
                is_new = False

        enriched.append(
            {
                **r,
                "segment_id": seg_id,
                "is_new_segment": is_new,
                "similarity": sim,
                "segment_start_t": seg_start_t,
            }
        )
        if cap:
            prev_caption = cap

    return enriched


def _jaccard(a: str, b: str) -> float:
    """Token-overlap similarity between two caption strings (0=different, 1=identical)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class VideoKnowledge:
    """Accumulates structured observations across pipeline steps.

    Each step deposits its output here via ``add_*`` methods.  Later steps
    query ``context_for_frame(t_sec)`` or ``domain_hint()`` to receive all
    prior knowledge relevant to that moment in the video.

    Design goals:
    - Per-frame context: Florence caption + depth profile + detections +
      ASR + OCR + scene segment — all aligned by timestamp.
    - Domain summary: Gemma scene type + entity inventory → enriches Qwen
      system prompt and Florence domain hint.
    - Continuity: previous Qwen structured results feed back as "prior state"
      for the next Qwen call, letting the model track what changed.
    """

    def __init__(self, video_name: str, duration_sec: float, frame_count: int) -> None:
        self.video_name = video_name
        self.duration_sec = duration_sec
        self.frame_count = frame_count

        # Gemma-derived domain knowledge (step 03)
        self.scene_type: str = ""  # dominant zero-shot category
        self.n_transitions: int = 0
        self.n_clusters: int = 0
        self.gemma_mnn_dino: float = 0.0

        # Per-frame outputs keyed by t_sec (steps 04, 05, 06, 07, 08)
        self._captions: dict[float, str] = {}  # Florence caption text
        self._asr: dict[float, str] = {}  # ASR subtitle text
        self._ocr: dict[float, str] = {}  # OCR visible text
        self._depth: dict[float, dict] = {}  # depth summary dict
        self._detections: dict[float, list[str]] = {}  # detected labels at t
        self._state_fusion: dict[float, dict[str, Any]] = {}  # fused platform state at t

        # Sorted timestamp index for nearest-frame lookups
        self._ts_captions: list[float] = []
        self._ts_depth: list[float] = []
        self._ts_detections: list[float] = []
        self._ts_state_fusion: list[float] = []

        # Scene segments from caption analysis (step 04 enrichment)
        self._segments: list[dict[str, Any]] = []

        # Entity inventory: all distinct labels seen across all frames
        self.known_entities: list[str] = []

        # Last Qwen result: feeds into next Qwen call as "previous state"
        self._last_qwen: dict[str, Any] = {}

        # Physical state summary (step_physical_state)
        self._physical_state: dict[str, Any] | None = None

    # -- Deposit methods -------------------------------------------------------

    def add_gemma(self, task_results: dict[str, Any], mnn_dino: float = 0.0) -> None:
        """Deposit Gemma analysis results (step 03)."""
        clf = task_results.get("scene_classification", {})
        if clf.get("category_distribution"):
            self.scene_type = next(iter(clf["category_distribution"]), "")
        sc = task_results.get("scene_change_detection", {})
        self.n_transitions = sc.get("n_changes", 0)
        cl = task_results.get("scene_clustering", {})
        self.n_clusters = cl.get("n_clusters", 0)
        self.gemma_mnn_dino = mnn_dino

    def add_captions(self, caption_results: list[dict[str, Any]]) -> None:
        """Deposit Florence per-frame captions (step 04) and derive segments."""
        self._captions = {
            r["t_sec"]: r.get("caption") or "" for r in caption_results if "t_sec" in r
        }
        self._ts_captions = sorted(self._captions)
        # Re-use existing segment analysis
        enriched = _analyze_caption_sequence(caption_results)
        seg_map: dict[int, dict[str, Any]] = {}
        for r in enriched:
            sid = r["segment_id"]
            if sid not in seg_map:
                seg_map[sid] = {
                    "segment_id": sid,
                    "start_t": r["t_sec"],
                    "end_t": r["t_sec"],
                    "caption": r.get("caption") or "",
                }
            else:
                seg_map[sid]["end_t"] = r["t_sec"]
        self._segments = [seg_map[k] for k in sorted(seg_map)]

    def add_asr(self, subtitle_map: dict[float, str]) -> None:
        """Deposit ASR subtitle map (step 05)."""
        self._asr = {float(k): v for k, v in subtitle_map.items() if v}

    def add_ocr(self, ocr_results: list[dict[str, Any]]) -> None:
        """Deposit OCR per-frame results (step 06)."""
        self._ocr = {
            r["t_sec"]: r["ocr_text"] for r in ocr_results if r.get("ocr_text") and "t_sec" in r
        }

    def add_depth(self, depth_results: list[dict[str, Any]]) -> None:
        """Deposit depth estimation per-frame results (step 07)."""
        self._depth = {r["t_sec"]: r for r in depth_results if "t_sec" in r}
        self._ts_depth = sorted(self._depth)

    def add_detections(self, detection_results: list[dict[str, Any]]) -> None:
        """Deposit object detection per-frame results (step 08)."""
        entity_set: set = set()
        for r in detection_results:
            t = r.get("t_sec")
            if t is None:
                continue
            labels = [d["label"] for d in r.get("detections", []) if d.get("label")]
            self._detections[float(t)] = labels
            entity_set.update(labels)
        self._ts_detections = sorted(self._detections)
        # Keep top entities by frequency
        counts: dict[str, int] = {}
        for labels in self._detections.values():
            for lbl in labels:
                counts[lbl] = counts.get(lbl, 0) + 1
        self.known_entities = [k for k, _ in sorted(counts.items(), key=lambda x: -x[1])[:15]]

    def add_physical_state(self, summary: dict[str, Any]) -> None:
        """Deposit physical state summary (step_physical_state)."""
        if not summary.get("skipped"):
            self._physical_state = summary

    def add_state_fusion(self, posterior_samples: list[Any]) -> None:
        """Deposit fused platform-state posterior samples."""
        self._state_fusion = {
            float(sample.t_sec): sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
            for sample in posterior_samples
            if getattr(sample, "t_sec", None) is not None
        }
        self._ts_state_fusion = sorted(self._state_fusion)

    def update_qwen_state(self, result: dict[str, Any]) -> None:
        """Record the most recent Qwen output for use as prior state context."""
        if not result.get("service_unavailable") and not result.get("parse_error"):
            self._last_qwen = result

    # -- Query methods ---------------------------------------------------------

    def physical_state_hint(self) -> str:
        """One-line physical state summary for prompt injection."""
        ps = self._physical_state
        if not ps:
            return ""
        return (
            f"pose_conf={ps.get('platform_pose_confidence', 0.0):.2f}  "
            f"occupancy={ps.get('near_field_occupancy_density', 0.0):.2f}  "
            f"free_space={ps.get('free_space_estimate', 1.0):.2f}  "
            f"tracks={ps.get('confirmed_tracks', 0)}"
        )

    def domain_hint(self) -> str:
        """Short domain summary for use as a model prompt prefix."""
        parts: list[str] = []
        if self.scene_type:
            parts.append(f"Dominant scene: {self.scene_type}")
        if self.known_entities:
            parts.append(f"Known objects: {', '.join(self.known_entities[:6])}")
        if self.n_transitions:
            parts.append(f"Visual transitions: {self.n_transitions}")
        phys = self.physical_state_hint()
        if phys:
            parts.append(f"Physical: {phys}")
        return " | ".join(parts)

    def context_for_frame(self, t_sec: float, asr_window: float = 2.0) -> str:
        """Build a multi-line context string for *t_sec* from all deposited knowledge.

        Returned string is injected into Qwen's user prompt so it can
        reason with full situational awareness, not just the raw image.
        """
        lines: list[str] = []

        # Florence caption for this frame
        cap = self._nearest(self._ts_captions, self._captions, t_sec, max_gap=2.0)
        if cap:
            lines.append(f"[Prior scene description]: {cap[:150]}")

        # Current scene segment
        seg = self._segment_at(t_sec)
        if seg and seg.get("caption"):
            lines.append(
                f"[Scene segment {seg['segment_id'] + 1}, "
                f"{seg['start_t']:.1f}s–{seg['end_t']:.1f}s]: "
                f"{seg['caption'][:120]}"
            )

        # ASR in window
        asr_parts = [txt for ts, txt in self._asr.items() if abs(ts - t_sec) <= asr_window]
        if asr_parts:
            lines.append(f"[Audio context]: {' '.join(asr_parts)[:120]}")

        # OCR exact or ±1 s
        ocr_parts = [txt for ts, txt in self._ocr.items() if abs(ts - t_sec) <= 1.0]
        if ocr_parts:
            lines.append(f"[Visible text]: {' '.join(ocr_parts)[:100]}")

        # Depth profile
        dep = self._nearest(self._ts_depth, self._depth, t_sec, max_gap=2.0)
        if dep:
            nr = dep.get("near_ratio", dep.get("near_frac", 0.0))
            mn = dep.get("mean_depth", dep.get("median", 0.0))
            if nr or mn:
                lines.append(f"[Depth profile]: near_ratio={nr:.2f}  mean={mn:.2f}")

        # Detected objects at this timestamp
        dets = self._nearest(self._ts_detections, self._detections, t_sec, max_gap=2.0)
        if dets:
            lines.append(f"[Detected objects]: {', '.join(dets[:8])}")

        fused = self._nearest(self._ts_state_fusion, self._state_fusion, t_sec, max_gap=2.0)
        if fused:
            pos = fused.get("position_enu_m") or {}
            vel = fused.get("velocity_enu_mps") or {}
            lines.append(
                "[Fused platform state]: "
                f"pos=({pos.get('x', 0.0):.1f}, {pos.get('y', 0.0):.1f}, {pos.get('z', 0.0):.1f}) m  "
                f"vel=({vel.get('x', 0.0):.1f}, {vel.get('y', 0.0):.1f}, {vel.get('z', 0.0):.1f}) m/s  "
                f"quality={fused.get('quality', 'unknown')}"
            )

        # Prior Qwen state (what the model extracted from the previous frame)
        if self._last_qwen:
            prev_vg = self._last_qwen.get("vehicle_groups", [])
            prev_road = self._last_qwen.get("road_surface", "")
            prev_cond = self._last_qwen.get("road_condition", "")
            if prev_vg or prev_road:
                vg_str = (
                    "; ".join(f"{g.get('count', 1)}×{g.get('type', '?')}" for g in prev_vg)
                    if prev_vg
                    else "none"
                )
                lines.append(
                    f"[Prior frame state]: vehicles={vg_str}  "
                    f"road={prev_road}  condition={prev_cond}"
                )

        return "\n".join(lines)

    # -- Private helpers -------------------------------------------------------

    @staticmethod
    def _nearest(ts_index: list[float], data: dict, t: float, max_gap: float = 5.0):
        """Return the value in *data* whose key is closest to *t*, within *max_gap*."""
        if not ts_index:
            return None
        idx = min(range(len(ts_index)), key=lambda i: abs(ts_index[i] - t))
        if abs(ts_index[idx] - t) <= max_gap:
            return data.get(ts_index[idx])
        return None

    def _segment_at(self, t: float) -> dict[str, Any] | None:
        """Return the scene segment that contains timestamp *t*."""
        for seg in self._segments:
            if seg["start_t"] <= t <= seg["end_t"] + 0.5:
                return seg
        return self._segments[-1] if self._segments else None
