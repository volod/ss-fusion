"""Shared agentic helpers for LangGraph pipeline nodes.

All functions are stateless — they operate on raw data and return results.
No LangGraph imports here so helpers can be unit-tested independently.
"""

import json
import time
from typing import Any

import httpx

from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger(__name__)

# Default tracking targets used when Gemma JSON parse fails completely.
DEFAULT_TRACKING_TARGETS: list[str] = ["person", "vehicle", "sign"]

# Minimum cosine similarity for a Gemma claim to be considered frame-supported.
GEMMA_CLAIM_MIN_SIM: float = 0.25

# MoE consensus threshold — frames below this get flagged as low-agreement.
MOE_CONSENSUS_THRESHOLD: float = 0.5


# -- JSON guard ----------------------------------------------------------------


def json_guard(raw: str, required_keys: list[str]) -> dict[str, Any] | None:
    """Parse *raw* as JSON and return the dict only if all *required_keys* present."""
    try:
        # Strip markdown code fences if present.
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return None
        if all(k in parsed for k in required_keys):
            return parsed
    except Exception:
        pass
    return None


# -- LLM call with exponential back-off ---------------------------------------


def llm_call_with_retry(
    endpoint: str,
    payload: dict[str, Any],
    *,
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    timeout_sec: float = 120.0,
) -> tuple[str, int]:
    """POST *payload* to *endpoint* up to *max_attempts* times.

    Returns ``(response_text, attempt_number_used)`` on success.
    Raises ``RuntimeError`` if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = httpx.post(endpoint, json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text, attempt
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                sleep_sec = backoff_base ** (attempt - 1)
                _log.debug(
                    "LLM call attempt %d failed (%s) — retrying in %.1fs", attempt, exc, sleep_sec
                )
                time.sleep(sleep_sec)
    raise RuntimeError(
        f"LLM endpoint {endpoint!r} failed after {max_attempts} attempts"
    ) from last_exc


# -- Critique pass -------------------------------------------------------------


def critique_pass(
    endpoint: str,
    model: str,
    generation: str,
    evidence_summary: str,
    *,
    timeout_sec: float = 60.0,
) -> str:
    """Ask the LLM to verdict the *generation* against *evidence_summary*.

    Returns one of: ``"PASS"`` | ``"MINOR_ISSUES"`` | ``"MAJOR_CONTRADICTION"`` |
    ``"CRITIQUE_FAILED"``.
    """
    prompt = (
        f"Evidence from video analysis:\n{evidence_summary}\n\n"
        f"Generated output (first 600 chars):\n{generation[:600]}\n\n"
        "Does the output contradict the evidence? "
        "Reply with exactly one of: PASS / MINOR_ISSUES / MAJOR_CONTRADICTION. "
        "Then give one sentence of explanation."
    )
    try:
        raw, _ = llm_call_with_retry(
            endpoint,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.0,
            },
            max_attempts=2,
            timeout_sec=timeout_sec,
        )
        upper = raw.upper()
        if "MAJOR_CONTRADICTION" in upper:
            return "MAJOR_CONTRADICTION"
        if "MINOR_ISSUES" in upper:
            return "MINOR_ISSUES"
        return "PASS"
    except Exception as exc:
        _log.debug("Critique pass failed: %s", exc)
        return "CRITIQUE_FAILED"


# -- MoE consensus scoring -----------------------------------------------------


def _jaccard(a: str, b: str) -> float:
    """Token-overlap Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def moe_consensus_score(
    expert_outputs: list[dict[str, Any]], field: str = "recommended_action"
) -> float:
    """Mean pairwise Jaccard similarity of *field* across expert outputs."""
    texts = [str(e.get(field, "")) for e in expert_outputs if field in e]
    if len(texts) < 2:
        return 1.0
    pairs = 0
    total = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += _jaccard(texts[i], texts[j])
            pairs += 1
    return total / pairs if pairs else 1.0


def low_agreement_frames(
    results: list[dict[str, Any]],
    experts_key: str = "experts",
    threshold: float = MOE_CONSENSUS_THRESHOLD,
) -> list[int]:
    """Return indices of frames whose MoE consensus score is below *threshold*."""
    low = []
    for idx, frame_result in enumerate(results):
        experts = frame_result.get(experts_key, [])
        if len(experts) < 2:
            continue
        score = moe_consensus_score(experts)
        if score < threshold:
            low.append(idx)
    return low


# -- Evidence summary builder --------------------------------------------------


def build_evidence_summary(state: dict[str, Any]) -> str:
    """Build a concise factual evidence string from accumulated state for critique prompts."""
    vc = state.get("video_context", {})
    parts: list[str] = []
    if vc.get("gemma_analysis"):
        ga = vc["gemma_analysis"]
        parts.append(
            f"Gemma: scene_type={ga.get('task_results', {}).get('scene_type', '?')}, "
            f"n_frames={ga.get('n_frames', 0)}"
        )
    captions = vc.get("captions", [])
    if captions:
        sample = captions[0].get("caption", "")[:120] if captions else ""
        parts.append(f'Florence captions ({len(captions)}): "{sample}…"')
    detections = vc.get("detections", {})
    if detections:
        top = sorted(detections.items(), key=lambda x: -x[1])[:5]
        parts.append(f"Detections: {', '.join(f'{k}({v})' for k, v in top)}")
    asr = vc.get("asr_segments", [])
    if asr:
        first_text = asr[0].get("text", "")[:80] if asr else ""
        parts.append(f'ASR ({len(asr)} segments): "{first_text}…"')
    return "\n".join(parts) or "No structured evidence available."
