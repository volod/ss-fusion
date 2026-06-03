"""Video synthesis (ontology + narrative) and agentic-flow artifact steps."""

import json
import time
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger
from ._agentic import (
    _build_agentic_flow_prompt,
    _build_agentic_flow_prompt_compact,
    _build_agentic_flow_prompt_simple,
    _build_context_prompt,
    _fallback_agentic_flow_analysis,
    _is_simple_agentic_audit,
    _is_valid_agentic_flow_analysis,
    _reasoning_timeout_for_model,
    _strip_thinking_tokens,
)

_log = get_logger(__name__)


def step_agentic_flow_artifact(
    video_name: str,
    video_dir: Path,
    video_context: dict[str, Any],
    api_url: str,
    model: str,
) -> dict[str, Any]:
    """Final step: generate an artifact tracing agentic context and risks."""
    from ...steps.caption import _log_vram_snapshot
    from ...steps.report import write_agentic_flow_md

    result: dict[str, Any] = {"skipped": True, "llm_used": False, "model": model or "deterministic"}
    output_path = video_dir / "agentic_flow.md"
    llm_analysis = ""
    t0 = time.time()
    _log_vram_snapshot("before reasoning sidecar use")

    if api_url:
        try:
            import httpx

            endpoint = f"{api_url.rstrip('/')}/chat/completions"
            timeout_sec = _reasoning_timeout_for_model(model, api_url=api_url)
            is_simple = _is_simple_agentic_audit(video_context)
            if is_simple:
                attempts = [
                    {
                        "label": "simple",
                        "prompt": _build_agentic_flow_prompt_simple(video_name, video_context),
                        "max_tokens": int(
                            getattr(settings, "REASONING_MAX_TOKENS_SIMPLE", 1000) or 1000
                        ),
                    },
                ]
            else:
                attempts = [
                    {
                        "label": "compact",
                        "prompt": _build_agentic_flow_prompt_compact(video_name, video_context),
                        # deepseek-r1 uses chain-of-thought <think> tokens before answering;
                        # 1600 gives ~600 thinking tokens + ~1000 for the answer body.
                        "max_tokens": int(
                            getattr(settings, "REASONING_MAX_TOKENS_COMPACT", 1600) or 1600
                        ),
                    },
                    {
                        "label": "full",
                        "prompt": _build_agentic_flow_prompt(video_name, video_context),
                        "max_tokens": int(
                            getattr(settings, "REASONING_MAX_TOKENS_FULL", 2400) or 2400
                        ),
                    },
                ]
            last_exc: Exception | None = None
            for idx, attempt in enumerate(attempts, 1):
                try:
                    _log.info(
                        "  Agentic flow reasoning attempt %d/%d (%s, model=%s timeout=%.0fs max_tokens=%d)",
                        idx,
                        len(attempts),
                        attempt["label"],
                        model,
                        timeout_sec,
                        attempt["max_tokens"],
                    )
                    resp = httpx.post(
                        endpoint,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": attempt["prompt"]}],
                            "max_tokens": attempt["max_tokens"],
                            "temperature": 0.0,
                        },
                        timeout=timeout_sec,
                    )
                    resp.raise_for_status()
                    candidate = _strip_thinking_tokens(
                        resp.json()["choices"][0]["message"]["content"]
                    )
                    if _is_valid_agentic_flow_analysis(
                        candidate, simple=is_simple and attempt["label"] == "simple"
                    ):
                        llm_analysis = candidate
                        result["llm_used"] = True
                        _log.info("  [ok] Agentic flow analysis generated with %s", model)
                        break
                    if candidate:
                        _log.warning(
                            "  Agentic flow reasoning attempt %d returned incomplete output; falling back",
                            idx,
                        )
                except Exception as exc:
                    last_exc = exc
                    _log.warning("  Agentic flow reasoning attempt %d failed (%s)", idx, exc)
            if is_simple and not llm_analysis and api_url:
                try:
                    attempt = {
                        "label": "compact",
                        "prompt": _build_agentic_flow_prompt_compact(video_name, video_context),
                        "max_tokens": int(
                            getattr(settings, "REASONING_MAX_TOKENS_COMPACT", 1600) or 1600
                        ),
                    }
                    _log.info(
                        "  Agentic flow reasoning fallback (%s, model=%s timeout=%.0fs max_tokens=%d)",
                        attempt["label"],
                        model,
                        timeout_sec,
                        attempt["max_tokens"],
                    )
                    resp = httpx.post(
                        endpoint,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": attempt["prompt"]}],
                            "max_tokens": attempt["max_tokens"],
                            "temperature": 0.0,
                        },
                        timeout=timeout_sec,
                    )
                    resp.raise_for_status()
                    candidate = _strip_thinking_tokens(
                        resp.json()["choices"][0]["message"]["content"]
                    )
                    if _is_valid_agentic_flow_analysis(candidate, simple=False):
                        llm_analysis = candidate
                        result["llm_used"] = True
                        _log.info("  [ok] Agentic flow analysis generated with %s", model)
                except Exception as exc:
                    last_exc = exc
                    _log.warning("  Agentic flow reasoning fallback failed (%s)", exc)
            if not llm_analysis and last_exc is not None:
                raise last_exc
        except Exception as exc:
            _log.warning("  Agentic flow reasoning failed (%s) — using deterministic fallback", exc)

    if not llm_analysis:
        llm_analysis = _fallback_agentic_flow_analysis(video_context)
        result["model"] = "deterministic-fallback"

    elapsed = time.time() - t0
    write_agentic_flow_md(
        output_path,
        video_name,
        video_context.get("agentic_trace", []),
        elapsed,
        result["model"],
        llm_analysis,
        video_context,
    )
    _log_vram_snapshot("after reasoning sidecar use")
    result.update({"skipped": False, "elapsed_sec": elapsed, "output_path": str(output_path)})
    return result


# -- Step 22: video synthesis --------------------------------------------------


def step_video_synthesis(
    video_name: str,
    video_dir: Path,
    video_context: dict[str, Any],
    api_url: str,
    model: str,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Step 26: synthesise video ontology + narrative via Ollama/vLLM API.

    Uses all accumulated context from steps A–H as input.  No local model is
    loaded — this is a pure API call, so CLIP+DINO can remain offloaded.
    Writes ``video_synthesis.md`` and ``video_ontology.json``.
    """
    from ...steps.caption import _log_vram_snapshot
    from ...steps.report import write_video_synthesis_md

    result: dict[str, Any] = {"skipped": True, "ontology": {}, "narrative": ""}
    if not api_url:
        _log.info("  Synthesis skipped (no QWEN_API_URL / --qwen-api-url set)")
        return result

    try:
        import httpx
    except ImportError:
        _log.warning("  httpx unavailable — skipping video synthesis")
        return result

    from ...steps.caption import _compute_sidecar_timeout

    context_str = _build_context_prompt(video_name, video_context)
    # Cap context to avoid exceeding Ollama's default num_ctx (2048 tokens).
    # ~3000 chars ≈ 750 tokens, leaving headroom for the prompt suffix + output.
    if len(context_str) > 3000:
        context_str = context_str[:3000] + "\n[context truncated]"
    endpoint = f"{api_url.rstrip('/')}/chat/completions"
    _synthesis_timeout = _compute_sidecar_timeout(model, api_url, resources)
    # _compute_sidecar_timeout models cold-load time but not long generation.
    # Ontology needs ~512 tokens + prompt; narrative up to 1024 tokens.
    # Floor at 180s so generation completes even on a well-fitting model.
    _synthesis_timeout = max(_synthesis_timeout, 180.0)
    # Ollama-specific: expand context window so large prompts don't get a 500.
    _ollama_options = {"num_ctx": 8192}
    t0 = time.time()
    _log_vram_snapshot("before synthesis sidecar use")
    ontology: dict[str, Any] = {}
    narrative = ""

    # 1. Request structured ontology JSON
    ontology_prompt = (
        f"{context_str}\n\n"
        "Based on all the above observations, produce a structured video ontology "
        "as valid JSON with these fields:\n"
        "{\n"
        '  "domain": "string (e.g. outdoor_surveillance, urban_traffic, aerial_reconnaissance)",\n'
        '  "environment": "string (terrain/setting description)",\n'
        '  "primary_activities": ["list of main activities observed"],\n'
        '  "key_objects": ["list of key objects/entities"],\n'
        '  "temporal_structure": "string (how scene evolves over time)",\n'
        '  "scene_complexity": "low|medium|high",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Output only the JSON object, no other text."
    )
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": ontology_prompt}],
                "max_tokens": 512,
                "temperature": 0.1,
                "options": _ollama_options,
            },
            timeout=_synthesis_timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        ontology = json.loads(raw.strip())
        _log.info("  [ok] Video ontology generated  (domain=%s)", ontology.get("domain", "?"))
    except Exception as exc:
        _log.warning("  Ontology generation failed (%s)", exc)

    # 2. Request fine-grained narrative
    narrative_prompt = (
        f"{context_str}\n\n"
        "Write a fine-grained narrative description of this video in markdown. Cover:\n"
        "1. **Opening scene** — what is visible in the first frames\n"
        "2. **Main activity** — primary events, motion, and content\n"
        "3. **Environmental context** — terrain, lighting, setting details\n"
        "4. **Notable details** — specific objects, text, audio cues if any\n"
        "5. **Temporal evolution** — how the scene changes over time\n"
        "6. **Summary** — one-sentence overall description\n\n"
        "Be specific and grounded in the observations above. Use technical language "
        "appropriate for outdoor robotics and surveillance contexts."
    )
    try:
        resp = httpx.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": narrative_prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
                "options": _ollama_options,
            },
            timeout=_synthesis_timeout,
        )
        resp.raise_for_status()
        narrative = resp.json()["choices"][0]["message"]["content"].strip()
        _log.info("  [ok] Video narrative generated (%d chars)", len(narrative))
    except Exception as exc:
        _log.warning("  Narrative generation failed (%s)", exc)

    elapsed = time.time() - t0
    _log.info("  [ok] Video synthesis complete in %.1fs", elapsed)

    write_video_synthesis_md(
        video_dir / "video_synthesis.md",
        video_name,
        ontology,
        narrative,
        elapsed,
        model,
        video_context.get("local_threat", {}),
        video_context.get("policy_decision", {}),
        video_context.get("threat_primitives", {}),
        video_context.get("unidrive_analysis", []),
        video_context.get("physical_state", {}),
    )
    if ontology:
        (video_dir / "video_ontology.json").write_text(
            json.dumps(ontology, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _log.info("  [ok] Ontology saved → video_ontology.json")

    result.update(
        {"skipped": False, "ontology": ontology, "narrative": narrative, "elapsed_sec": elapsed}
    )
    _log_vram_snapshot("after synthesis sidecar use")
    return result


# -- Per-video orchestrator ----------------------------------------------------
