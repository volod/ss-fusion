"""Ollama sidecar helpers: model resolution, unload, adaptive timeout."""

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.logging import get_logger

_log = get_logger("pipeline.local.caption")

# Preferred Gemma model order: smallest usable first so we never pick a 26B/31B
# when a lighter option is available.
_GEMMA_PREFERENCE_ORDER = [
    "gemma4:e4b",
    "gemma4:4b",
    "gemma3:4b",
    "gemma3:1b",
    "gemma4:12b",
    "gemma3:12b",
    "gemma4:26b",
    "gemma4:31b",
    "gemma3:27b",
]

_REASONING_PREFERENCE_ORDER = [
    "deepseek-r1:32b",
    "qwen3:32b",
    "qwen3:30b",
    "deepseek-r1:14b",
    "qwen3:14b",
    "deepseek-r1:8b",
    "qwen3:8b",
    "gemma3:27b",
    "gemma3:12b",
    "gemma4:12b",
    "gemma3:4b",
    "gemma4:4b",
    "gemma4:e4b",
    "gemma3:1b",
]


def _list_ollama_models(api_url: str) -> list[str]:
    """Return model names available in the Ollama instance at *api_url*."""
    try:
        import httpx

        base = api_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        resp = httpx.get(f"{base}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def _get_ollama_model_size_gb(model_name: str, api_url: str) -> float:
    """Return the on-disk size of *model_name* in GiB, or 0.0 if unavailable."""
    try:
        import httpx

        base = api_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        resp = httpx.get(f"{base}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            for m in resp.json().get("models", []):
                if m.get("name") == model_name:
                    size_bytes = m.get("size", 0)
                    return size_bytes / (1024**3)
    except Exception:
        pass
    return 0.0


def _estimate_model_size_gb_from_name(model_name: str) -> float:
    """Rough size estimate from model name tags when Ollama size is unavailable."""
    m = (model_name or "").lower()
    # ordered largest → smallest so first match wins
    for tag, gb in [
        ("671b", 420.0),
        ("405b", 250.0),
        ("72b", 45.0),
        ("70b", 44.0),
        ("32b", 20.0),
        ("31b", 19.0),
        ("30b", 19.0),
        ("27b", 17.0),
        ("26b", 16.0),
        ("14b", 9.0),
        ("12b", 8.0),
        ("8b", 5.5),
        ("7b", 5.0),
        ("e4b", 9.6),  # e4b is Gemma4 efficient-4bit ~9.6 GB
        ("4b", 3.5),
        ("3b", 2.5),
        ("2b", 1.8),
        ("1b", 1.0),
    ]:
        if tag in m:
            return gb
    return 5.0  # unknown: assume mid-size


def _compute_sidecar_timeout(
    model_name: str,
    api_url: str,
    resources: dict | None = None,
) -> float:
    """Return an adaptive timeout (seconds) for a sidecar inference request.

    The timeout scales with:
      - Model size (larger = slower to cold-load from disk)
      - VRAM vs model size ratio (model doesn't fit → offloads to RAM → much slower)
      - RAM size (low RAM = more swapping pressure)

    Override at any time with env var ``SELFSUVIS_SIDECAR_TIMEOUT_SEC``.

    Tier summary (model fits in VRAM, fast NVMe assumed for high-end systems):
      model < 0.5× VRAM  →  45 s   (comfortably fits, likely fast machine)
      model < 1.0× VRAM  →  90 s   (snug fit)
      model < 2.0× VRAM  →  180 s  (partial RAM offload)
      model ≥ 2.0× VRAM  →  300 s  (heavy offload / CPU-only)
    """
    import os as _os

    override = _os.environ.get("SELFSUVIS_SIDECAR_TIMEOUT_SEC", "").strip()
    if override:
        try:
            return max(10.0, float(override))
        except ValueError:
            pass

    if resources is None:
        try:
            from selfsuvis.pipeline.vision.registry import detect_resources

            resources = detect_resources()
        except Exception:
            resources = {}

    vram_gb = resources.get("vram_gb", 0.0)
    ram_gb = resources.get("ram_gb", 8.0)

    model_size_gb = _get_ollama_model_size_gb(model_name, api_url)
    if model_size_gb <= 0:
        model_size_gb = _estimate_model_size_gb_from_name(model_name)

    if vram_gb <= 0:
        # CPU-only: load time dominated by RAM bandwidth
        base = 60.0 + model_size_gb * 20.0
    else:
        ratio = model_size_gb / vram_gb
        if ratio < 0.5:
            base = 45.0
        elif ratio < 1.0:
            base = 90.0
        elif ratio < 2.0:
            base = 180.0
        else:
            base = 300.0

    # Low RAM machines swap more aggressively under memory pressure
    if 0 < ram_gb < 16:
        base *= 1.5

    return min(base, 600.0)


def _recommend_gemma_sidecar_models(resources: dict) -> tuple[str, str]:
    """Return recommended (analysis_model, reasoning_model) for current hardware.

    Analysis runs over sampled video frames and should stay relatively light.
    Reasoning runs once at the end and can use a larger long-thinking model.
    """
    vram = resources.get("vram_gb", 0.0)
    free_vram = resources.get("free_vram_gb", vram)
    ram = resources.get("ram_gb", 0.0)

    if free_vram >= 64 or vram >= 80:
        return "gemma4:26b", "deepseek-r1:32b"
    if free_vram >= 32 or vram >= 48:
        return "gemma4:12b", "qwen3:30b"
    if free_vram >= 18 or vram >= 24:
        return "gemma4:4b", "deepseek-r1:14b"
    if free_vram >= 10 or vram >= 12:
        return "gemma4:e4b", "qwen3:14b"

    # CPU / mixed RAM-heavy fallback. Keep analysis lighter; spend RAM on the final audit.
    if ram >= 96:
        return "gemma4:12b", "deepseek-r1:32b"
    if ram >= 64:
        return "gemma4:4b", "deepseek-r1:14b"
    if ram >= 32:
        return "gemma4:e4b", "qwen3:8b"
    return "gemma3:1b", "gemma4:e4b"


def _resolve_ollama_model_with_preferences(
    api_url: str,
    configured_model: str,
    *,
    preference_order: list[str],
    family_prefixes: tuple[str, ...],
    label: str,
) -> str:
    """Resolve a requested Ollama model against the instance model list."""
    available = _list_ollama_models(api_url)
    if not available:
        return configured_model
    if configured_model in available:
        return configured_model
    for preferred in preference_order:
        if preferred in available:
            _log.warning(
                "  %s model '%s' not found in Ollama; auto-selected '%s'. Pull the desired model with: ollama pull %s",
                label,
                configured_model,
                preferred,
                configured_model,
            )
            return preferred
    family_models = [m for m in available if m.startswith(family_prefixes)]
    if family_models:
        chosen = family_models[0]
        _log.warning(
            "  %s model '%s' not found; using first available family match: '%s'",
            label,
            configured_model,
            chosen,
        )
        return chosen
    return configured_model


def _resolve_ollama_gemma_model(api_url: str, configured_model: str) -> str:
    """Return the best available Gemma model for *api_url*.

    1. If *configured_model* is present in Ollama → use it.
    2. Otherwise scan available models and return the lightest Gemma by
       ``_GEMMA_PREFERENCE_ORDER``, or the first gemma* found.
    3. Falls back to *configured_model* (caller will get a 404 and fail clearly).
    """
    resolved = _resolve_ollama_model_with_preferences(
        api_url,
        configured_model,
        preference_order=_GEMMA_PREFERENCE_ORDER,
        family_prefixes=("gemma",),
        label="Gemma analysis",
    )
    if resolved == configured_model:
        available = _list_ollama_models(api_url)
        if available and not any(m.startswith("gemma") for m in available):
            _log.error(
                "No Gemma model found in Ollama. Pull one with: ollama pull gemma4:e4b\n"
                "Available models: %s",
                available,
            )
    return resolved


def _resolve_ollama_reasoning_model(api_url: str, configured_model: str) -> str:
    """Resolve the final reasoning model against Ollama availability."""
    return _resolve_ollama_model_with_preferences(
        api_url,
        configured_model,
        preference_order=_REASONING_PREFERENCE_ORDER,
        family_prefixes=("deepseek", "qwen", "llama", "gemma"),
        label="Reasoning",
    )


def _unload_ollama_model(api_url: str, model: str) -> bool:
    """Ask Ollama to evict *model* from VRAM by setting keep_alive=0.

    Only works when *api_url* points to an Ollama server (the /api/generate
    endpoint is Ollama-specific; vLLM will return 404 and we silently ignore
    that).  Returns True if the model was successfully unloaded.

    Typical VRAM freed: ~11–12 GiB for a 7B-param model, giving Florence-2
    (~1.5 GiB FP16) plenty of room to load locally.  Ollama auto-reloads the
    model on the next inference request (step 12), so no explicit warmup needed.
    """
    try:
        import httpx
    except ImportError:
        return False
    base = api_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        resp = httpx.post(
            f"{base}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=15.0,
        )
        if resp.status_code == 200:
            _log.info("  Ollama: '%s' unloaded from VRAM", model)
            return True
        _log.debug("  Ollama unload returned HTTP %d — may be vLLM (ignored)", resp.status_code)
    except Exception as exc:
        _log.debug("  Could not contact Ollama for unload: %s", exc)
    return False


def _unload_known_sidecars(pairs: list[tuple[str, str]]) -> int:
    """Unload all known Ollama sidecars from prior steps/runs when possible."""
    seen: set[tuple[str, str]] = set()
    unload_count = 0
    for url, model in pairs:
        if not url or not model:
            continue
        key = (url, model)
        if key in seen:
            continue
        seen.add(key)
        if _unload_ollama_model(url, model):
            unload_count += 1
    return unload_count
