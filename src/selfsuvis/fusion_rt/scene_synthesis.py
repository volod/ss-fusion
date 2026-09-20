"""Multi-modal scene synthesis for a monitored site.

SceneSynthesizer fuses heterogeneous sensor observations into a unified,
human-readable scene narrative and a structured scene snapshot:

  LoRaWAN sensor readings     -┐
  Frigate camera detections    ├-► SceneSynthesizer -► SceneSynthesis
  RtspCaptioner scene_timeline -┘     (LLM reasoning backend)
  Acoustic observations

The LLM call goes to the existing REASONING_API_URL backend so no new
infrastructure is required.  Results are cached for SCENE_SYNTHESIS_CACHE_SEC
(default 10 s) to avoid hammering the LLM on every API request.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from selfsuvis.pipeline.core import get_logger, settings

from .site_snapshot import CombinedSiteSnapshot, CombinedSiteState

logger = get_logger(__name__)

_DEFAULT_CACHE_SEC = 10.0

# -- Public model --------------------------------------------------------------


class SceneSynthesis(BaseModel):
    """Synthesised point-in-time scene understanding for the monitored site."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    narrative: str = ""
    threat_summary: str = ""
    dominant_activities: list[str] = Field(default_factory=list)
    environmental_conditions: dict[str, Any] = Field(default_factory=dict)
    active_objects: list[str] = Field(default_factory=list)
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    sources_used: list[str] = Field(default_factory=list)


# -- Synthesizer ---------------------------------------------------------------


class SceneSynthesizer:
    """Produce a unified SceneSynthesis from the current SiteState + MQTT captions.

    Args:
        aggregator:       CombinedSiteSnapshot holding live sensor + camera state.
        cache_sec:        Minimum seconds between LLM calls per site.
        reasoning_url:    Override for REASONING_API_URL.
        reasoning_model:  Override for REASONING_MODEL.
    """

    def __init__(
        self,
        aggregator: CombinedSiteSnapshot,
        db_pool: Any | None = None,
        cache_sec: float = _DEFAULT_CACHE_SEC,
        reasoning_url: str | None = None,
        reasoning_model: str | None = None,
    ) -> None:
        self._aggregator = aggregator
        self._captions: list[dict[str, Any]] = []
        self._caption_limit = 20
        self._cache_sec = cache_sec
        if reasoning_url is None:
            url = settings.REASONING_API_URL or settings.GEMMA_API_URL
        else:
            url = reasoning_url
        self._reasoning_url = (url or "").rstrip("/")
        self._reasoning_model = (
            reasoning_model or settings.REASONING_MODEL or settings.GEMMA_API_MODEL
        )
        self._cache: SceneSynthesis | None = None
        self._cache_ts: float = 0.0
        self._lock = asyncio.Lock()
        if db_pool is not None:
            logger.debug("SceneSynthesizer ignores db_pool; captions arrive over MQTT")

    async def ingest_caption(self, caption: Any) -> None:
        """Keep a rolling window of contract scene-caption messages."""
        if hasattr(caption, "model_dump"):
            data = caption.model_dump(mode="json")
        elif isinstance(caption, dict):
            data = dict(caption)
        else:
            return
        async with self._lock:
            self._captions.insert(0, data)
            del self._captions[self._caption_limit :]

    async def synthesize(self, force: bool = False) -> SceneSynthesis:
        """Return a (possibly cached) scene synthesis."""
        async with self._lock:
            if not force and self._cache and (time.monotonic() - self._cache_ts) < self._cache_sec:
                return self._cache

            state = await self._aggregator.get_state()
            captions = list(self._captions)
            result = await self._call_llm(state, captions)
            self._cache = result
            self._cache_ts = time.monotonic()
            return result

    # -- LLM call -------------------------------------------------------------

    async def _call_llm(
        self, state: CombinedSiteState, captions: list[dict[str, Any]]
    ) -> SceneSynthesis:
        prompt = _build_prompt(state, captions)
        sources = []

        if state.sensor_count:
            sources.append(f"{state.sensor_count} LoRaWAN sensors")
        if state.camera_count:
            sources.append(f"{state.camera_count} cameras")
        if captions:
            sources.append(f"{len(captions)} live captions")

        if not self._reasoning_url:
            return SceneSynthesis(
                narrative="No LLM backend configured (REASONING_API_URL not set).",
                sources_used=sources,
            )

        try:
            raw = await asyncio.wait_for(
                self._call_openai_compat(prompt), timeout=float(settings.REASONING_TIMEOUT_SEC)
            )
            parsed = _parse_llm_response(raw)
            parsed.sources_used = sources
            return parsed
        except asyncio.TimeoutError:
            logger.warning("SceneSynthesizer: LLM call timed out")
            return SceneSynthesis(narrative="(synthesis timeout)", sources_used=sources)
        except Exception as exc:
            logger.warning("SceneSynthesizer: LLM error: %s", exc)
            return SceneSynthesis(narrative=f"(synthesis error: {exc})", sources_used=sources)

    async def _call_openai_compat(self, prompt: str) -> str:
        payload = {
            "model": self._reasoning_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": settings.REASONING_MAX_TOKENS_COMPACT,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=float(settings.REASONING_TIMEOUT_SEC)) as client:
            resp = await client.post(f"{self._reasoning_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


# -- Prompt construction -------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a site-awareness AI for an outdoor monitoring system. "
    "Given multi-modal sensor data, produce a concise scene synthesis as JSON. "
    "Be factual. If data is absent, say so rather than inventing details."
)

_SYNTHESIS_SCHEMA = (
    '{"narrative":"<1-3 sentence summary of current site state>",'
    '"threat_summary":"<any active threats or anomalies, or none>",'
    '"dominant_activities":["<activity>"],'
    '"environmental_conditions":{"temperature_c":null,"humidity_pct":null,"co2_ppm":null},'
    '"active_objects":["<detected object classes>"],'
    '"alerts":[{"type":"<alert_type>","detail":"<detail>"}],'
    '"confidence":<0.0-1.0>}'
)


def _build_prompt(state: CombinedSiteState, captions: list[dict[str, Any]]) -> str:
    lines = ["## Current Site Sensor State\n"]

    if state.sensors:
        lines.append("### Environmental Sensors")
        for s in state.sensors:
            parts = [f"device={s.dev_eui}", f"seen={s.last_seen.isoformat()}"]
            if s.temperature_c is not None:
                parts.append(f"temp={s.temperature_c:.1f}°C")
            if s.humidity_pct is not None:
                parts.append(f"humidity={s.humidity_pct:.0f}%")
            if s.co2_ppm is not None:
                parts.append(f"CO2={s.co2_ppm:.0f}ppm")
            if s.motion is not None:
                parts.append(f"motion={s.motion}")
            if s.battery_v is not None:
                parts.append(f"battery={s.battery_v:.2f}V")
            lines.append("  " + ", ".join(parts))

    if state.cameras:
        lines.append("\n### Camera Detections (last 2 min)")
        for c in state.cameras:
            det_summary = ", ".join(
                f"{d['label']}({d['score']:.2f})" for d in c.recent_detections[:5]
            )
            lines.append(
                f"  camera={c.camera} objects=[{det_summary}] motion_events={c.total_events}"
            )

    if captions:
        lines.append("\n### Live Scene Captions (most recent)")
        for cap in captions[:5]:
            ts = cap.get("created_at") or cap.get("ts", "?")
            text = cap.get("caption") or ""
            mission = cap.get("mission_id", "")
            lines.append(f"  [{mission}@{ts}] {text[:200]}")

    lines.append(f"\nMotion detected by LoRaWAN: {state.active_motion}")
    lines.append(f"\nRespond in JSON matching this schema:\n{_SYNTHESIS_SCHEMA}")

    return "\n".join(lines)


def _parse_llm_response(text: str) -> SceneSynthesis:
    """Extract JSON from the LLM response and map it to SceneSynthesis."""
    # Find the JSON block (model may wrap it in markdown fences)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return SceneSynthesis(narrative=text.strip())
    try:
        data = json.loads(text[start:end])
        env = data.get("environmental_conditions") or {}
        return SceneSynthesis(
            narrative=str(data.get("narrative") or ""),
            threat_summary=str(data.get("threat_summary") or ""),
            dominant_activities=list(data.get("dominant_activities") or []),
            environmental_conditions={k: v for k, v in env.items() if v is not None},
            active_objects=list(data.get("active_objects") or []),
            alerts=list(data.get("alerts") or []),
            confidence=float(data.get("confidence") or 0.0),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return SceneSynthesis(narrative=text[:500].strip())
