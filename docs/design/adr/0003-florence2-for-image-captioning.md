# ADR-0003: Florence-2 as the Default Frame Captioner

Date: 2026-03-23  
Status: Accepted

## Context

The pipeline needs a local, automatable image-to-text captioner to support:
- searchable frame descriptions
- human-readable mission context
- caption-confidence-based downstream heuristics

## Decision

Use Florence-2 as the default captioning model in the vision pipeline.

Current implementation:
- `src/selfsuvis/pipeline/vision/florence.py`
- captions and confidence proxies feed search, analytics, and active-learning logic

The captioner is treated as the default fast path, while larger VLMs such as Gemma or
Qwen add richer reasoning in later steps rather than replacing Florence everywhere.

## Consequences

Positive:
- Fast enough for routine frame-level captioning
- Well integrated with the existing Python / transformers stack
- Produces a usable confidence proxy for downstream scoring

Trade-offs:
- Captions can still be generic on difficult outdoor scenes
- Confidence is heuristic, not a calibrated uncertainty measure
