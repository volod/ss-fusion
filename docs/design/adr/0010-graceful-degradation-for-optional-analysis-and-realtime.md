# ADR-0010: Prefer Graceful Degradation over Hard Failure for Optional Analysis Paths

Date: 2026-05-02  
Status: Accepted

## Context

The system combines core ingest/search functions with optional capabilities such
as 3D mapping, realtime sensor fusion, reranking, sidecar VLM calls, and ss-sens
stream analysis. In local and edge environments, some of these dependencies are
frequently unavailable, slow, or only partially configured.

Failing the entire run whenever an optional stage is unavailable would make the
product fragile and reduce the usefulness of partial results.

## Decision

Design optional analysis and realtime features to degrade gracefully:
- keep ingest and search usable when advanced stages fail
- reduce confidence when sensors or model services are stale or missing
- skip optional paths when dependencies are absent instead of crashing the whole
  application

Current implementation examples:
- realtime degraded-mode policies in `src/selfsuvis/pipeline/realtime/degraded_mode.py`
- soft-skip mapping and optional heavy stages in `src/selfsuvis/worker/main.py`
- rerank suppression during re-embedding in `src/selfsuvis/app/services/search.py`
- optional ss-sens MQTT consumer in `src/selfsuvis/pipeline/realtime/contract_consumer.py`
- reference realtime mapper fallback in `docker/realtime/docker-compose.realtime.yml`
- optional MAVSDK / ROS bridge runtimes in `src/selfsuvis/realtime/bridge_runtime.py`
- open-source realtime sidecar selection via `src/selfsuvis/realtime/adapters/`

## Consequences

Positive:
- Core indexing, storage, and retrieval remain usable under imperfect conditions
- Local development is practical without every heavyweight dependency installed
- Realtime outputs can express reduced confidence instead of false precision
- The base API and worker can run without MAVSDK, ROS, or external SLAM /
  occupancy engines installed

Trade-offs:
- Operators must monitor degraded states explicitly; success no longer means
  “all stages ran”
- Partial outputs increase the burden on reporting and diagnostics
- Soft-fail behavior can hide integration regressions if observability is weak
- Optional runtime dependencies now exist at several layers: Python imports,
  sidecar containers, bridge daemons, and mission-stage jobs
