# ADR-0004: pycolmap for Poses, Gaussian-Splat Mapping for Dense 3D Output

Date: 2026-03-23  
Status: Accepted

## Context

The system needs:
- camera pose recovery from mission video
- dense 3D outputs for spatially anchored analytics and mapping

These are separate concerns and should remain separate in the architecture.

## Decision

Use:
- pycolmap-based SfM for camera pose recovery
- Gaussian-splat / mapper pipeline for dense 3D map generation

Current implementation lives under:
- `src/selfsuvis/pipeline/mapping/`
- worker / local workflow orchestration in `ssv_vdp/`

The degraded path is intentional: when pose recovery fails, the rest of the pipeline
still runs without dense 3D output.

## Consequences

Positive:
- Clear split between pose estimation and dense mapping
- CPU-capable SfM path plus optional heavier dense reconstruction
- Degraded mode keeps indexing and analysis usable even without successful mapping

Trade-offs:
- Dense mapping remains the slowest and operationally heaviest part of the stack
- SfM still fails on weak-overlap or low-texture footage
