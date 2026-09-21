"""Batch execution of a planned spec."""

import time
from pathlib import Path
from typing import Any

from ss_kernel.bindings import parse_binding, run_http, run_python
from ss_kernel.contracts import LedgerStep, PipelineSpec, StepManifest
from ss_kernel.plan import Plan


class ExecuteError(RuntimeError):
    """A step binding failed."""


def _legacy_remaining(spec: PipelineSpec, plan: Plan, manifests: dict[str, StepManifest]) -> bool:
    if not plan.remaining:
        return False
    remaining = plan.remaining_set()
    return all(
        manifests[step.manifest_id].wrap == "legacy"
        for step in spec.steps
        if step.step_id in remaining
    )


def execute_steps(
    spec: PipelineSpec,
    manifests: dict[str, StepManifest],
    plan: Plan,
    *,
    run_dir: Path,
    video: Path | None,
    extra: dict[str, Any] | None = None,
) -> tuple[list[LedgerStep], dict[str, Any]]:
    """Run remaining steps. Returns ledger rows and adapter extras (coverage, timings)."""
    extra_out: dict[str, Any] = dict(extra or {})
    rows: list[LedgerStep] = []
    ports: dict[str, Any] = {}
    if _legacy_remaining(spec, plan, manifests):
        from ss_kernel.adapters.local import run_legacy_batch

        extra_out.update(
            run_legacy_batch(
                spec,
                manifests,
                plan,
                run_dir=run_dir,
                video=video,
            )
        )
        timings = extra_out.get("timings") or {}
        for step in spec.steps:
            manifest = manifests[step.manifest_id]
            tier = next(item for item in manifest.tiers if item.name == step.tier)
            if not step.enabled:
                status = "skipped"
                latency = 0
            elif step.step_id in {item.step_id for item in plan.shed}:
                status = "shed"
                latency = 0
            else:
                status = "completed"
                latency = int(timings.get(step.step_id, tier.latency_ms))
            rows.append(
                LedgerStep(
                    step_id=step.step_id,
                    manifest_version=manifest.version,
                    tier=step.tier,
                    status=status,
                    latency_ms=latency,
                    vram_mb=tier.vram_mb,
                    quality=tier.quality,
                    degradation="" if status == "completed" else status,
                )
            )
        return rows, extra_out

    shed_ids = {item.step_id for item in plan.shed}
    remaining = plan.remaining_set()
    for step in spec.steps:
        manifest = manifests[step.manifest_id]
        tier = next(item for item in manifest.tiers if item.name == step.tier)
        if not step.enabled:
            rows.append(
                LedgerStep(
                    step_id=step.step_id,
                    manifest_version=manifest.version,
                    tier=step.tier,
                    status="skipped",
                    latency_ms=0,
                    vram_mb=tier.vram_mb,
                    quality=tier.quality,
                    degradation="skipped",
                )
            )
            continue
        if step.step_id in shed_ids:
            rows.append(
                LedgerStep(
                    step_id=step.step_id,
                    manifest_version=manifest.version,
                    tier=step.tier,
                    status="shed",
                    latency_ms=0,
                    vram_mb=tier.vram_mb,
                    quality=tier.quality,
                    degradation=next(
                        item.reason for item in plan.shed if item.step_id == step.step_id
                    ),
                )
            )
            continue
        if step.step_id not in remaining:
            continue
        context = {
            "run_dir": str(run_dir),
            "video": str(video) if video is not None else "",
            "step_id": step.step_id,
            "parameters": step.parameters,
            "ports": ports,
            "spec_id": spec.spec_id,
        }
        kind, target = parse_binding(manifest.binding)
        started = time.perf_counter()
        try:
            if kind == "python":
                produced = run_python(target, context)
            else:
                produced = run_http(target, context)
        except Exception as exc:
            raise ExecuteError(f"{step.step_id}: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        ports[step.step_id] = produced
        rows.append(
            LedgerStep(
                step_id=step.step_id,
                manifest_version=manifest.version,
                tier=step.tier,
                status="completed",
                latency_ms=elapsed_ms,
                vram_mb=tier.vram_mb,
                quality=tier.quality,
                degradation="",
            )
        )
    return rows, extra_out
