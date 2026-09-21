"""Validate, plan, execute, and persist a run ledger."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from ss_kernel.catalog import default_manifests
from ss_kernel.contracts import (
    LedgerArtifact,
    PipelineSpec,
    RunLedger,
    inventory_hash,
    spec_hash,
)
from ss_kernel.execute import execute_steps
from ss_kernel.hashutil import sha256_file
from ss_kernel.load import load_manifests, validate_spec
from ss_kernel.plan import Plan, plan_run

LEDGER_NAME = "ledger.json"


def _manifests(path: str | Path | None) -> dict:
    if path is None:
        return default_manifests()
    return load_manifests(path)


def hash_run_files(run_dir: Path) -> list[LedgerArtifact]:
    artifacts: list[LedgerArtifact] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == LEDGER_NAME and path.parent == run_dir:
            continue
        rel = path.relative_to(run_dir)
        if any(part.startswith("_") for part in rel.parts[:-1]):
            continue
        artifacts.append(LedgerArtifact(path=rel.as_posix(), sha256=sha256_file(path)))
    return artifacts


def build_ledger(
    spec: PipelineSpec,
    plan: Plan,
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    step_rows: list,
    artifacts: list[LedgerArtifact],
    extra: dict | None = None,
) -> RunLedger:
    extra = extra or {}
    enabled_count = len(plan.enabled)
    completeness = (len(plan.remaining) / enabled_count) if enabled_count else 1.0
    payload = RunLedger(
        run_id=run_id,
        spec_id=spec.spec_id,
        spec_version=spec.version,
        spec_hash=spec_hash(spec),
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        completeness=completeness,
        plan_order=list(plan.order),
        shed_steps=list(plan.shed),
        steps=list(step_rows),
        artifacts=artifacts,
        artifact_count=len(artifacts),
        inventory_sha256=inventory_hash(artifacts),
        coverage=str(extra.get("coverage") or ""),
        n_frames=int(extra.get("n_frames") or 0),
        video_artifact_count=int(extra.get("video_artifact_count") or 0),
        video_artifact_list=list(extra.get("video_artifact_list") or []),
    )
    RunLedger.model_validate(payload.model_dump(mode="json"))
    return payload


def write_ledger(run_dir: Path, ledger: RunLedger) -> Path:
    path = run_dir / LEDGER_NAME
    path.write_text(
        ledger.model_dump_json(indent=2) + "\n",
        encoding="ascii",
    )
    return path


def execute_run(
    spec_path: str | Path,
    *,
    video: str | Path | None = None,
    run_dir: str | Path,
    run_id: str | None = None,
    manifests_path: str | Path | None = None,
) -> RunLedger:
    """Plan and execute `spec_path`, writing `ledger.json` under `run_dir`."""
    catalogs = _manifests(manifests_path)
    spec = validate_spec(spec_path, catalogs)
    plan = plan_run(spec, catalogs)
    dest = Path(run_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ident = run_id or uuid.uuid4().hex
    started = datetime.now(UTC)
    status = "completed"
    extra: dict = {}
    try:
        rows, extra = execute_steps(
            spec,
            catalogs,
            plan,
            run_dir=dest,
            video=Path(video) if video is not None else None,
        )
    except Exception:
        status = "failed"
        finished = datetime.now(UTC)
        ledger = build_ledger(
            spec,
            plan,
            run_id=ident,
            started_at=started,
            finished_at=finished,
            status=status,
            step_rows=[],
            artifacts=hash_run_files(dest),
            extra=extra,
        )
        write_ledger(dest, ledger)
        raise
    finished = datetime.now(UTC)
    ledger = build_ledger(
        spec,
        plan,
        run_id=ident,
        started_at=started,
        finished_at=finished,
        status=status,
        step_rows=rows,
        artifacts=hash_run_files(dest),
        extra=extra,
    )
    write_ledger(dest, ledger)
    return ledger
