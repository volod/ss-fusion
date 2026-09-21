"""ss-kernel CLI: validate, plan, and run a pipeline spec."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ss_kernel.catalog import default_manifests
from ss_kernel.engine import execute_run
from ss_kernel.load import load_manifests, validate_spec
from ss_kernel.plan import plan_run
from ss_kit.paths import data_path, discover_project_root


def _catalog(manifests: str | None):
    if manifests:
        return load_manifests(manifests)
    return default_manifests()


def _run_dir(run_id: str) -> Path:
    root = discover_project_root()
    return data_path(root, "ss-fusion", "runs", run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ss-kernel", description="Reference pipeline kernel")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate a spec against contracts")
    p_validate.add_argument("spec")
    p_validate.add_argument("--manifests")

    p_plan = sub.add_parser("plan", help="Print the planned order and shed list")
    p_plan.add_argument("spec")
    p_plan.add_argument("--manifests")

    p_run = sub.add_parser("run", help="Plan, execute, and write a run ledger")
    p_run.add_argument("spec")
    p_run.add_argument("--video", help="Input video (required for local-research specs)")
    p_run.add_argument("--manifests")
    p_run.add_argument("--run-id")
    p_run.add_argument("--run-dir", help="Override the run directory")

    args = parser.parse_args(argv)
    catalogs = _catalog(getattr(args, "manifests", None))
    if args.cmd == "validate":
        spec = validate_spec(args.spec, catalogs)
        print(f"ok {spec.spec_id} {spec.version} steps={len(spec.steps)}")
        return 0
    if args.cmd == "plan":
        spec = validate_spec(args.spec, catalogs)
        plan = plan_run(spec, catalogs)
        payload = {
            "spec_id": spec.spec_id,
            "order": plan.order,
            "remaining": plan.remaining,
            "shed": [item.model_dump() for item in plan.shed],
            "estimated_ms": plan.estimated_ms,
            "completeness": plan.completeness,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(args.run_dir) if args.run_dir else _run_dir(run_id)
    ledger = execute_run(
        args.spec,
        video=args.video,
        run_dir=dest,
        run_id=run_id,
        manifests_path=args.manifests,
    )
    print(f"run_id={ledger.run_id}")
    print(f"status={ledger.status}")
    print(f"ledger={dest / 'ledger.json'}")
    print(f"shed={[item.step_id for item in ledger.shed_steps]}")
    if ledger.coverage:
        print(f"coverage={ledger.coverage}")
    print(f"artifact_count={ledger.artifact_count}")
    return 0 if ledger.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
