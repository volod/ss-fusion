"""Kernel unit tests: contracts, planning, shed order, and batch execution."""

from pathlib import Path

import pytest

from ss_kernel.adapters.local import build_legacy_argv
from ss_kernel.catalog import default_manifests
from ss_kernel.cli import main as kernel_main
from ss_kernel.contracts import PipelineSpec, RunLedger, StepManifest
from ss_kernel.engine import execute_run, hash_run_files
from ss_kernel.load import load_manifests, validate_spec
from ss_kernel.plan import PlanError, plan_run

ROOT = Path(__file__).resolve().parents[3]
OVER_BUDGET = ROOT / "tests" / "fixtures" / "kernel" / "over-budget.yaml"
DEMO_MANIFESTS = ROOT / "tests" / "fixtures" / "kernel" / "demo_manifests.yaml"
DEEP = ROOT / "specs" / "deep-investigation.yaml"


def test_over_budget_sheds_in_declared_order(tmp_path: Path) -> None:
    manifests = load_manifests(DEMO_MANIFESTS)
    spec = validate_spec(OVER_BUDGET, manifests)
    plan = plan_run(spec, manifests)
    assert plan.order == ["alpha", "beta", "gamma"]
    assert [item.step_id for item in plan.shed] == ["gamma", "beta"]
    assert plan.remaining == ["alpha"]
    assert plan.shed[0].reason == "over_budget"
    assert plan.shed[1].reason == "over_budget"

    ledger = execute_run(
        OVER_BUDGET,
        run_dir=tmp_path,
        run_id="shed-fixture",
        manifests_path=DEMO_MANIFESTS,
    )
    RunLedger.model_validate(ledger.model_dump(mode="json"))
    assert ledger.status == "completed"
    assert [item.step_id for item in ledger.shed_steps] == ["gamma", "beta"]
    by_id = {item.step_id: item for item in ledger.steps}
    assert by_id["alpha"].status == "completed"
    assert by_id["beta"].status == "shed"
    assert by_id["gamma"].status == "shed"
    assert (tmp_path / "alpha.txt").is_file()
    assert not (tmp_path / "beta.txt").is_file()
    assert not (tmp_path / "gamma.txt").is_file()
    assert (tmp_path / "ledger.json").is_file()
    reloaded = RunLedger.model_validate_json((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert reloaded.shed_steps[0].step_id == "gamma"


def test_deep_investigation_validates_and_plans() -> None:
    spec = validate_spec(DEEP)
    PipelineSpec.model_validate(spec.model_dump(mode="json"))
    manifests = default_manifests()
    for step in spec.steps:
        assert step.manifest_id in manifests
        StepManifest.model_validate(manifests[step.manifest_id].model_dump(mode="json"))
    plan = plan_run(spec, manifests)
    assert plan.shed == []
    assert plan.order[:4] == [
        "extract-frames",
        "index-vectors",
        "gemma-analysis",
        "florence-caption",
    ]
    assert "extract-frames" in plan.remaining
    assert "ssl-finetune" in plan.remaining
    assert "drone-detection" in plan.remaining
    assert "drone-audio" in plan.remaining
    assert "drau-eval" in plan.remaining
    assert "asr" not in plan.remaining
    assert "florence-caption" in plan.remaining


def test_cli_plan_over_budget(capsys: pytest.CaptureFixture[str]) -> None:
    code = kernel_main(["plan", str(OVER_BUDGET), "--manifests", str(DEMO_MANIFESTS)])
    assert code == 0
    out = capsys.readouterr().out
    assert "gamma" in out
    assert "beta" in out


def test_cycle_is_a_plan_error(tmp_path: Path) -> None:
    spec_path = tmp_path / "cycle.yaml"
    spec_path.write_text(
        (OVER_BUDGET.read_text(encoding="utf-8")).replace(
            "depends_on: []",
            "depends_on: [gamma]",
            1,
        ),
        encoding="ascii",
    )
    manifests = load_manifests(DEMO_MANIFESTS)
    spec = validate_spec(spec_path, manifests)
    with pytest.raises(PlanError, match="cycle"):
        plan_run(spec, manifests)


def test_vram_envelope_rejects_oversize_tier() -> None:
    spec = validate_spec(DEEP)
    spec.envelope.max_vram_mb = 1
    with pytest.raises(PlanError, match="vram_mb"):
        plan_run(spec, default_manifests())


def test_legacy_argv_matches_golden_skip_set() -> None:
    spec = validate_spec(DEEP)
    manifests = default_manifests()
    plan = plan_run(spec, manifests)
    argv = build_legacy_argv(
        spec, manifests, plan, Path("tests/assets/vid_testsrc.mp4"), Path("/tmp/out")
    )
    for flag in (
        "--no-qdrant",
        "--no-asr",
        "--no-ocr",
        "--no-depth",
        "--no-detection",
        "--no-world-model",
        "--no-unidrive",
        "--no-qwen",
        "--no-scenetok",
        "--no-yolo",
        "--no-sam",
        "--no-rfdetr",
        "--no-sfm",
        "--no-gsplat",
        "--no-distill",
        "--epochs",
        "1",
    ):
        assert flag in argv
    assert "--device" in argv
    assert "cuda" in argv
    assert "--no-drone-detection" not in argv
    assert "--no-drone-audio" not in argv
    assert "--no-drau-eval" not in argv


def test_min_completeness_rejects_over_shed(tmp_path: Path) -> None:
    spec_path = tmp_path / "tight.yaml"
    spec_path.write_text(
        OVER_BUDGET.read_text(encoding="utf-8").replace(
            "min_completeness: 0.0",
            "min_completeness: 0.9",
            1,
        ),
        encoding="ascii",
    )
    manifests = load_manifests(DEMO_MANIFESTS)
    spec = validate_spec(spec_path, manifests)
    with pytest.raises(PlanError, match="min_completeness"):
        plan_run(spec, manifests)


def test_http_binding_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http(target: str, context: dict) -> dict:
        marker = Path(context["run_dir"]) / "http.txt"
        marker.write_text(target, encoding="ascii")
        return {"path": str(marker)}

    monkeypatch.setattr("ss_kernel.execute.run_http", fake_http)
    spec_path = tmp_path / "http.yaml"
    spec_path.write_text(
        """
spec_id: http-demo
version: "1.0.0"
name: HTTP binding fixture
inputs: []
steps:
  - step_id: ping
    manifest_id: demo-http
    tier: cpu
    depends_on: []
    enabled: true
    parameters: {}
sla:
  latency_budget_ms: 1000
  min_completeness: 0.0
  shed_order: []
envelope:
  device_class: cpu
  max_vram_mb: 0
  max_latency_ms: 1000
outputs:
  - {name: ledger, path: ledger.json}
parameters: {}
""".strip()
        + "\n",
        encoding="ascii",
    )
    manifests_path = tmp_path / "http-manifests.yaml"
    manifests_path.write_text(
        """
manifests:
  - manifest_id: demo-http
    version: "1.0.0"
    name: HTTP ping
    ports_in: []
    ports_out: []
    tiers:
      - {name: cpu, device_class: cpu, latency_ms: 1, vram_mb: 0, quality: 1.0}
    binding: "http://127.0.0.1:9/step"
    skip_flags: []
    wrap: independent
""".strip()
        + "\n",
        encoding="ascii",
    )
    ledger = execute_run(
        spec_path,
        run_dir=tmp_path / "run",
        run_id="http-demo",
        manifests_path=manifests_path,
    )
    assert ledger.status == "completed"
    assert (tmp_path / "run" / "http.txt").read_text(encoding="ascii") == "http://127.0.0.1:9/step"


def test_hash_run_files_skips_underscore_cache_dirs(tmp_path: Path) -> None:
    keep = tmp_path / "output" / "keep.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("keep\n", encoding="ascii")
    hidden = tmp_path / "output" / "_cache" / "blob.bin"
    hidden.parent.mkdir(parents=True)
    hidden.write_text("blob\n", encoding="ascii")
    rows = hash_run_files(tmp_path)
    paths = [item.path for item in rows]
    assert "output/keep.txt" in paths
    assert "output/_cache/blob.bin" not in paths
