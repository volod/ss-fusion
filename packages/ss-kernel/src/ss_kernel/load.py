"""Load pipeline specs and step manifests from YAML."""

from pathlib import Path
from typing import Any

import yaml

from ss_kernel.contracts import PipelineSpec, StepManifest


def _read_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if not text.isascii():
        raise ValueError(f"{path}: source must be ASCII")
    data = yaml.safe_load(text)
    if data is None:
        raise ValueError(f"{path}: empty YAML")
    return data


def load_spec(path: str | Path) -> PipelineSpec:
    """Validate a pipeline spec YAML file against the pipeline-spec contract."""
    spec_path = Path(path)
    data = _read_yaml(spec_path)
    if not isinstance(data, dict):
        raise ValueError(f"{spec_path}: spec must be a mapping")
    return PipelineSpec.model_validate(data)


def load_manifests(path: str | Path | None = None) -> dict[str, StepManifest]:
    """Load step manifests from a YAML list (or `{manifests: [...]}`)."""
    if path is None:
        from ss_kernel.catalog import default_manifests

        return default_manifests()
    manifest_path = Path(path)
    data = _read_yaml(manifest_path)
    if isinstance(data, dict) and "manifests" in data:
        rows = data["manifests"]
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"{manifest_path}: expected a list of manifests")
    manifests: dict[str, StepManifest] = {}
    for row in rows:
        item = StepManifest.model_validate(row)
        if item.manifest_id in manifests:
            raise ValueError(f"{manifest_path}: duplicate manifest_id {item.manifest_id}")
        manifests[item.manifest_id] = item
    return manifests


def validate_spec(
    spec_path: str | Path,
    manifests: dict[str, StepManifest] | None = None,
) -> PipelineSpec:
    """Load the spec and check that every step names a known manifest and tier."""
    spec = load_spec(spec_path)
    catalog = manifests if manifests is not None else load_manifests()
    seen: set[str] = set()
    for step in spec.steps:
        if step.step_id in seen:
            raise ValueError(f"duplicate step_id {step.step_id}")
        seen.add(step.step_id)
        manifest = catalog.get(step.manifest_id)
        if manifest is None:
            raise ValueError(f"{step.step_id}: unknown manifest {step.manifest_id}")
        if not any(tier.name == step.tier for tier in manifest.tiers):
            raise ValueError(f"{step.step_id}: unknown tier {step.tier}")
    step_ids = {step.step_id for step in spec.steps}
    for step in spec.steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                raise ValueError(f"{step.step_id}: unknown dependency {dep}")
    for shed_id in spec.sla.shed_order:
        if shed_id not in step_ids:
            raise ValueError(f"sla.shed_order: unknown step {shed_id}")
    if spec.envelope.max_latency_ms < spec.sla.latency_budget_ms:
        raise ValueError("envelope.max_latency_ms is below sla.latency_budget_ms")
    return spec
