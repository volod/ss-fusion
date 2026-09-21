"""ODCS v3.1 contract models for the pipeline kernel.

These models follow `ss_contracts.base.ContractModel`. The YAML sources live under
`contracts/odcs/` in this repository. They are not in ss-common tag v0.2.1; a later
ss-common release can take them without changing the ids.
"""

from typing import Any, ClassVar

from pydantic import Field

from ss_contracts.base import ContractModel, ContractRecord, ContractTimestamp
from ss_kernel.hashutil import sha256_hex

_ID = r"^[a-z][a-z0-9-]*$"
_SEMVER = r"^[0-9]+\.[0-9]+\.[0-9]+$"
_HEX64 = r"^[0-9a-f]{64}$"
_PORT_TYPE = r"^[a-z][a-z0-9_]*$"
_BINDING = r"^(python:[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*|https?://.+)$"
_DEVICE = r"^(cpu|cuda|edge)$"
_STEP_STATUS = r"^(completed|skipped|shed|failed)$"
_RUN_STATUS = r"^(completed|failed)$"


class SpecInput(ContractRecord):
    """One declared pipeline input."""

    name: str = Field(min_length=1, description="Input name.")
    type: str = Field(pattern=_PORT_TYPE, description="Port type (path, string, json).")
    required: bool = Field(description="Whether the input must be supplied.")


class SpecStep(ContractRecord):
    """One step in a pipeline spec: a manifest, selected tier, and dependencies."""

    step_id: str = Field(pattern=_ID, description="Step id unique in this spec.")
    manifest_id: str = Field(pattern=_ID, description="Step manifest id.")
    tier: str = Field(min_length=1, description="Selected tier name from the manifest.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step ids that must complete before this step.",
    )
    enabled: bool = Field(
        default=True,
        description="False excludes the step from the plan (skip, not shed).",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter values for this step; empty when none.",
    )


class SpecSla(ContractRecord):
    """Latency budget, completeness floor, and shed order."""

    latency_budget_ms: int = Field(
        ge=1,
        description="Maximum planned sum of selected-tier latency_ms.",
    )
    min_completeness: float = Field(
        ge=0,
        le=1,
        description="Minimum fraction of enabled steps that must remain after shedding.",
    )
    shed_order: list[str] = Field(
        description="Enabled step ids to drop first when the latency budget is exceeded.",
    )


class SpecEnvelope(ContractRecord):
    """Resource envelope the planner checks selected tiers against."""

    device_class: str = Field(pattern=_DEVICE, description="cpu, cuda, or edge.")
    max_vram_mb: int = Field(ge=0, description="VRAM ceiling for any selected tier.")
    max_latency_ms: int = Field(
        ge=1,
        description="Hard ceiling; must be at least sla.latency_budget_ms.",
    )


class SpecOutput(ContractRecord):
    """A declared output path relative to the run directory."""

    name: str = Field(min_length=1, description="Output name.")
    path: str = Field(min_length=1, description="POSIX path relative to the run directory.")


class PipelineSpec(ContractModel):
    """A pipeline as data: inputs, steps, SLA, envelope, and outputs."""

    CONTRACT_ID: ClassVar[str] = "pipeline-spec"
    CONTRACT_VERSION: ClassVar[str] = "1.0.0"
    SS_BINDING: ClassVar[dict[str, str]] = {"owner": "ss-fusion"}

    spec_id: str = Field(pattern=_ID, description="Pipeline spec id.")
    version: str = Field(pattern=_SEMVER, description="Spec version.")
    name: str = Field(min_length=1, description="Human-readable spec name.")
    inputs: list[SpecInput] = Field(description="Declared inputs.")
    steps: list[SpecStep] = Field(description="Steps in authoring order.")
    sla: SpecSla = Field(description="Latency budget and shed order.")
    envelope: SpecEnvelope = Field(description="Resource envelope.")
    outputs: list[SpecOutput] = Field(description="Declared outputs.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Spec-level parameters (legacy CLI flags for the local adapter).",
    )


class ManifestPort(ContractRecord):
    """A typed step port."""

    name: str = Field(min_length=1, description="Port name.")
    type: str = Field(pattern=_PORT_TYPE, description="Port type.")


class ManifestTier(ContractRecord):
    """Cost model and quality for one device class."""

    name: str = Field(min_length=1, description="Tier name.")
    device_class: str = Field(pattern=_DEVICE, description="cpu, cuda, or edge.")
    latency_ms: int = Field(ge=0, description="Nominal latency for planning.")
    vram_mb: int = Field(ge=0, description="Nominal VRAM for the envelope check.")
    quality: float = Field(ge=0, le=1, description="Quality metric in 0..1.")


class StepManifest(ContractModel):
    """Versioned step: ports, tiers, and a python: or http: binding."""

    CONTRACT_ID: ClassVar[str] = "step-manifest"
    CONTRACT_VERSION: ClassVar[str] = "1.0.0"
    SS_BINDING: ClassVar[dict[str, str]] = {"owner": "ss-fusion"}

    manifest_id: str = Field(pattern=_ID, description="Manifest id.")
    version: str = Field(pattern=_SEMVER, description="Manifest version.")
    name: str = Field(min_length=1, description="Human-readable step name.")
    ports_in: list[ManifestPort] = Field(description="Input ports.")
    ports_out: list[ManifestPort] = Field(description="Output ports.")
    tiers: list[ManifestTier] = Field(description="Tiers with cost and quality.")
    binding: str = Field(
        pattern=_BINDING,
        description="python:module:function or an http(s) URL.",
    )
    skip_flags: list[str] = Field(
        default_factory=list,
        description="ssv --no-* flags overlaid when this step is shed.",
    )
    wrap: str = Field(
        default="independent",
        pattern=r"^(independent|legacy)$",
        description="independent runs via the binding; legacy stays on the monolith.",
    )


class ShedRecord(ContractRecord):
    """One step dropped to meet the latency budget."""

    step_id: str = Field(pattern=_ID, description="Shed step id.")
    reason: str = Field(min_length=1, description="over_budget or depends_on_shed:<id>.")


class LedgerStep(ContractRecord):
    """Per-step execution record."""

    step_id: str = Field(pattern=_ID, description="Step id.")
    manifest_version: str = Field(pattern=_SEMVER, description="Manifest version used.")
    tier: str = Field(min_length=1, description="Selected tier.")
    status: str = Field(pattern=_STEP_STATUS, description="completed, skipped, shed, or failed.")
    latency_ms: int = Field(ge=0, description="Measured or planned latency.")
    vram_mb: int = Field(ge=0, description="Nominal VRAM from the selected tier.")
    quality: float = Field(ge=0, le=1, description="Tier quality metric.")
    degradation: str = Field(
        default="",
        description="Degradation note; empty when none.",
    )


class LedgerArtifact(ContractRecord):
    """One content-hashed output file relative to the run directory."""

    path: str = Field(min_length=1, description="POSIX path relative to the run directory.")
    sha256: str = Field(pattern=_HEX64, description="SHA-256 of the file, lowercase hex.")


class RunLedger(ContractModel):
    """Record of one planned and executed pipeline run."""

    CONTRACT_ID: ClassVar[str] = "run-ledger"
    CONTRACT_VERSION: ClassVar[str] = "1.0.0"
    SS_BINDING: ClassVar[dict[str, str]] = {"owner": "ss-fusion"}

    run_id: str = Field(min_length=1, description="Run identifier.")
    spec_id: str = Field(pattern=_ID, description="Pipeline spec id.")
    spec_version: str = Field(pattern=_SEMVER, description="Spec version.")
    spec_hash: str = Field(pattern=_HEX64, description="SHA-256 of the canonical spec JSON.")
    started_at: ContractTimestamp = Field(description="Run start time, UTC.")
    finished_at: ContractTimestamp = Field(description="Run finish time, UTC.")
    status: str = Field(pattern=_RUN_STATUS, description="completed or failed.")
    completeness: float = Field(
        ge=0,
        le=1,
        description="Enabled steps that were not shed, over enabled steps.",
    )
    plan_order: list[str] = Field(description="Topological order of enabled steps before shed.")
    shed_steps: list[ShedRecord] = Field(
        description="Steps dropped for the latency budget, in shed order.",
    )
    steps: list[LedgerStep] = Field(description="Per-step records including skipped and shed.")
    artifacts: list[LedgerArtifact] = Field(
        description="Content-hashed files under the run directory; empty when none.",
    )
    artifact_count: int = Field(ge=0, description="Number of hashed artifacts.")
    inventory_sha256: str = Field(
        pattern=_HEX64,
        description="SHA-256 of the canonical path+digest listing.",
    )
    coverage: str = Field(
        default="",
        description="Florence/Qwen/ASR/OCR coverage string; empty when not a local run.",
    )
    n_frames: int = Field(default=0, ge=0, description="Frame count from analysis_summary.")
    video_artifact_count: int = Field(
        default=0,
        ge=0,
        description="Legacy analysis_summary artifact_count; 0 when absent.",
    )
    video_artifact_list: list[str] = Field(
        default_factory=list,
        description="Relative paths under the video output directory.",
    )


CONTRACTS: dict[str, type[ContractModel]] = {
    "pipeline-spec": PipelineSpec,
    "step-manifest": StepManifest,
    "run-ledger": RunLedger,
}


def spec_hash(spec: PipelineSpec) -> str:
    """SHA-256 of the canonical JSON form of `spec`."""
    payload = spec.model_dump(mode="json", exclude_none=True)
    return sha256_hex(payload)


def inventory_hash(artifacts: list[LedgerArtifact]) -> str:
    """SHA-256 of sorted `path sha256` lines."""
    lines = [f"{item.path} {item.sha256}" for item in artifacts]
    lines.sort()
    return sha256_hex("\n".join(lines) + ("\n" if lines else ""))
