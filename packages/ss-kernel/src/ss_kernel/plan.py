"""Topological planning, resource-envelope checks, and SLA shed order."""

from dataclasses import dataclass, field

from ss_kernel.contracts import ManifestTier, PipelineSpec, ShedRecord, StepManifest


class PlanError(ValueError):
    """The spec cannot be planned."""


@dataclass
class PlannedStep:
    step_id: str
    manifest: StepManifest
    tier: ManifestTier
    enabled: bool
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    order: list[str]
    enabled: list[str]
    remaining: list[str]
    shed: list[ShedRecord] = field(default_factory=list)
    estimated_ms: int = 0
    completeness: float = 1.0

    def remaining_set(self) -> set[str]:
        return set(self.remaining)


def _tier(manifest: StepManifest, name: str) -> ManifestTier:
    for item in manifest.tiers:
        if item.name == name:
            return item
    raise PlanError(f"{manifest.manifest_id}: missing tier {name}")


def _topo(steps: list[PlannedStep]) -> list[str]:
    ids = [step.step_id for step in steps]
    id_set = set(ids)
    deps: dict[str, set[str]] = {}
    incoming: dict[str, int] = {step.step_id: 0 for step in steps}
    for step in steps:
        wanted = [dep for dep in _spec_deps(step) if dep in id_set]
        deps[step.step_id] = set(wanted)
        incoming[step.step_id] = len(wanted)
    children: dict[str, list[str]] = {step_id: [] for step_id in ids}
    for step_id, parents in deps.items():
        for parent in parents:
            children[parent].append(step_id)
    queue = [step_id for step_id in ids if incoming[step_id] == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children[node]:
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    if len(order) != len(ids):
        cyclic = [step_id for step_id, count in incoming.items() if count > 0]
        raise PlanError(f"cycle in depends_on: {', '.join(cyclic)}")
    return order


def _spec_deps(step: PlannedStep) -> list[str]:
    return list(step.depends_on)


def _walk_enabled_deps(
    step_id: str,
    by_id: dict[str, PlannedStep],
    enabled_set: set[str],
) -> list[str]:
    """Enabled dependencies, walking through disabled nodes."""
    found: list[str] = []
    stack = list(by_id[step_id].depends_on)
    seen: set[str] = set()
    while stack:
        dep = stack.pop()
        if dep in seen or dep not in by_id:
            continue
        seen.add(dep)
        if dep in enabled_set:
            found.append(dep)
        else:
            stack.extend(by_id[dep].depends_on)
    found.reverse()
    return found


def plan_run(spec: PipelineSpec, manifests: dict[str, StepManifest]) -> Plan:
    """Return topological order, envelope checks, and shed-order SLA cuts."""
    by_id: dict[str, PlannedStep] = {}
    for step in spec.steps:
        manifest = manifests[step.manifest_id]
        tier = _tier(manifest, step.tier)
        planned = PlannedStep(
            step_id=step.step_id,
            manifest=manifest,
            tier=tier,
            enabled=step.enabled,
            depends_on=list(step.depends_on),
        )
        by_id[step.step_id] = planned
        if step.enabled and tier.vram_mb > spec.envelope.max_vram_mb:
            raise PlanError(
                f"{step.step_id}: tier {step.tier} vram_mb {tier.vram_mb} "
                f"exceeds envelope {spec.envelope.max_vram_mb}"
            )
        if step.enabled and spec.envelope.device_class == "cpu" and tier.device_class == "cuda":
            raise PlanError(f"{step.step_id}: cuda tier not allowed on cpu envelope")

    enabled = [step.step_id for step in spec.steps if step.enabled]
    enabled_set = set(enabled)
    for step_id in enabled:
        by_id[step_id].depends_on = _walk_enabled_deps(step_id, by_id, enabled_set)
    enabled_steps = [by_id[step_id] for step_id in enabled]
    order = _topo(enabled_steps) if enabled_steps else []
    remaining = list(order)
    remaining_set = set(remaining)
    children: dict[str, list[str]] = {step_id: [] for step_id in remaining}
    for step_id in remaining:
        for dep in by_id[step_id].depends_on:
            if dep in remaining_set:
                children[dep].append(step_id)

    def estimated(ids: list[str]) -> int:
        return sum(by_id[step_id].tier.latency_ms for step_id in ids)

    shed: list[ShedRecord] = []
    budget = spec.sla.latency_budget_ms

    for candidate in spec.sla.shed_order:
        if estimated(remaining) <= budget:
            break
        if candidate not in remaining_set:
            continue
        drop = [candidate]
        stack = [candidate]
        while stack:
            node = stack.pop()
            for child in children[node]:
                if child in remaining_set and child not in drop:
                    drop.append(child)
                    stack.append(child)
        for step_id in drop:
            remaining.remove(step_id)
            remaining_set.remove(step_id)
            if step_id == candidate:
                shed.append(ShedRecord(step_id=step_id, reason="over_budget"))
            else:
                shed.append(ShedRecord(step_id=step_id, reason=f"depends_on_shed:{candidate}"))

    if estimated(remaining) > spec.envelope.max_latency_ms:
        raise PlanError(
            f"planned latency {estimated(remaining)} ms exceeds envelope.max_latency_ms "
            f"{spec.envelope.max_latency_ms}"
        )

    enabled_count = len(enabled)
    completeness = (len(remaining) / enabled_count) if enabled_count else 1.0
    if completeness < spec.sla.min_completeness:
        raise PlanError(
            f"completeness {completeness:.3f} below sla.min_completeness "
            f"{spec.sla.min_completeness}"
        )
    return Plan(
        order=order,
        enabled=enabled,
        remaining=remaining,
        shed=shed,
        estimated_ms=estimated(remaining),
        completeness=completeness,
    )
