# Implementation Plan (forward work)

Forward-only: this file describes work that remains. Available behavior belongs in
[current-state documentation](current.md). Product behavior belongs in the
[specification](../design/spec.md).

Every task serves a capability from the [capability registry](../design/spec.md#capability-registry).
Capability groups follow registry order in both lanes.

## Agent Implementation Tasks

### Pipeline kernel -- `pipeline-kernel`

#### kernel-reference-runtime

A pipeline cannot be described as data, so a different set of steps, SLAs, or resource envelope
needs code changes, and runs cannot be benchmarked against each other.

- Serves: `pipeline-kernel` -- [Pipeline kernel](../design/spec.md#pipeline-kernel)
- Agent status: RUN NEEDED
- Dependencies: none.
- User-visible outcome: a pipeline is a spec file (steps, tiers, SLA, outputs) run by a Python
  reference kernel that writes a run ledger, and `specs/deep-investigation.yaml` reproduces
  today's local run.
- Scope boundary: in scope -- ODCS contracts for pipeline spec, step manifest, and run ledger
  (released through ss-common); `packages/ss-kernel` with spec validation, planning
  (topological order, resource-envelope check), batch execution, SLA accounting (latency budget,
  completeness, shed order), and ledger writing; step manifests wrapping existing steps (typed
  ports, parameters, tiers, `python:` or `http:` binding); `specs/deep-investigation.yaml`. Out of
  scope -- streaming mode, `video-index` and edge specs, the benchmark harness, a native kernel,
  extraction to its own repository, and removal of the LangGraph and monolith orchestrators.
- Data and artifact paths: `packages/ss-kernel/`, `specs/`; ledgers under
  `$DATA_DIR/ss-fusion/runs/<run-id>/ledger.json`.
- Execution path: `ss-kernel run specs/deep-investigation.yaml --video tests/assets/vid_testsrc.mp4`.
- Acceptance gates: the ledger validates against its contract; modality coverage and the artifact
  list match the legacy run; a fixture spec whose step exceeds the latency budget sheds steps in
  declared order and records it in the ledger.
- Documentation target: `docs/impl/current/kernel.md`.

## Human-Assisted Tasks

None.

## Future-task candidates

Deliberately not scheduled yet: `kernel-streaming-mode`, `kernel-video-index-spec`,
`kernel-benchmark-harness`, `kernel-native-edge`, `ss-kernel-extraction`.
