# kernel-reference-runtime

## Task and scope

- Id / capability: `kernel-reference-runtime` / `pipeline-kernel`
- State: accepted
- Source: plan task; working tree at start of the task included standalone-ci work
  that is recorded in `0001`.
- Plan counts at start: 1 agent (RUN NEEDED), 0 human; next eligible agent
  task `kernel-reference-runtime`; next human: none.
- Accepted task:

```markdown
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
```

- Amendments: ODCS YAML is authored in this repository using ss-common v0.2.1
  tooling rules; runtime models subclass `ss_contracts.base.ContractModel`. The
  three contracts are not in ss-common tag `v0.2.1`; they stay here until a later
  ss-common release can carry them.

## Implementation

- Workspace member `packages/ss-kernel` (0.2.1), CLI `ss-kernel`, first-party
  `ss_kernel`. GitHub bootstrap installs it editable with no torch.
- Contracts: `pipeline-spec`, `step-manifest`, `run-ledger` under `contracts/odcs/`
  and `ss_kernel.contracts`.
- Planner: topological order, VRAM and device-class envelope, SLA shed order,
  completeness floor. Disabled middle nodes are walked so the enabled chain stays
  connected.
- Execution: independent `python:` / `http:` bindings; `wrap: legacy` research
  steps run as one `ssv --mode local` batch. `legacy_cli` is the golden command;
  manifest `skip_flags` apply only to shed steps.
- `specs/deep-investigation.yaml` encodes the golden skip set. All 35 research
  steps remain on the monolith (valid negative).
- Current-state page: [kernel.md](../current/kernel.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Ledger contract | `tests/unit/kernel/test_kernel.py` plus CUDA `ledger.json` | pass; `RunLedger.model_validate` on `$DATA_DIR/ss-fusion/runs/kernel-deep-investigation/ledger.json` |
| Shed-order fixture | `tests/fixtures/kernel/over-budget.yaml` | pass; sheds `gamma` then `beta`; completeness floor rejects `min_completeness: 0.9` |
| Legacy coverage / artifacts | `ss-kernel run specs/deep-investigation.yaml --video tests/assets/vid_testsrc.mp4` | pass; coverage `100/0/0/0%`, `n_frames` 10, shed `[]`; video files match the staged golden set except four Gemma/Qwen sidecar files when those APIs are unset; drone-detection 832 files; drone-audio skip report |
| `make ci` | `make ci` | pass; 521 passed, 2 deselected |

## Audit handoff

`none identified` after reviewing contracts, planner, legacy argv mapping, CUDA
ledger, GitHub bootstrap, and documentation shipping. Residual not scheduled:
streaming, video-index/edge specs, benchmark harness, native kernel, extraction
to its own repository.

## Close or resume

All declared gates passed. `pipeline-kernel` is `shipped`. Plan counts after: 0
agent, 0 human. Future-task candidates remain listed in `plan.md` and are not
scheduled.
