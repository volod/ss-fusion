# Pipeline kernel

A pipeline is a spec file. `ss-kernel` validates it, plans a topological order against a
resource envelope, sheds steps in declared SLA order when the latency budget does not fit,
executes remaining steps, and writes a content-hashed run ledger.

Record: [0002-pipeline-kernel-kernel-reference-runtime](../records/0002-pipeline-kernel-kernel-reference-runtime.md).

## Layout

| Path | Role |
| --- | --- |
| `packages/ss-kernel` | Distribution `ss-kernel` 0.2.1, console script `ss-kernel` |
| `ss_kernel.contracts` | `PipelineSpec`, `StepManifest`, `RunLedger` on `ss_contracts.base.ContractModel` |
| `contracts/odcs/` | ODCS v3.1 YAML for those three contracts |
| `specs/deep-investigation.yaml` | Local-research spec (golden skip set) |
| `tests/fixtures/kernel/over-budget.yaml` | Shed-order fixture |
| `$DATA_DIR/ss-fusion/runs/<run-id>/ledger.json` | Run ledger |

The ODCS sources live in this repository. They are not in ss-common tag `v0.2.1`; runtime
models still subclass `ContractModel` so a later ss-common tag can take the YAML without
changing contract ids.

## Commands

| Command | Does |
| --- | --- |
| `ss-kernel validate specs/deep-investigation.yaml` | Load and check the spec against manifests |
| `ss-kernel plan specs/deep-investigation.yaml` | Print order, remaining, shed, estimated latency |
| `ss-kernel run specs/deep-investigation.yaml --video tests/assets/vid_testsrc.mp4` | Plan, execute, write the ledger |

`make ci` and `make ci-github` run `tests/unit/kernel` (no torch). The CUDA `ss-kernel run`
is acceptance evidence, not part of either gate. Torch is not a locked dependency; the
CUDA host uses the vision venv (`make test-unit` path).

## Planning and execution

Enabled steps are ordered by `depends_on`, walking through disabled nodes so a skipped
middle step does not break the chain. Selected tiers must fit `envelope.max_vram_mb` and
`device_class`. If the sum of remaining `latency_ms` exceeds `sla.latency_budget_ms`, the
planner drops `sla.shed_order` (and dependents) until the budget fits, then rejects the
plan when completeness would fall below `sla.min_completeness`.

Bindings: `python:module:function` and `http:` / `https:` URLs. Independent steps run one
binding each (the shed fixture). Every research step in `deep-investigation.yaml` has
`wrap: legacy` and runs as one `ssv --mode local` batch. `parameters.legacy_cli` is the
golden command. Manifest `skip_flags` are overlaid only for shed steps, not for
`enabled: false`.

Valid negative result: the 35 local-research steps stay on the monolith orchestrator.
They are listed in `ss_kernel.catalog`. LangGraph `graph.py` is unchanged.

## deep-investigation

The spec enables the steps the golden local command actually runs, including
drone-detection, drone-audio, and drau-eval. ASR, OCR, depth, detection, YOLO/SAM,
tracking, world-model, Qwen, UniDrive, SceneTok, Cosmos3, SfM/splat, and distill stay
`enabled: false` and are skipped by `legacy_cli` `--no-*` flags.

On this CUDA host, `ss-kernel run specs/deep-investigation.yaml --video tests/assets/vid_testsrc.mp4`
wrote `$DATA_DIR/ss-fusion/runs/kernel-deep-investigation/ledger.json` with status
`completed`, shed `[]`, coverage F/Q/A/O `100/0/0/0%`, `n_frames` 10. The video directory
has the same relative paths as the staged golden set except four optional-sidecar files
(`gemma_captions.md`, `runtime_cache/gemma_responses.json`, `video_ontology.json`,
`video_synthesis.md`) that the monolith only writes when Gemma or Qwen API URLs are set.
Drone-detection produced 832 files. Drone-audio wrote the skip report (one-class cache).
drau-eval skipped because no ONNX was exported. Ledger hashing omits `_`-prefixed cache
directories under the run dir.

## Tests

`tests/unit/kernel/test_kernel.py` checks the shed-order fixture and ledger contract, spec
validation and plan order, cycle and VRAM envelope errors, completeness floor, golden CLI
flags, and the mocked `http:` binding.
