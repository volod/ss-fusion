# ss-fusion Specification

ss-fusion is the GPU-host research pipeline, model factory, and site-operations fusion runtime.
It publishes `ss-perception` and `ss-mapping` as packages that ss-video pins. It pins
[`volod/ss-common`](https://github.com/volod/ss-common) tag `v0.1.0`.

This document owns product behavior, boundaries, evaluations, and the
[capability registry](#capability-registry). The [forward plan](../impl/plan.md) owns work that
remains; [current implementation](../impl/current.md) owns what exists.

## Design principles

- **Share light, not heavy.** Runtime helpers live in ss-common. Perception and mapping are
  versioned packages published from this repository, never copied into ss-video.
- **No algorithm rewrite on extract.** Staging moves modules; it does not change fusion math,
  training, or the 36-step runner.
- **Evidence over prose.** Every capability declares an evaluation and a valid negative result.

## Research pipeline

The 36-step per-mission research workflow (`ssv_vdp`): perception, sensor fusion, tracking, 3D
reconstruction, SSL adaptation, distillation, threat inference, and reasoning audit. Console
scripts `ssv`, `ssv-export`, `ssv-gallery`, and the audio helpers are unchanged. Current state:
[local pipeline](../impl/current/local-pipeline.md).

**Evaluation.** A local run emits `analysis_summary.json` with modality coverage; step unit tests
pass. Valid negative result: a missing optional sidecar or model is recorded as degraded coverage,
not a crash.

## Fusion runtime

fusion-rt is a FastAPI process serving `/api/v1/*` and `/site/state`, `/site/threat`,
`/site/synthesis`, and `WS /site/stream`. It consumes contract sensor events from ss-sens and
camera events and scene captions from ss-video. It is part of this repository's shipped
research-and-operations surface; details live in
[current implementation](../impl/current.md#fusion-rt).

## Standalone CI

**Problem.** GitHub Actions runs a minutes-long light job (`make ci-github`: no torch, no locked
vision install). A clone of this repository may still fail a locked `uv sync` or the fusion-rt
Docker suite, which are not part of that job.

**Behavior.** After [volod/ss-fusion](https://github.com/volod/ss-fusion) exists, a clone of this
repository completes locked `uv sync --group dev`, `make ci`, and the fusion-rt Docker tests.
GitHub Actions stays the light job.

**Boundary.** Does not install the vision extra, does not re-run the CUDA `ssv` golden, and does
not change fusion algorithms. Compose files that still live only in ss-video are copied here if
the Docker suite needs them.

**Evaluation.** `make ci` is green on the clone; fusion-rt Docker tests are green; the log is under
`$DATA_DIR/standalone-build/`. Valid negative result: if GitHub-hosted runners cannot run the
Docker suite, the circle runs on a self-hosted clone of this repository and is recorded.

## Pipeline kernel

**Problem.** Two hard-wired orchestrators (`src/ssv_vdp/pipeline/runner.py` and the LangGraph
`graph.py`) run one fixed 36-step topology. Nothing describes a pipeline as data, so a different
set of steps, SLAs, output completeness, or resource envelope needs code changes, and runs cannot
be benchmarked against each other.

**Behavior.** A pipeline is a spec file: inputs, steps (each a versioned step with a selected
tier), SLA (window, latency budget, minimum output completeness, shed order, escalation rules),
and declared outputs. Each step has a manifest: typed ports, parameter schema, tiers with a cost
model per device class and a quality metric, and a binding (`python:`, `exec:`, `http:` sidecar,
`onnx:`). The kernel validates and plans a spec against a resource envelope, executes it, enforces
the SLA by shedding steps in declared order, and writes a run ledger (spec hash, step versions,
per-step latency and resources, quality metrics, degradations, content-hashed artifacts). A Python
reference kernel comes first; the spec, manifest, and ledger contracts plus a conformance suite
let a native Rust or Go kernel replace it on edge devices.

**Boundary.** The reference kernel runs batch mode first. Streaming mode, the `video-index` and
edge specs, the benchmark harness, a native kernel, and extraction to its own repository are later
work.

**Evaluation.** `deep-investigation.yaml` reproduces the legacy local run's modality coverage and
artifact list; the ledger validates against its contract; a fixture spec over its latency budget
sheds steps in declared order and records it. Valid negative result: steps that cannot be wrapped
without behavior change stay on the legacy orchestrator and are listed.

## Capability Registry

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 1 | `research-pipeline` | shipped | Local run emits `analysis_summary.json` with modality coverage; step unit tests | [Local pipeline](../impl/current/local-pipeline.md) |
| 2 | `standalone-ci` | planned | Locked `make ci` and fusion-rt Docker tests on a clone of this repository | -- |
| 3 | `pipeline-kernel` | planned | Spec reproduces legacy run; ledger contract; shed-order fixture | -- |

## Extending this specification

A capability gap is a product discovery, not an automatic refusal and not permission for silent
scope growth. Use this lifecycle in order:

1. State the problem in operator or domain terms.
2. Amend the owning section of this specification, including what the capability does not do.
3. Declare the measurement, acceptance signal, and valid negative result before implementation.
4. Add a `planned` registry row with that evaluation.
5. Put tasks under the capability in the implementation line; every task declares `Serves`.
6. Build and evaluate, document available behavior under current state, remove finished plan
   scope, and change the registry row to `shipped` with its implementation link.

When implementation reveals that an existing capability has the wrong boundary, update its section
instead of creating a workaround that the specification cannot explain.

## Specification and plan integrity

The registry and the [implementation plan](../impl/plan.md) are two views of one product:

- every task serves a registered capability and sits in its capability group;
- every capability declares an evaluation;
- every planned capability has at least one open task;
- every shipped capability links to current-state documentation;
- groups follow registry order in each task lane;
- every task declares the fields required by the [planning workflow](../guide/planning-workflow.md);
- lane statuses match whether an agent can finish independently or a human action gates acceptance;
- required tasks precede optional refinements within a capability;
- dependencies name open or recorded tasks, and start dependencies do not form a cycle.

`make lint-spec-plan` enforces these checks as part of `make ci`.
