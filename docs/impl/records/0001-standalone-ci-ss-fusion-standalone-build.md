# ss-fusion-standalone-build

## Task and scope

- Id / capability: `ss-fusion-standalone-build` / `standalone-ci`
- State: accepted
- Source: plan task; working tree at start of the task.
- Plan counts at start: 2 agent (both RUN NEEDED), 0 human; next eligible agent
  task `ss-fusion-standalone-build`; next human: none.
- Accepted task:

```markdown
#### ss-fusion-standalone-build

GitHub CI is a minutes-long light job, so the locked install and fusion-rt Docker circle are not
proven on a clone of this repository.

- Serves: `standalone-ci` -- [Standalone CI](../design/spec.md#standalone-ci)
- Agent status: RUN NEEDED
- Dependencies: none.
- User-visible outcome: a clone of [volod/ss-fusion](https://github.com/volod/ss-fusion) completes
  the full build circle: locked `uv sync`, `make ci`, and fusion-rt Docker tests.
- Scope boundary: in scope -- work in this published repository; `uv sync --locked --group dev`;
  `make ci`; the fusion-rt Docker suite (`make test-fusion-rt`, or compose and Dockerfiles added
  here if they still live only in ss-video); the log under `$DATA_DIR/standalone-build/`. Out of
  scope -- the vision extra and CUDA `ssv` golden run; ss-video flatten; creating or tagging the
  GitHub repository.
- Data and artifact paths: `$DATA_DIR/standalone-build/`.
- Execution path: clone `https://github.com/volod/ss-fusion`; `uv sync --locked --group dev` and
  `make ci`; fusion-rt Docker tests; the log file ends with PASS.
- Acceptance gates: locked sync completes on the published clone; `make ci` is green; fusion-rt
  Docker tests are green; the log is under `$DATA_DIR/standalone-build/`.
- Documentation target: `docs/impl/current.md`.
```

- Amendments: authorized by the task request: no backward compatibility with related
  modules (ss-video); public interfaces may change; workspace member versions are
  `0.2.1` for downstream pins. ss-common stays on the only published tag `v0.1.0`
  (GitHub tags page has no `v0.2.1`).

## Implementation

- Slim fusion-rt compose stack under `docker/fusion-rt/` (postgres, redis, mosquitto,
  fusion-rt, tests). Images install from this tree; no vision extra; no `ss-fusion`
  (langgraph). `make test-fusion-rt` and `make standalone-build` are the entrypoints.
- fusion-rt dropped the `ss-fusion` distribution dependency. `probability_union` lives
  in `selfsuvis.pipeline.core.prob`; `selfsuvis.pipeline.fusion.utils` re-exports it.
- HTTP ingest binds a timezone-aware `datetime` to asyncpg (`EventEnvelope.ts`).
- `test-fusion-rt-reset` falls back to a root Docker wipe when host uid cannot delete
  postgres volume files.
- Current-state page: [standalone-ci.md](../current/standalone-ci.md).

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Locked sync | `uv sync --locked --group dev` | pass; ss-common `v0.1.0` commit `4493f66` |
| `make ci` | `make ci` | pass; 512 passed, 2 deselected |
| fusion-rt Docker | `make test-fusion-rt` | pass; 3 passed in 2.30s |
| Log | `$DATA_DIR/standalone-build/build.log` ends with PASS | pass |

GitHub-hosted runners do not run the Docker suite (valid negative). This evidence is
from the self-hosted CUDA clone (Compose v5.5.1, RTX 4060 Ti). Compose project
`ss-fusion-rt-test` was brought down after the run.

## Audit handoff

`none identified` after reviewing the Docker suite, fusion-rt dependency cut,
ingest timestamp bind, Makefile reset fallback, and documentation shipping.

## Close or resume

All declared gates passed. `standalone-ci` is `shipped`. Plan counts after: 1 agent
(`kernel-reference-runtime`, RUN NEEDED), 0 human. Next eligible agent task:
`kernel-reference-runtime`.
