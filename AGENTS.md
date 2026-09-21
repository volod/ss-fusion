# AGENTS.md project rules

This file is the canonical instruction source for every coding agent in this repository.
Tool-specific files (`CLAUDE.md`, `GEMINI.md`, `.codex`) link here and keep only
integration-specific routing.

## Development guardrails

- **Git:** Do not commit, push, rewrite history, or revert user changes unless explicitly asked.
- **Scope:** Preserve unrelated work. Diagnose without changing code when the request is diagnostic.
- **Python:** Support Python 3.11 or newer. Use `uv`, `uv.lock`, and `pyproject.toml` for
  dependency management. Use Make targets for standard workflows.
- **Typing:** Do not add `from __future__ import annotations`; use normal annotations and
  `TYPE_CHECKING` imports when needed.
- **Paths:** Never hardcode machine-specific absolute paths. Resolve from the project root and
  honor `.env` and `DATA_DIR`; the uv and tool caches live under `$DATA_DIR`.
- **Secrets:** Never commit credentials or include them in logs, tests, fixtures, or documentation.
- **Dependencies:** Add the smallest justified dependency. Update `uv.lock` in the same change.
- **ASCII:** Use ASCII in logs, docs, comments, and generated shell output.
- Pin ss-common at git tag `v0.2.1`. Do not take a path sibling.
- Torch is not a locked dependency. CUDA wheels are installed by the host install path that
  matches the detected driver (same rule as the parent video repository).

## Package rules

This repository is a uv workspace:

| Member | Distribution | Import surface |
| --- | --- | --- |
| `packages/ss-perception` | `ss-perception` | `selfsuvis.pipeline.core`, `vision`, `labeling`, `models`, frame I/O, vector index |
| `packages/ss-mapping` | `ss-mapping` | `selfsuvis.pipeline.mapping` except ICP global-map fusion |
| `packages/ss-fusion` | `ss-fusion` | `ssv_vdp`, state estimation, training, analysis, analytics, visualization, model factory |
| `apps/fusion-rt` | `fusion-rt` | `selfsuvis.fusion_rt` FastAPI service |

`selfsuvis` is a pkgutil namespace package shared across members. Runtime data belongs under
`$DATA_DIR` (default `.data/` in this project). Never write a module-local `.data/` inside `src/`.

Heavy native builds cap parallelism with `ss-kit max-jobs`. Compiled wheels are cached under
`$DATA_DIR/wheels/<package-name>_<key>/`.

## Usual commands

- `make ci` — locked install, lint, doc-link and spec-plan checks, light unit tests
- `make ci-github` — GitHub gate: no torch, fusion-rt and workspace tests only
- `make test-fusion-rt` — slim fusion-rt Docker suite (postgres, redis, mosquitto)
- `make standalone-build` — locked sync, `make ci`, fusion-rt Docker tests; log under `$DATA_DIR/standalone-build/`
- `make test-unit` — full unit tests (needs a CUDA venv with vision deps)
- `ssv --mode local --video tests/assets/vid_testsrc.mp4` — local research pipeline
- `.venv/bin/uvicorn selfsuvis.fusion_rt.app:app --host 0.0.0.0 --port 8001` — fusion-rt

## Documentation lifecycle

| Question | Source of truth |
| --- | --- |
| What should the product do? | `docs/design/spec.md` (capability registry, boundaries, evaluation) |
| What work remains? | `docs/impl/plan.md` (forward-only) |
| What exists and where? | `docs/impl/current.md` and `docs/impl/current/` |
| What happened to a finished task? | `docs/impl/records/` |
| How is work performed? | `docs/guide/` and this file |

`docs/impl/plan.md` is FORWARD-ONLY: it contains only work that remains. Delivered behavior lives
under `docs/impl/current.md`. Product behavior and boundaries live in `docs/design/spec.md`.

After every product or developer-facing change, before reporting completion:

1. Record what exists in the narrowest current-state page.
2. Remove the completed task from `docs/impl/plan.md`; retain only residual future work.
3. Route anything surfaced during implementation exactly once: a chore is done now or dropped;
   an audit of the work just produced is performed as part of completion; more work for a
   registered capability becomes a task, `(optional)` when it is a refinement; a new product
   capability follows "Extending this specification" in `docs/design/spec.md` first.
4. Update current-doc indexes when adding a page and run `make lint-doc-links`.
5. Compare plan task counts before and after and state which capabilities moved.

Do not put completion notes, dates, measurements, or history in the plan.

## Task lanes and integrity

The plan has two lanes, **Agent Implementation Tasks** (`CLEAR`, `RUN NEEDED`) and
**Human-Assisted Tasks** (`BLOCKED BY HUMAN`, `HUMAN-GATED`, plus a `Human step` field). Task
fields, ordering, dependencies, and records are defined in the
[planning workflow](docs/guide/planning-workflow.md). `make lint-spec-plan` enforces them; fix
document disagreements when it fails and do not loosen the checker. `make plan-status` names the
next eligible task per lane.

## Completion discipline

Before declaring a plan task complete: keep a task record under `docs/impl/records/`, run the
declared tests and `make ci`, update current docs, remove finished plan scope, and inspect
`git status`. Confirm that only intended files changed and that no process, port, temporary
scaffold, or external resource started by the work remains active. Runtime artifacts under
`DATA_DIR` are evidence; keep them unless the task says otherwise.
