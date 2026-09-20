# Task Record

Copy to `NNNN-<group>-<task-id>.md` using the
[naming rules](../../guide/planning-workflow.md#record-file-naming), index it in the
[records README](README.md), and replace this paragraph with the task title. Keep evidence
concise and the accepted task text complete.

## Task and scope

- Id / capability: `task-id` / `capability-id`
- State: active, blocked, or accepted (only after every required gate passes).
- Source: plan task or ad hoc request; commit at start and any unrelated dirty files.
- Plan counts at start: open tasks (agent, human) and the next eligible task per lane
  (`make plan-status`).
- Accepted task: the full original block, verbatim, in a fenced Markdown block.
- Amendments: none, or each revised block with its reason and who authorized it.

## Implementation

Changed modules and behavior, reused code, decisions and rejected alternatives, compatibility
effects (public API, CLI contract, env vars, `pyproject.toml` and lockfiles), limitations, and the
current-state page link.

## Acceptance evidence

| Gate | Exact command, test or artifact | Result and limit |
| --- | --- | --- |
| Each acceptance gate | Reproducible evidence | pass, fail, not-run, blocked or valid-negative |

Missing, mocked or stale evidence does not prove acceptance. Runtime artifacts stay under
`$DATA_DIR` and are cited by path, not copied here.

## Audit handoff

`none identified` with the reviewed scope, or one entry per `AUD-<task-id>-N`: the observation,
blocking or nonblocking, its evidence and impact, and exactly one owner (a plan task or a
registered capability that receives a new task).

## Close or resume

Gates passed and remaining, the next action, current-page and index updates, plan counts after,
and capability status changes.
