# Planning Workflow Guide

The documentation lifecycle keeps product intent, future work, and available behavior separate:

| Question | Source of truth |
| --- | --- |
| What should the product do? | [Specification](../design/spec.md) |
| What work remains? | [Implementation plan](../impl/plan.md) |
| What exists and where? | [Current implementation](../impl/current.md) |
| How should work be performed? | This guide and [AGENTS.md](../../AGENTS.md) |

## The discovery circle

1. **Observe a need.** State what a user or operator cannot do or trust.
2. **Classify it once.** Choose chore, audit of the current work, extension of a registered
   capability, or a new capability.
3. **Specify a capability gap.** For a new capability, define behavior, boundary, evaluation, and a
   valid negative result in the specification before code.
4. **Register and schedule.** Add a `planned` registry row, then tasks in the correct plan lane.
5. **Build and evaluate.** Implement the smallest bounded slice and run its declared checks.
6. **Close the loop.** Move available behavior to current-state docs, remove finished plan scope,
   update the registry, and classify anything newly surfaced.

A chore is handled during the change when necessary or dropped. An audit of the work just produced
is part of completion and does not become a permanent task. More work for an existing capability
may become a task; mark a refinement `(optional)`. A new capability always returns to the
specification.

## Task lanes

Use **Agent Implementation Tasks** when an agent can reach acceptance using repository fixtures,
deterministic tools, or an authorized non-interactive run:

- `CLEAR`: code, tests, and docs can finish locally.
- `RUN NEEDED`: implementation is deterministic, but acceptance includes a declared heavier run
  (Docker stack, local pipeline run, multi-arch image build).

Use **Human-Assisted Tasks** when acceptance itself needs a person or an authority unavailable to
an agent:

- `BLOCKED BY HUMAN`: an agent prepares support, but a human-provided artifact gates completion
  (hardware, a pushed repository, a physical deployment).
- `HUMAN-GATED`: the outcome is human judgment, authorization, private access, or spend approval.

Add `Research: yes` when the path is uncertain and a well-supported negative result is acceptable.
Research is not a reason to omit scope or acceptance criteria.

A dependency on a task in the other lane is written out in the `Dependencies` line as a
cross-lane block.

## Dependencies

The `Dependencies` field names other tasks by their backticked ids. Every id named before the
first of these markers is a start prerequisite: the task becomes eligible only once that task has
left the plan.

- `Optional:` -- soft inputs the task uses when present, such as example flows or artifacts.
- `Cross-lane note:` -- an other-lane task that gates acceptance evidence or a default flip, not
  the start of the work (for example, real hardware or a human sign-off after an agent build).
- `Blocks:` -- work that this task gates.

Ids after a marker are informational. Every named id must be an open task or a finished task with
a record under `docs/impl/records/`, and start prerequisites must not form a cycle. Write
`none.` when a task has no prerequisites.

## Task shape

Task ids are stable lowercase kebab-case slugs. A task heading may end with `(optional)`.

```markdown
### Capability name -- `capability-id`

#### stable-task-id

Describe the unresolved operator problem in present or future tense.

- Serves: `capability-id` -- [Specification section](../design/spec.md#section)
- Agent status: CLEAR
- Dependencies: none.
- User-visible outcome: State what becomes possible or trustworthy.
- Scope boundary: State what is in scope and explicitly out of scope.
- Data and artifact paths: Name repository-relative or `$DATA_DIR` locations.
- Execution path: Name modules, fixtures, commands, and any declared run.
- Acceptance gates: State deterministic checks and the negative-result rule.
- Documentation target: The current-state page that receives the result.
```

Human-lane tasks add a `Human step` line after `User-visible outcome` naming the action only a
person can take; their `Scope boundary` names what the agent prepares.

## Ordering

Capability groups follow the registry. Within a group:

1. hard prerequisites;
2. changes to inputs that later evaluation consumes;
3. required work before optional refinements;
4. cheap deterministic work before expensive runs.

If priorities change, edit the registry order and move the same groups in both lanes. Do not encode
priority in task wording. Take the first required task in the earliest group that has one.

## Task records

Each task started from the plan gets a record under [`docs/impl/records/`](../impl/records/README.md),
copied from the [template](../impl/records/template.md). The record keeps the accepted task text
verbatim, the implementation notes, and the acceptance evidence after the task leaves the plan.
Current-state pages link the record; the plan never does.

### Record file naming

Name a record `NNNN-<group>-<task-id>.md`: `NNNN` is the next unused four-digit sequence from the
[records index](../impl/records/README.md), `<group>` is the capability id the task serves, and
`<task-id>` is the plan task id. Ad hoc work that did not come from the plan uses `adhoc` as the
group. Add the record to the index table in the same change.

## Completion transition

Before reporting completion:

1. Count plan tasks by lane.
2. Record behavior, locations, commands, tests, and results in the narrowest current-state page.
3. Remove the finished task and retain only genuinely open residual scope.
4. Mark a newly complete capability `shipped` and add its current-documentation link.
5. Run the integrity checks and the tests the task declared.
6. Close the task record with its evidence and add it to the records index.
7. Count tasks again and report which capabilities moved.

Current-state pages retain decisions and results. The plan never becomes a changelog.

## Checks

| Command | What it enforces |
| --- | --- |
| `make lint-spec-plan` | Registry rows, task fields (`Human step` in the human lane only), lane statuses, group placement and registry order, required before optional, known and acyclic dependencies, no history or dates in the plan |
| `make lint-doc-links` | Every relative link and heading anchor in `docs/`, `README.md`, `AGENTS.md`, and the tool adapters resolves |
| `make plan-status` | Task counts by lane and status, and the next eligible task per lane |

Both lint targets run inside `make ci`. The implementation lives in `ss_kit.quality`
from ss-common tag `v0.1.0`. When a check fails, fix the documents, not the checker.
