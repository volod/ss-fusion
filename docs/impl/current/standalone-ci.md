# Standalone CI

A clone of this repository completes locked `uv sync --group dev`, `make ci`, and
the slim fusion-rt Docker suite. GitHub Actions stays the minutes-long light job
(`make ci-github`, no torch, no locked vision install).

Record: [0001-standalone-ci-ss-fusion-standalone-build](../records/0001-standalone-ci-ss-fusion-standalone-build.md).

## Commands

| Command | Does |
| --- | --- |
| `uv sync --locked --group dev` | Locked workspace install (ss-common git tag `v0.2.1`) |
| `make ci` | Locked install, Ruff, doc-links, spec-plan, `--ci-light` unit tests |
| `make test-fusion-rt` | Postgres, Redis, Mosquitto, fusion-rt image from this tree, HTTP/MQTT tests |
| `make standalone-build` | The three gates above; log under `$DATA_DIR/standalone-build/build.log` |

GitHub CI is unchanged: `.github/workflows/ci.yml` runs `make ci-github` with a
10-minute timeout. `tests/unit/test_github_ci.py` and
`tests/unit/test_standalone_ci.py` assert that split.

## fusion-rt Docker suite

Compose lives under `docker/fusion-rt/` (not ss-video). The runtime image
(`docker/fusion-rt/Dockerfile`) installs `ss-perception` and `fusion-rt` from the
local tree plus ss-common `v0.2.1`. It does not install `ss-fusion` (langgraph)
or the vision extra. The tests image is an HTTP/MQTT client
(`tests/docker/test_fusion_rt.py`).

Services: postgres, redis, mosquitto, fusion-rt, tests. Host ports are not
published so the suite can run beside ss-video. Volumes and the PASS log live
under `$DATA_DIR/standalone-build/`.

fusion-rt no longer depends on the `ss-fusion` distribution. Shared
`probability_union` lives in `selfsuvis.pipeline.core.prob` (`ss-perception`);
`selfsuvis.pipeline.fusion.utils` re-exports it for the research pipeline.
HTTP ingest (`POST /api/v1/events/{modality}`) parses `ts` as datetime and binds
a timezone-aware value to asyncpg. `make test-fusion-rt-reset` deletes bind-mount
postgres files via a root container when the host user cannot.

## Evaluation

`make standalone-build` writes `$DATA_DIR/standalone-build/build.log` ending in
`PASS`. Valid negative result from the specification: GitHub-hosted runners do
not run this Docker suite; the circle is the self-hosted clone.
