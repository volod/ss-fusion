# Current Implementation

ss-fusion is a uv workspace. `make ci` is the gate. Runtime data lives under `$DATA_DIR`
(default `.data/` in this project).

| Need | Read |
| --- | --- |
| Workspace members, commands, tests | this page |
| Locked install and fusion-rt Docker | [standalone-ci.md](current/standalone-ci.md) |
| 36-step local research pipeline | [local-pipeline.md](current/local-pipeline.md) |
| Pipeline kernel (forward) | [plan.md](plan.md) |

## Workspace

| Member | Distribution | Owns |
| --- | --- | --- |
| `packages/ss-perception` | `ss-perception` | `selfsuvis.pipeline.core`, `vision`, `labeling`, `models`, frame I/O (`media/{frames,ffmpeg,gps,dedup,heuristics,audio,fs_common,subprocess_common}`), vector index (`storage/{vector_store,qdrant,recent_index,common}`) |
| `packages/ss-mapping` | `ss-mapping` | `selfsuvis.pipeline.mapping` except ICP global-map fusion (`icp.py`, `mapper.py` stay in ss-video) |
| `packages/ss-fusion` | `ss-fusion` | `ssv_vdp`, `pipeline/fusion` (state estimation), `pipeline/training`, `pipeline/analysis`, `analytics/`, `visualization/`, `scripts/{prepare_models,sensors,scenetok_server,add_sensor_key,seed_test_events}`, `pipeline/storage/elastic.py` |
| `apps/fusion-rt` | `fusion-rt` | `selfsuvis.fusion_rt` FastAPI app |

Workspace members (`ss-perception`, `ss-mapping`, `ss-fusion`, `fusion-rt`) are version `0.2.1`.
`selfsuvis` is a pkgutil namespace package (`extend_path`) so ss-video can install
`ss-perception` and `ss-mapping` and keep `from selfsuvis.pipeline.core import ...`.
ss-common is git tag `v0.2.1`, not a path sibling. `[tool.ss-split] siblings = []`.

Console scripts (`ssv`, `ssv-export`, `ssv-gallery`, audio helpers, `ssv-models`) are on
`ss-fusion`. fusion-rt is `uvicorn selfsuvis.fusion_rt.app:app` on port 8001.

## fusion-rt

FastAPI title `fusion-rt` with `ss_kit.web` auth, fusion Postgres pool, correlator, and
MQTT consume of `sensor-event`, `sensor-state`, `camera-event`, and `scene-caption`.
Routes: `/api/v1/*`, `/site/state`, `/site/threat`, `/site/synthesis`, `WS /site/stream`.
OpenAPI: export from `selfsuvis.fusion_rt.app`. Unit tests live under `tests/unit/fusion_rt/`. The slim Docker suite is `make test-fusion-rt`.

## Commands

| Command | Does |
| --- | --- |
| `make ci` | `uv sync --locked --group dev`, Ruff, doc-links, spec-plan, `--ci-light` unit tests |
| `make ci-github` | Light venv (no torch); workspace smoke + fusion-rt units (GitHub, minutes) |
| `make test-fusion-rt` | Slim fusion-rt Docker suite (postgres, redis, mosquitto, no vision extra) |
| `make standalone-build` | Locked sync, `make ci`, `make test-fusion-rt`; log under `$DATA_DIR/standalone-build/` |
| `make test-unit` | Full unit tests (CUDA venv with vision deps) |
| `ssv --mode local --video tests/assets/vid_testsrc.mp4` | Local research run |
| `uvicorn selfsuvis.fusion_rt.app:app --host 0.0.0.0 --port 8001` | fusion-rt |

`docker/vllm/` is the reasoning/vision sidecar compose for Qwen. `docker/fusion-rt/` is the slim fusion-rt test stack.

## Tests

`tests/conftest.py` supports `--ci-light` (skips torch/cv2/ffmpeg paths). GitHub CI runs
`make ci-github` (workspace smoke + fusion-rt units + standalone-ci file checks;
`tests/unit/test_github_ci.py` asserts the workflow is a 10-minute single-Python job
and that `github-bootstrap` does not `uv sync --locked`). `make test-fusion-rt` runs the
slim fusion-rt Docker suite from this repository (no vision extra). `make ci` is the
locked local gate. `make test-unit` needs the local vision install. The locked install
plus fusion-rt Docker circle is `make standalone-build`; the log is
`$DATA_DIR/standalone-build/build.log`. Details:
[standalone CI](current/standalone-ci.md).

Golden local-run output for the extract is recorded under
`$DATA_DIR/split-check/ss-fusion-golden/` in the staging repository. On this CUDA host both
`ssv --mode local --video tests/assets/vid_testsrc.mp4` (legacy tree, then staged project,
identical skip flags) wrote `analysis_summary.json` with coverage F/Q/A/O 100/0/0/0%,
`artifact_count` 866, `n_frames` 10, and the same 867-file artifact set. `make ci-github` is the
GitHub gate; `make ci` is the locked local gate; the golden CUDA run is acceptance evidence, not
part of either.

## Deliberate gaps

The pipeline kernel (`packages/ss-kernel`, `specs/deep-investigation.yaml`) is
forward work in [plan.md](plan.md). ICP global-map fusion and the video API remain
in ss-video.
