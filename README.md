# ss-fusion

Research pipeline, model factory, shared perception and mapping packages, and the fusion-rt
site-operations service.

This tree is a uv workspace. Pin ss-common at
[`volod/ss-common`](https://github.com/volod/ss-common) tag `v0.1.0`.
Workspace members are version `0.2.1`.

```bash
make ci-github
make ci
make test-fusion-rt
ssv --mode local --video tests/assets/vid_testsrc.mp4
.venv/bin/uvicorn selfsuvis.fusion_rt.app:app --host 0.0.0.0 --port 8001
```

Current behavior: [docs/impl/current.md](docs/impl/current.md).
Product intent: [docs/design/spec.md](docs/design/spec.md).
