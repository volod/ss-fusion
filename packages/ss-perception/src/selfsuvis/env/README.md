# Environment configuration

Env files define defaults per environment. Set `APP_ENV` to select:

- `dev` — local development (default)
- `test` — integration tests (Docker)
- `prod` — production / `make up`

Files: `projects/ss-fusion/packages/ss-perception/src/selfsuvis/env/dev.env`, `test.env`, and `prod.env`. Environment variables override file values.

For local overrides, create `.env` in the project root; it is loaded after the selected preset and overrides it.

Generate a resource-aware root `.env` with:

```bash
ssv-env --env dev
ssv-env --env prod --profile full
ssv-env --interactive
```
