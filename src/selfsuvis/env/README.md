# Environment configuration

Env files define defaults per environment. Set `APP_ENV` to select:

- `dev` — local development (default)
- `test` — integration tests (Docker)
- `prod` — production / `make up`

Files: `env/dev.env`, `env/test.env`, `env/prod.env`. Environment variables override file values.

For local overrides, create `.env` in the project root; it is loaded after the selected preset and overrides it.

Generate a resource-aware root `.env` with:

```bash
selfsuvis-env --env dev
selfsuvis-env --env prod --profile full
selfsuvis-env --interactive
```
