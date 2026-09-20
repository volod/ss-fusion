# ss-fusion developer entrypoints.
SHELL := /bin/bash
PROJECT_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
VENV := $(PROJECT_ROOT)/.venv
PY := $(VENV)/bin/python
PYTHON_VERSION ?= 3.11
DATA_DIR ?= .data
DATA_ROOT := $(if $(filter /%,$(DATA_DIR)),$(DATA_DIR),$(PROJECT_ROOT)/$(DATA_DIR))
PYTEST_CACHE := -o cache_dir=$(DATA_ROOT)/cache/pytest

unexport VIRTUAL_ENV

export RUFF_CACHE_DIR := $(DATA_ROOT)/cache/ruff
export UV_CACHE_DIR ?= $(DATA_ROOT)/uv-cache

.DEFAULT_GOAL := help

.PHONY: help bootstrap github-bootstrap venv lock format format-check lint test test-github test-unit \
 lint-doc-links lint-spec-plan plan-status ci ci-github

help: ## List available targets
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make \n"} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## Create/update .venv from uv.lock with workspace members and dev group
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required"; exit 1; }
	@uv sync --locked --group dev --python "$(PYTHON_VERSION)"

# Minutes-long GitHub install: editable workspace members without pulling torch,
# langgraph, or the vision extra. ss-common is the published git tag.
github-bootstrap: ## Light venv for GitHub CI (no torch, no uv sync --locked)
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required"; exit 1; }
	@uv venv "$(VENV)" --python "$(PYTHON_VERSION)" --clear
	@uv pip install --python "$(VENV)" --no-deps \
		-e "$(PROJECT_ROOT)/packages/ss-perception" \
		-e "$(PROJECT_ROOT)/packages/ss-mapping" \
		-e "$(PROJECT_ROOT)/packages/ss-fusion" \
		-e "$(PROJECT_ROOT)/apps/fusion-rt"
	@uv pip install --python "$(VENV)" \
		"pytest>=8.0" "pytest-asyncio>=0.23" "ruff>=0.16,<0.17" \
		"fastapi>=0.115" "pydantic>=2.8" "sse-starlette>=1.6" \
		"httpx>=0.27" "python-multipart>=0.0.9" \
		"PyYAML>=6.0" "python-dotenv>=1.0" "redis>=5.0" \
		"jinja2>=3.1" "aiomqtt>=2.3" "numpy>=1.26" "scipy>=1.10" "asyncpg>=0.29" \
		"ss-common[web,mqtt] @ git+https://github.com/volod/ss-common.git@v0.1.0"

venv: bootstrap ## Alias for bootstrap

lock: ## Refresh uv.lock after dependency changes
	@uv lock

format: ## Format production code and tests with Ruff
	@"$(VENV)/bin/ruff" format packages apps tests
	@"$(VENV)/bin/ruff" check --fix packages apps tests

format-check: ## Check Python formatting without changing files
	@"$(VENV)/bin/ruff" format --check packages apps tests docs

lint: format-check ## Run Ruff lint checks
	@"$(VENV)/bin/ruff" check packages apps tests

test: ## Light unit tests (no torch/cv2/ffmpeg); needs `make bootstrap`
	@"$(PY)" -m pytest tests/unit $(PYTEST_CACHE) --ci-light

test-github: ## Minimal GitHub suite: workspace smoke + fusion-rt HTTP units
	@"$(PY)" -m pytest tests/unit/test_workspace.py tests/unit/fusion_rt tests/unit/test_github_ci.py \
		$(PYTEST_CACHE) --ci-light -q

test-unit: ## Full unit tests (needs a CUDA venv with vision deps)
	@DATA_DIR="$(DATA_ROOT)" "$(PY)" -m pytest tests/unit $(PYTEST_CACHE) -v

lint-doc-links: ## Check that relative Markdown links and anchors resolve
	@"$(PY)" -m ss_kit.quality.doc_links --root "$(PROJECT_ROOT)"

lint-spec-plan: ## Check capability registry, task structure, status, and ordering
	@"$(PY)" -m ss_kit.quality.plan_integrity --root "$(PROJECT_ROOT)"

plan-status: ## Count tasks by lane/status and show the next eligible work
	@"$(PY)" -m ss_kit.quality.plan_summary --root "$(PROJECT_ROOT)"

ci: bootstrap lint lint-doc-links lint-spec-plan test ## Locked local gate (split-check)

ci-github: github-bootstrap lint lint-doc-links lint-spec-plan test-github ## GitHub CI (minutes, no torch)
