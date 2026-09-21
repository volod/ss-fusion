#!/bin/sh
set -eu
# Install this repository inside Docker builders from the local workspace.
# ss-common comes from the published git tag. fusion-rt does not install
# ss-fusion (langgraph) or the vision extra (torch).
#
# Usage:
#   sh install-python.sh --runtime     fusion-rt (no torch, no langgraph)
SPEC="${1:-.}"
SS_COMMON_GIT="${SS_COMMON_GIT:-git+https://github.com/volod/ss-common.git@v0.2.1}"
ROOT="${PROJECT_ROOT:-/src}"

if [ "$SPEC" != "--runtime" ]; then
  echo "ERROR: only --runtime is supported in this image" >&2
  exit 1
fi

echo "fusion-rt runtime install (no torch, no langgraph)"
pip install --prefer-binary "ss-common[web,mqtt] @ ${SS_COMMON_GIT}"
pip install --no-deps \
  "${ROOT}/packages/ss-perception" \
  "${ROOT}/apps/fusion-rt"
pip install --prefer-binary \
  "fastapi>=0.115" "uvicorn[standard]>=0.27" "asyncpg>=0.29" "redis>=5.0" \
  "aiomqtt>=2.3" "sse-starlette>=1.6" "httpx>=0.27" "python-multipart>=0.0.9" \
  "numpy>=1.26" "pillow>=10.2" "pydantic>=2.8" "python-dotenv>=1.0" \
  "PyYAML>=6.0" "psutil>=5.9"
python -c "from selfsuvis.fusion_rt.app import app; print('fusion-rt import ok', app.title)"
