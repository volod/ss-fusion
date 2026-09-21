# syntax=docker/dockerfile:1
# fusion-rt only: FastAPI + correlator + MQTT. Built from this workspace.
# No vision extra or SAM.
FROM python:3.11-slim AS builder

WORKDIR /src
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PROJECT_ROOT=/src

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv "$VIRTUAL_ENV"
COPY packages/ss-perception /src/packages/ss-perception
COPY apps/fusion-rt /src/apps/fusion-rt
COPY docker/install-python.sh /tmp/install-python.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    sh /tmp/install-python.sh --runtime

FROM python:3.11-slim

WORKDIR /app
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
CMD ["uvicorn", "selfsuvis.fusion_rt.app:app", "--host", "0.0.0.0", "--port", "8000"]
