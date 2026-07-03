# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && python -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Dep-layer cache: pinned transitive graph lives in requirements.lock and only
# changes when deps are refreshed. Installing deps before copying the source
# keeps the (expensive) torch/faiss/sentence-transformers wheels cached across
# ordinary app-code edits.
COPY requirements.lock ./
RUN "${VIRTUAL_ENV}/bin/pip" install -r requirements.lock

# App-install layer: fast (metadata-only) because all deps are already present.
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
RUN "${VIRTUAL_ENV}/bin/pip" install --no-deps .

FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CV_ARXIV_CONFIG=/app/config.yaml \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/appuser --shell /bin/bash appuser \
    && install -d -o appuser -g appuser /app/instance

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser config.example.yaml run.py wsgi.py ./
# Baked config for the docker-compose `local-ai` profile's AI service, selected
# via CV_ARXIV_CONFIG so `docker compose --profile local-ai up` needs no host file.
COPY --chown=appuser:appuser config.local-ai.example.yaml ./config.local-ai.yaml

USER appuser

VOLUME ["/app/instance"]

EXPOSE 5000

# Probe the cheap /healthz liveness/readiness endpoint (200 when DB + FAISS are
# healthy). It reports the version and paper_count too, so a failing container is
# visible to `docker ps` / compose without hitting a rendered page.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/healthz || exit 1

CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--threads", "4", "--no-browser", "--expose"]
