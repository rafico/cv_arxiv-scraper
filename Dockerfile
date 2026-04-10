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

COPY requirements.txt ./

RUN "${VIRTUAL_ENV}/bin/pip" install -r requirements.txt

FROM python:3.12-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CV_ARXIV_CONFIG=/app/config.yaml \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/appuser --shell /bin/bash appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser config.example.yaml digest_cli.py export_cli.py gmail_auth_setup.py run.py scrape_cli.py wsgi.py ./
COPY --chown=appuser:appuser requirements.txt ./

RUN install -d -o appuser -g appuser /app/instance

USER appuser

VOLUME ["/app/instance"]

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "--worker-class", "gthread", "wsgi:app"]
