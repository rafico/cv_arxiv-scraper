# Canonical developer/agent commands. These mirror what CI enforces
# (.github/workflows/ci.yml) so "green locally" means "green in CI".
#
# Usage: `make <target>`. Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help setup run run-prod test test-fast test-cov lint format types audit check ci tailwind clean

PYTHON ?= python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install the package with dev extras (editable)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

run: ## Run the dev server (Flask debug, opens browser)
	$(PYTHON) run.py --debug

run-prod: ## Run the production-style server (in-process gunicorn, 1 worker)
	$(PYTHON) run.py

test: ## Run the full test suite (excludes slow e2e browser tests)
	$(PYTHON) -m pytest -m "not e2e" -q

test-fast: ## Fast inner loop: skip slow + e2e tests
	$(PYTHON) -m pytest -m "not slow and not e2e" -q

test-all: ## Everything including Playwright e2e (needs: playwright install)
	$(PYTHON) -m pytest -q

test-cov: ## Tests with coverage report (as CI runs it)
	$(PYTHON) -m pytest --cov --cov-report=term-missing

lint: ## Lint + format check + hygiene hooks (pre-commit owns this in CI)
	$(PYTHON) -m pre_commit run --all-files --show-diff-on-failure

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

types: ## Type-check (advisory; matches CI's `mypy app`)
	$(PYTHON) -m mypy app

audit: ## Security audit of resolved deps (advisory; matches CI)
	$(PYTHON) -m pip_audit --skip-editable --strict

tailwind: ## Rebuild Tailwind CSS from the local CLI binary
	./tailwindcss -i app/static/src.css -o app/static/style.css --minify

check: lint types test ## Run the full local gate before pushing (lint + types + test)

ci: check audit ## Everything CI runs (minus e2e browser install)

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ */__pycache__ \
		.coverage coverage.xml build dist *.egg-info
