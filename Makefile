# Developer commands for macOS and Linux.
# Everything runs inside the local .venv so no global packages are touched.

PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python

.DEFAULT_GOAL := help
.PHONY: help setup run test lint format check clean

help: ## Show the available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt
	$(VENV_PYTHON) -m pip install -e .
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Run 'make run' to start the app."

run: ## Start the Streamlit application
	$(VENV_PYTHON) -m streamlit run app.py

test: ## Run the unit tests
	$(VENV_PYTHON) -m pytest

lint: ## Check code style and formatting
	$(VENV_PYTHON) -m ruff check .
	$(VENV_PYTHON) -m ruff format --check .

format: ## Reformat the code
	$(VENV_PYTHON) -m ruff format .
	$(VENV_PYTHON) -m ruff check . --fix

check: lint test ## Run linting and tests together

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
