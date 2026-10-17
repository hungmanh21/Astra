.PHONY: check lint fmt test run

# The venv lives on the Linux filesystem: /mnt/d (WSL2) is far too slow for thousands of small files.
# Override by exporting UV_PROJECT_ENVIRONMENT before calling make.
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/astra

# Everything that must pass before committing (see CLAUDE.md, session workflow).
check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

run:
	uv run uvicorn backend.main:create_app --factory --port 8000
