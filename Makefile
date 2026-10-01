.PHONY: check lint fmt test

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
