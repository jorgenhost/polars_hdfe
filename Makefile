SHELL=/bin/bash

.PHONY: setup install install-release pre-commit test run run-release

# Replaces 'venv' - creates env and installs dependencies from pyproject.toml
setup:
	uv sync

install:
	uv run maturin develop

install-release:
	uv run maturin develop --release

pre-commit:
	cargo +nightly fmt --all && cargo clippy --all-features
	uv run ruff check . --fix --exit-non-zero-on-fix
	uv run ruff format polars_hdfe tests
	uv run mypy polars_hdfe tests
test:
	uv run pytest tests

run: install
	uv run python run.py

run-release: install-release
	uv run python run.py

