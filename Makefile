# Convenience commands. Run `make help` to see them all.

.PHONY: help install format lint typecheck test check clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode with dev extras.
	pip install -e ".[dev]"

format:  ## Auto-format code with black and ruff --fix.
	black src tests
	ruff check --fix src tests

lint:  ## Lint with ruff (no auto-fix).
	ruff check src tests

typecheck:  ## Run mypy.
	mypy

test:  ## Run pytest with coverage.
	pytest

check: lint typecheck test  ## Run all checks (CI-style, no auto-fix).
	black --check src tests

clean:  ## Remove build artifacts and caches.
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
