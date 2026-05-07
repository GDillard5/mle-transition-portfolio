# ds_project

A minimal data-science project that demonstrates production Python practices:
strict typing with **mypy**, linting with **ruff**, and formatting with **black**.
The example task is a small end-to-end pipeline: load the Iris dataset, train a
logistic regression classifier, and report metrics.

## Project layout

```
ds_project/
├── pyproject.toml          # all tool config lives here
├── src/ds_project/         # importable package
│   ├── __init__.py
│   ├── config.py           # Pydantic settings — typed configuration
│   ├── data.py             # data loading + splitting
│   ├── features.py         # feature engineering
│   ├── model.py            # model training + evaluation
│   └── cli.py              # entry point: `ds-train`
├── tests/                  # pytest tests
└── data/                   # placeholder for raw / processed data
```

The `src/` layout is preferred for libraries because it forces you to install
the package before importing it — preventing accidental imports of files that
aren't actually shipped.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Daily workflow

| Tool   | Command                   | Purpose                                 |
| ------ | ------------------------- | --------------------------------------- |
| black  | `black src tests`         | Format code (auto-fixes whitespace)     |
| ruff   | `ruff check src tests`    | Lint code (catches bugs & style issues) |
| ruff   | `ruff check --fix .`      | Auto-fix simple issues                  |
| mypy   | `mypy`                    | Type-check (catches bugs before runtime)|
| pytest | `pytest`                  | Run tests with coverage                 |

Run them in this order: **black → ruff → mypy → pytest**. Black formats first
so ruff and mypy don't complain about whitespace, then ruff catches lint
issues, then mypy catches type errors, then pytest verifies behavior.

A one-liner for CI or pre-commit:

```bash
black --check src tests && ruff check src tests && mypy && pytest
```

## Run the pipeline

```bash
ds-train --test-size 0.2 --random-state 42
```

## What to study in this project

1. **`pyproject.toml`** — single source of truth for all tooling.
   Read every comment.
2. **`config.py`** — Pydantic gives you typed, validated config. Notice how
   mypy understands every field's type.
3. **`data.py`** — uses `dataclass` and `TypedDict` for structured returns.
4. **`model.py`** — shows `Protocol` for duck-typed interfaces and a
   `Generic` class.
5. **`cli.py`** — shows `argparse` with type hints and how to wire it all
   together.
