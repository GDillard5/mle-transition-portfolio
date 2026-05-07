"""Tests for ds_project.cli."""

from __future__ import annotations

import json

import pytest

from ds_project.cli import main


def test_cli_runs_and_prints_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--test-size", "0.2", "--random-state", "42"])
    assert exit_code == 0

    captured = capsys.readouterr()
    metrics = json.loads(captured.out)
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.85


def test_cli_rejects_invalid_test_size() -> None:
    with pytest.raises(ValueError):
        main(["--test-size", "1.5"])
