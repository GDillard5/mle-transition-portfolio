"""Command-line entry point: `ds-train`.

Parses arguments, builds a `TrainingConfig`, runs the pipeline, and prints
the metrics. Demonstrates a typed `main` function and how to validate input
through Pydantic before doing real work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from ds_project.config import TrainingConfig
from ds_project.data import load_iris_split
from ds_project.model import IrisClassifier


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Isolated for easy testing."""
    parser = argparse.ArgumentParser(
        prog="ds-train",
        description="Train an Iris classifier and print evaluation metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (0, 1).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=200, help="Optimizer iterations.")
    parser.add_argument(
        "--c",
        dest="regularization_c",
        type=float,
        default=1.0,
        help="Inverse regularization strength.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training pipeline.

    Args:
        argv: Optional argument list (for testing). Defaults to ``sys.argv``.

    Returns:
        Process exit code: 0 on success, non-zero on failure.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Pydantic validates ranges and types here. A bad CLI value raises a
    # clean ValidationError instead of crashing deep inside sklearn.
    config = TrainingConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
        regularization_c=args.regularization_c,
    )

    split = load_iris_split(test_size=config.test_size, random_state=config.random_state)
    classifier = IrisClassifier(config=config).fit(split)
    metrics = classifier.evaluate(split)

    print(json.dumps(metrics.to_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
