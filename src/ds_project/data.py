"""Data loading utilities.

Demonstrates:
* `dataclass` for a structured return type that mypy understands deeply.
* `numpy.typing.NDArray` for typed array shapes.
* Re-export of a small, stable surface from `__init__.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A train/test split of a tabular dataset.

    Using `frozen=True` makes instances hashable and prevents accidental
    mutation. `slots=True` saves memory and prevents typos creating new
    attributes — a common bug with plain classes.
    """

    x_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return int(self.x_train.shape[1])

    @property
    def n_classes(self) -> int:
        """Number of distinct target classes."""
        return len(self.target_names)


def load_iris_split(
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Load the Iris dataset and split it into train/test partitions.

    Args:
        test_size: Fraction of samples reserved for evaluation.
        random_state: Seed for the shuffle, ensuring reproducibility.

    Returns:
        A `DatasetSplit` containing typed numpy arrays and metadata.

    Raises:
        ValueError: If `test_size` is not in the open interval (0, 1).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")

    bunch = load_iris()
    x: NDArray[np.float64] = bunch.data.astype(np.float64)
    y: NDArray[np.int64] = bunch.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=tuple(bunch.feature_names),
        target_names=tuple(bunch.target_names),
    )
