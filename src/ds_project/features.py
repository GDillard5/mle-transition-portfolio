"""Feature engineering helpers.

Demonstrates `typing.Protocol` for structural ("duck") typing — a common
pattern when wrapping scikit-learn transformers that share an interface
without inheriting from a common base.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


@runtime_checkable
class Transformer(Protocol):
    """Anything with `.fit` and `.transform` methods on numpy arrays.

    Marking this Protocol `runtime_checkable` lets you use `isinstance(obj,
    Transformer)` at runtime — useful in tests. mypy uses it statically
    regardless.
    """

    def fit(self, x: NDArray[np.float64]) -> Transformer:
        """Fit the transformer to data. Returns self for chaining."""
        ...

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the learned transformation."""
        ...


def fit_scaler(x_train: NDArray[np.float64]) -> StandardScaler:
    """Fit a standard scaler on training features.

    Args:
        x_train: Training feature matrix of shape (n_samples, n_features).

    Returns:
        A fitted `StandardScaler` ready to transform new data.
    """
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler
