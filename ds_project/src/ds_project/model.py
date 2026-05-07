"""Model training and evaluation.

Demonstrates:
* `TypedDict` for a structured dictionary return type.
* A `dataclass` for grouped metrics.
* Composition of typed components (config + scaler + estimator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ds_project.config import TrainingConfig
from ds_project.data import DatasetSplit
from ds_project.features import fit_scaler


class MetricsDict(TypedDict):
    """A plain-dict view of the metrics, useful for JSON serialization."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Evaluation metrics for a fitted classifier."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def to_dict(self) -> MetricsDict:
        """Return a TypedDict — mypy validates the keys."""
        return MetricsDict(
            accuracy=self.accuracy,
            precision_macro=self.precision_macro,
            recall_macro=self.recall_macro,
            f1_macro=self.f1_macro,
        )


@dataclass(slots=True)
class IrisClassifier:
    """A simple logistic-regression classifier for the Iris dataset.

    Bundles the scaler and estimator so prediction always uses the same
    preprocessing as training. This is a tiny example of the pattern
    `sklearn.pipeline.Pipeline` formalizes.
    """

    config: TrainingConfig
    _scaler: StandardScaler | None = None
    _estimator: LogisticRegression | None = None

    def fit(self, split: DatasetSplit) -> IrisClassifier:
        """Fit scaler and estimator on the training split. Returns self."""
        self._scaler = fit_scaler(split.x_train)
        x_scaled = self._scaler.transform(split.x_train)

        self._estimator = LogisticRegression(
            C=self.config.regularization_c,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self._estimator.fit(x_scaled, split.y_train)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels for new samples."""
        if self._scaler is None or self._estimator is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")
        x_scaled = self._scaler.transform(x)
        predictions: NDArray[np.int64] = self._estimator.predict(x_scaled).astype(np.int64)
        return predictions

    def evaluate(self, split: DatasetSplit) -> ModelMetrics:
        """Compute classification metrics on the test split."""
        y_pred = self.predict(split.x_test)
        return ModelMetrics(
            accuracy=float(accuracy_score(split.y_test, y_pred)),
            precision_macro=float(
                precision_score(split.y_test, y_pred, average="macro", zero_division=0)
            ),
            recall_macro=float(
                recall_score(split.y_test, y_pred, average="macro", zero_division=0)
            ),
            f1_macro=float(f1_score(split.y_test, y_pred, average="macro", zero_division=0)),
        )
