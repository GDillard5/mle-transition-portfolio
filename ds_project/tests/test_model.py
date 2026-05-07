"""Tests for ds_project.model."""

from __future__ import annotations

import pytest

from ds_project.config import TrainingConfig
from ds_project.data import load_iris_split
from ds_project.model import IrisClassifier, ModelMetrics


def test_training_config_validates_ranges() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(test_size=1.5)
    with pytest.raises(ValueError):
        TrainingConfig(regularization_c=-1.0)


def test_classifier_fit_and_evaluate() -> None:
    split = load_iris_split(test_size=0.2, random_state=42)
    config = TrainingConfig(random_state=42)
    classifier = IrisClassifier(config=config).fit(split)

    metrics = classifier.evaluate(split)
    assert isinstance(metrics, ModelMetrics)
    # Iris is easy; logistic regression should clear 0.85 comfortably.
    assert metrics.accuracy >= 0.85
    assert 0.0 <= metrics.f1_macro <= 1.0


def test_predict_before_fit_raises() -> None:
    split = load_iris_split()
    classifier = IrisClassifier(config=TrainingConfig())
    with pytest.raises(RuntimeError, match="not fitted"):
        classifier.predict(split.x_test)


def test_metrics_to_dict_keys() -> None:
    split = load_iris_split()
    metrics = IrisClassifier(config=TrainingConfig()).fit(split).evaluate(split)
    d = metrics.to_dict()
    assert set(d.keys()) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
