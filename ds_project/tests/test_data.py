"""Tests for ds_project.data."""

from __future__ import annotations

import numpy as np
import pytest

from ds_project.data import DatasetSplit, load_iris_split


def test_load_iris_split_shapes() -> None:
    split = load_iris_split(test_size=0.25, random_state=0)

    assert isinstance(split, DatasetSplit)
    assert split.x_train.shape[1] == 4
    assert split.x_test.shape[1] == 4
    assert split.x_train.shape[0] + split.x_test.shape[0] == 150
    assert split.n_features == 4
    assert split.n_classes == 3


def test_load_iris_split_dtypes() -> None:
    split = load_iris_split()
    assert split.x_train.dtype == np.float64
    assert split.y_train.dtype == np.int64


def test_load_iris_split_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="test_size"):
        load_iris_split(test_size=0.0)
    with pytest.raises(ValueError, match="test_size"):
        load_iris_split(test_size=1.0)


def test_dataset_split_is_frozen() -> None:
    split = load_iris_split()
    with pytest.raises((AttributeError, Exception)):
        split.x_train = np.zeros((1, 4))  # type: ignore[misc]
