"""ds_project: a small example data-science package.

This package demonstrates production-quality typing and tooling for a
data-science codebase. The public API is intentionally small.
"""

from ds_project.config import TrainingConfig
from ds_project.data import DatasetSplit, load_iris_split
from ds_project.model import IrisClassifier, ModelMetrics

__all__ = [
    "DatasetSplit",
    "IrisClassifier",
    "ModelMetrics",
    "TrainingConfig",
    "load_iris_split",
]

__version__ = "0.1.0"
