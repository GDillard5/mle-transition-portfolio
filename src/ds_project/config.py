"""Typed configuration for the training pipeline.

We use Pydantic for runtime validation. mypy sees the field types and will
flag any caller that passes the wrong type. Pydantic also raises a clear
ValidationError at runtime if values are out of range — defense in depth.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    """Hyperparameters and run settings for training.

    Attributes:
        test_size: Fraction of the dataset reserved for evaluation. Must be
            in the open interval (0, 1).
        random_state: Seed for reproducibility. Any non-negative int.
        max_iter: Maximum optimizer iterations for logistic regression.
        regularization_c: Inverse of regularization strength; smaller means
            stronger regularization. Must be positive.
    """

    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0)
    max_iter: int = Field(default=200, gt=0)
    regularization_c: float = Field(default=1.0, gt=0.0)

    @field_validator("regularization_c")
    @classmethod
    def _warn_on_extreme_c(cls, v: float) -> float:
        """Demonstrate a custom validator with a typed signature."""
        if v > 1e4:
            # In real code, log a warning. Here we just illustrate the hook.
            pass
        return v

    model_config = {"frozen": True}  # immutable -> safer to pass around
