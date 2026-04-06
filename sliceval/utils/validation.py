"""Input validation for sliceval."""

import numpy as np
import pandas as pd

VALID_TASKS = ('binary', 'multiclass', 'regression')
CLASSIFICATION_METRICS = ('f1', 'precision', 'recall', 'accuracy', 'auc', 'ece')
REGRESSION_METRICS = ('rmse', 'mae')
PROBA_METRICS = ('ece', 'auc')
ALL_METRICS = CLASSIFICATION_METRICS + REGRESSION_METRICS


def validate_constructor(model, X, y, task, metrics):
    """Validate SliceEvaluator constructor inputs."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pd.DataFrame, got {type(X).__name__}")

    if isinstance(y, np.ndarray):
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise TypeError(f"y must be a pd.Series or 1D np.ndarray, got {type(y).__name__}")

    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")

    if task not in VALID_TASKS:
        raise ValueError(f"task must be 'binary', 'multiclass', or 'regression'. Got: '{task}'")

    for m in metrics:
        if m not in ALL_METRICS:
            raise ValueError(f"Unknown metric '{m}'. Valid metrics: {list(ALL_METRICS)}")
        if task == 'regression' and m in CLASSIFICATION_METRICS:
            raise ValueError(
                f"Metric '{m}' is not valid for task='regression'. "
                f"Valid regression metrics: {list(REGRESSION_METRICS)}"
            )
        if task != 'regression' and m in REGRESSION_METRICS:
            raise ValueError(
                f"Metric '{m}' is only valid for task='regression'. Got task='{task}'"
            )

    return y


def validate_proba_metrics(metrics, model):
    """Check that model has predict_proba if proba metrics are requested."""
    for m in metrics:
        if m in PROBA_METRICS and not hasattr(model, 'predict_proba'):
            raise ValueError(
                f"Metric '{m}' requires model.predict_proba(), "
                f"which is not available on this model."
            )


def validate_slice_mask(name, mask, expected_length):
    """Validate a resolved boolean mask."""
    if len(mask) != expected_length:
        raise ValueError(
            f"Slice '{name}' mask has length {len(mask)}, expected {expected_length}"
        )
    if mask.sum() == 0:
        raise ValueError(f"Slice '{name}' has 0 samples. Check your mask condition.")
