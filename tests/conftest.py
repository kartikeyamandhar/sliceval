"""Shared test fixtures for sliceval."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_data():
    """Imbalanced binary classification dataset. 900 majority, 100 minority."""
    np.random.seed(42)
    X = pd.DataFrame({
        'sensor_type': np.random.choice(['A', 'B'], size=1000, p=[0.9, 0.1]),
        'hour': np.random.randint(0, 24, size=1000),
        'temperature': np.random.normal(60, 15, size=1000),
    })
    y = pd.Series((X['sensor_type'] == 'A').astype(int))
    return X, y


@pytest.fixture
def perfect_model():
    """Model that predicts y_true exactly. Must be fit before use."""
    class _M:
        def __init__(self):
            self._y = None

        def fit(self, X, y):
            self._y = np.asarray(y).copy()
            return self

        def predict(self, X):
            return self._y

        def predict_proba(self, X):
            p = self._y.astype(float)
            return np.column_stack([1 - p, p])
    return _M()


@pytest.fixture
def biased_model():
    """Model that always predicts 1 (majority class)."""
    class _M:
        def __init__(self):
            self._n = None

        def fit(self, X, y):
            self._n = len(y)
            return self

        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
    return _M()


@pytest.fixture
def regression_data():
    """Simple regression dataset."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature_a': np.random.normal(0, 1, 500),
        'feature_b': np.random.choice(['X', 'Y'], 500),
    })
    y = pd.Series(X['feature_a'] * 2 + np.random.normal(0, 0.5, 500))
    return X, y


@pytest.fixture
def regression_model():
    """Model that returns stored predictions."""
    class _M:
        def __init__(self):
            self._preds = None

        def fit(self, X, y):
            self._preds = np.asarray(y).copy() + np.random.normal(0, 0.1, len(y))
            return self

        def predict(self, X):
            return self._preds
    return _M()
