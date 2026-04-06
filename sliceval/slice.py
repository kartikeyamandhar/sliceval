"""Data structures for sliceval."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Slice:
    """A subgroup of the test set defined by feature conditions."""
    name: str
    mask: np.ndarray              # Boolean, shape (n_test_samples,)
    n_samples: int                # Number of True entries in mask
    support: float                # n_samples / len(X_test)
    source: str                   # 'manual' | 'tree' | 'beam'
    feature_conditions: list = field(default_factory=list)
    # e.g. ['sensor_type == B', 'hour < 6']
    # Empty list for manual slices


@dataclass
class SliceMetrics:
    """Per-slice metric results."""
    slice_name: str
    n_samples: int
    support: float
    metrics: dict       # {'f1': 0.41, ...}
    ci_lower: dict      # {'f1': 0.33, ...}
    ci_upper: dict      # {'f1': 0.49, ...}
    delta: dict         # {'f1': -0.50, ...} slice metric minus global metric
    p_value: dict       # {'f1': 0.003, ...} significance vs global
