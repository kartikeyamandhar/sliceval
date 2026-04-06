"""Tests for metric computation, CI, and p-value."""

import numpy as np
import pytest

from sliceval.metrics import compute_ece, compute_slice_metrics, _get_metric_fn
from sliceval.utils.stats import compute_ci_bootstrap, compute_ci_wilson, compute_p_value


class TestECE:
    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert compute_ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-10)

    def test_worst_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ece = compute_ece(y_true, y_prob)
        assert ece >= 0.5

    def test_empty_bins_handled(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.01, 0.99])
        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert np.isfinite(ece)


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_pred = y_true.copy()
        fn = _get_metric_fn('accuracy', 'binary')
        lo, hi = compute_ci_bootstrap(y_true, y_pred, None, fn, 500, 0.05, 42)
        point = fn(y_true, y_pred, None)
        assert lo <= point <= hi

    def test_wider_ci_with_less_data(self):
        rng = np.random.RandomState(42)
        fn = _get_metric_fn('accuracy', 'binary')
        # Large sample
        y1 = rng.randint(0, 2, 500)
        lo1, hi1 = compute_ci_bootstrap(y1, y1, None, fn, 300, 0.05, 42)
        # Small sample
        y2 = rng.randint(0, 2, 30)
        lo2, hi2 = compute_ci_bootstrap(y2, y2, None, fn, 300, 0.05, 42)
        assert (hi2 - lo2) >= (hi1 - lo1)


class TestWilsonCI:
    def test_known_value(self):
        lo, hi = compute_ci_wilson(0.5, 100, 0.05)
        assert lo < 0.5 < hi
        assert lo > 0.35
        assert hi < 0.65

    def test_bounds_clamped(self):
        lo, hi = compute_ci_wilson(0.0, 10, 0.05)
        assert lo >= 0.0
        lo2, hi2 = compute_ci_wilson(1.0, 10, 0.05)
        assert hi2 <= 1.0


class TestPValue:
    def test_identical_slices_high_pvalue(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 200)
        fn = _get_metric_fn('accuracy', 'binary')
        p = compute_p_value(y, y, y, y, fn, 200, 42)
        # Slice identical to global -> p should be high
        assert p > 0.01

    def test_bad_slice_low_pvalue(self):
        rng = np.random.RandomState(42)
        # Global: 80% accuracy
        y_true_global = np.array([0, 1] * 100)
        y_pred_global = y_true_global.copy()
        # Slice: 0% accuracy (all wrong)
        y_true_slice = np.array([0] * 30)
        y_pred_slice = np.array([1] * 30)
        fn = _get_metric_fn('accuracy', 'binary')
        p = compute_p_value(y_true_slice, y_pred_slice,
                            y_true_global, y_pred_global, fn, 500, 42)
        assert p < 0.1


class TestSliceMetrics:
    def test_all_classification_metrics(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_pred = y_true.copy()
        y_prob = np.column_stack([1 - y_true.astype(float), y_true.astype(float)])
        metrics, ci_lo, ci_hi = compute_slice_metrics(
            y_true, y_pred, y_prob,
            ['f1', 'precision', 'recall', 'accuracy', 'auc', 'ece'],
            'binary', 'macro', 'bootstrap', 0.05, 100, 42,
        )
        for m in ['f1', 'precision', 'recall', 'accuracy', 'auc']:
            assert metrics[m] == pytest.approx(1.0, abs=0.01)
        assert metrics['ece'] == pytest.approx(0.0, abs=0.01)

    def test_regression_metrics(self):
        rng = np.random.RandomState(42)
        y_true = rng.normal(0, 1, 200)
        y_pred = y_true + rng.normal(0, 0.1, 200)
        metrics, ci_lo, ci_hi = compute_slice_metrics(
            y_true, y_pred, None,
            ['rmse', 'mae'], 'regression', 'macro',
            'bootstrap', 0.05, 100, 42,
        )
        assert metrics['rmse'] < 0.3
        assert metrics['mae'] < 0.3
        assert ci_lo['rmse'] < metrics['rmse']
        assert ci_hi['rmse'] > metrics['rmse']
