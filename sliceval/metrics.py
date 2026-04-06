"""Metric computation and confidence intervals for sliceval."""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
)

from .utils.stats import compute_ci_bootstrap, compute_ci_wilson


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error.

    Args:
        y_true: Binary labels (0/1).
        y_prob: Predicted probability of positive class (1D for binary,
                or max probability for multiclass).
        n_bins: Number of equal-width bins.

    Returns:
        ECE as float.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_confidence - bin_accuracy)
    return ece


def _get_metric_fn(metric_name, task, average='macro'):
    """Return a callable(y_true, y_pred, y_prob) -> float for the named metric."""

    avg = 'binary' if task == 'binary' else average

    def _f1(yt, yp, ypr):
        return f1_score(yt, yp, average=avg, zero_division=0)

    def _precision(yt, yp, ypr):
        return precision_score(yt, yp, average=avg, zero_division=0)

    def _recall(yt, yp, ypr):
        return recall_score(yt, yp, average=avg, zero_division=0)

    def _accuracy(yt, yp, ypr):
        return accuracy_score(yt, yp)

    def _auc(yt, yp, ypr):
        if ypr is None:
            return np.nan
        if ypr.ndim == 2:
            if ypr.shape[1] == 2:
                return roc_auc_score(yt, ypr[:, 1])
            else:
                return roc_auc_score(yt, ypr, multi_class='ovr', average='macro')
        return roc_auc_score(yt, ypr)

    def _ece(yt, yp, ypr):
        if ypr is None:
            return np.nan
        if ypr.ndim == 2:
            if ypr.shape[1] == 2:
                prob = ypr[:, 1]
            else:
                prob = ypr.max(axis=1)
        else:
            prob = ypr
        return compute_ece(yt, prob)

    def _rmse(yt, yp, ypr):
        return np.sqrt(mean_squared_error(yt, yp))

    def _mae(yt, yp, ypr):
        return mean_absolute_error(yt, yp)

    mapping = {
        'f1': _f1,
        'precision': _precision,
        'recall': _recall,
        'accuracy': _accuracy,
        'auc': _auc,
        'ece': _ece,
        'rmse': _rmse,
        'mae': _mae,
    }
    return mapping[metric_name]


# Metrics where Wilson CI is valid (binary proportions only)
_WILSON_ELIGIBLE = {'precision', 'recall', 'accuracy'}


def compute_slice_metrics(y_true, y_pred, y_prob, metric_names, task, average,
                          ci_method, ci_alpha, n_bootstrap, random_state):
    """Compute all requested metrics with CIs for a single slice.

    Returns:
        (metrics_dict, ci_lower_dict, ci_upper_dict)
    """
    metrics_dict = {}
    ci_lower_dict = {}
    ci_upper_dict = {}

    for name in metric_names:
        fn = _get_metric_fn(name, task, average)
        value = fn(y_true, y_pred, y_prob)
        metrics_dict[name] = value

        # CI
        use_wilson = (
            ci_method == 'wilson'
            and task == 'binary'
            and name in _WILSON_ELIGIBLE
        )

        if use_wilson and np.isfinite(value):
            lo, hi = compute_ci_wilson(value, len(y_true), ci_alpha)
        else:
            lo, hi = compute_ci_bootstrap(
                y_true, y_pred, y_prob, fn,
                n_bootstrap, ci_alpha, random_state,
            )

        ci_lower_dict[name] = lo
        ci_upper_dict[name] = hi

    return metrics_dict, ci_lower_dict, ci_upper_dict
