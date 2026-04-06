"""Statistical utilities: confidence intervals and significance tests."""

import numpy as np


def compute_ci_bootstrap(y_true, y_pred, y_prob, metric_fn, n_bootstrap, ci_alpha, random_state):
    """Bootstrap confidence interval for any metric.

    Args:
        y_true: Ground truth labels.
        y_pred: Predictions.
        y_prob: Predicted probabilities (may be None).
        metric_fn: Callable(y_true, y_pred, y_prob) -> float.
        n_bootstrap: Number of resamples.
        ci_alpha: Significance level (0.05 = 95% CI).
        random_state: RNG seed.

    Returns:
        (ci_lower, ci_upper) tuple.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx] if y_prob is not None else None
        try:
            s = metric_fn(yt, yp, ypr)
            if s is not None and np.isfinite(s):
                scores.append(s)
        except Exception:
            continue
    if len(scores) == 0:
        return (np.nan, np.nan)
    lo = np.percentile(scores, 100 * ci_alpha / 2)
    hi = np.percentile(scores, 100 * (1 - ci_alpha / 2))
    return (lo, hi)


def compute_ci_wilson(proportion, n, alpha):
    """Wilson score interval for a binary proportion.

    Args:
        proportion: Observed proportion (e.g. accuracy).
        n: Sample size.
        alpha: Significance level.

    Returns:
        (ci_lower, ci_upper) tuple.
    """
    try:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
    except ImportError:
        # Fallback: z for alpha=0.05 -> 1.96
        z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96

    denom = 1 + z ** 2 / n
    centre = (proportion + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((proportion * (1 - proportion) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_p_value(y_true_slice, y_pred_slice, y_true_global, y_pred_global,
                    metric_fn, n_permutations, random_state):
    """Permutation test for metric difference between slice and global.

    Tests whether the slice metric is significantly different from the global metric.

    Args:
        y_true_slice: Slice ground truth.
        y_pred_slice: Slice predictions.
        y_true_global: Full test set ground truth.
        y_pred_global: Full test set predictions.
        metric_fn: Callable(y_true, y_pred, y_prob) -> float. y_prob passed as None.
        n_permutations: Number of permutations.
        random_state: RNG seed.

    Returns:
        p-value (float).
    """
    rng = np.random.RandomState(random_state)
    n_slice = len(y_true_slice)
    n_total = len(y_true_global)

    try:
        observed_slice = metric_fn(y_true_slice, y_pred_slice, None)
        observed_global = metric_fn(y_true_global, y_pred_global, None)
    except Exception:
        return 1.0

    observed_diff = observed_slice - observed_global

    # Pool all samples
    pooled_y = y_true_global
    pooled_pred = y_pred_global

    count = 0
    for _ in range(n_permutations):
        idx = rng.choice(n_total, size=n_slice, replace=False)
        perm_y = pooled_y[idx]
        perm_pred = pooled_pred[idx]
        try:
            perm_metric = metric_fn(perm_y, perm_pred, None)
            perm_diff = perm_metric - observed_global
            if perm_diff <= observed_diff:
                count += 1
        except Exception:
            continue

    return count / max(n_permutations, 1)
