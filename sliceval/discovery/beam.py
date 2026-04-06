"""Beam search slice discovery (SliceFinder algorithm).

Enumerates feature conjunctions breadth-first. At each depth,
expands the best-scoring candidates. More exhaustive than tree-based
discovery but slower on wide feature spaces.
"""

import numpy as np
import pandas as pd

from ..slice import Slice
from ..metrics import _get_metric_fn
from ..utils.stats import compute_p_value


def discover_beam(X, y_true, y_pred, max_depth, min_support, n_slices,
                  metric, significance, task, beam_width=10, random_state=42):
    """Discover problematic slices via beam search over feature predicates.

    Args:
        X: Test features (pd.DataFrame).
        y_true: Ground truth (np.ndarray).
        y_pred: Model predictions (np.ndarray).
        max_depth: Max predicate conjunctions per slice.
        min_support: Min fraction of samples a slice must cover.
        n_slices: Max slices to return.
        metric: Metric name used to rank slices.
        significance: p-value threshold.
        task: 'binary' | 'multiclass' | 'regression'.
        beam_width: Candidates kept at each depth.
        random_state: RNG seed.

    Returns:
        List of Slice objects.
    """
    metric_fn = _get_metric_fn(metric, task)
    try:
        global_metric = metric_fn(y_true, y_pred, None)
    except Exception:
        return []

    n = len(X)

    # Step 1: Discretize continuous features, generate base predicates
    base_predicates = _generate_base_predicates(X, n_bins=4)

    # Step 2: Beam search
    # Each candidate is (predicate_list, mask)
    # predicate_list: list of (description_str, column_set)
    candidates = []
    for desc, mask, cols in base_predicates:
        if mask.sum() / n >= min_support:
            candidates.append(([desc], mask, cols))

    all_candidates = list(candidates)

    for depth in range(1, max_depth):
        scored = []
        for descs, mask, cols in candidates:
            if mask.sum() / n < min_support:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            try:
                val = metric_fn(yt, yp, None)
            except Exception:
                continue
            scored.append((val, descs, mask, cols))

        # Sort: lowest metric value = worst slice = most interesting
        scored.sort(key=lambda x: x[0])
        top = scored[:beam_width]

        if not top:
            break

        # Expand: combine top candidates with base predicates
        new_candidates = []
        for _, descs, mask, cols in top:
            for base_desc, base_mask, base_cols in base_predicates:
                # Don't combine predicates on the same column
                if cols & base_cols:
                    continue
                combined_mask = mask & base_mask
                if combined_mask.sum() / n < min_support:
                    continue
                new_descs = descs + [base_desc]
                new_cols = cols | base_cols
                new_candidates.append((new_descs, combined_mask, new_cols))

        all_candidates.extend(new_candidates)
        candidates = new_candidates

    # Step 3: Deduplicate by mask
    seen = set()
    unique = []
    for descs, mask, cols in all_candidates:
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        if mask.sum() / n < min_support:
            continue
        unique.append((descs, mask))

    # Step 4: Build Slice objects, rank, filter significance
    slices = []
    for descs, mask in unique:
        slices.append(Slice(
            name=" AND ".join(descs) + " (auto)",
            mask=mask,
            n_samples=int(mask.sum()),
            support=mask.sum() / n,
            source='beam',
            feature_conditions=list(descs),
        ))

    return _rank_and_filter(
        slices, y_true, y_pred, metric_fn, global_metric,
        significance, n_slices, random_state,
    )


def _generate_base_predicates(X, n_bins=4):
    """Generate single-feature predicates from the DataFrame.

    Returns:
        List of (description_str, boolean_mask, column_set) tuples.
    """
    predicates = []
    n = len(X)

    for col in X.columns:
        is_cat = (
            X[col].dtype == object
            or X[col].dtype.name == 'category'
            or pd.api.types.is_string_dtype(X[col])
        )
        if is_cat:
            # Categorical: one predicate per unique value
            for val in X[col].unique():
                mask = (X[col] == val).values
                predicates.append((f"{col} == {val}", mask, {col}))
        else:
            # Numeric: bin into quantile-based bins
            try:
                bins = pd.qcut(X[col], q=n_bins, duplicates='drop')
                for interval in bins.cat.categories:
                    mask = (X[col] >= interval.left) & (X[col] <= interval.right)
                    mask = mask.values
                    desc = f"{col} in [{interval.left:.4g}, {interval.right:.4g}]"
                    predicates.append((desc, mask, {col}))
            except (ValueError, TypeError):
                # Fallback: binary split at median
                med = X[col].median()
                mask_lo = (X[col] <= med).values
                mask_hi = (X[col] > med).values
                predicates.append((f"{col} <= {med:.4g}", mask_lo, {col}))
                predicates.append((f"{col} > {med:.4g}", mask_hi, {col}))

    return predicates


def _rank_and_filter(slices, y_true, y_pred, metric_fn, global_metric,
                     significance, n_slices, random_state):
    """Rank slices by metric drop, filter by significance, return top n."""
    if not slices:
        return []

    scored = []
    for sl in slices:
        yt = y_true[sl.mask]
        yp = y_pred[sl.mask]
        try:
            slice_metric = metric_fn(yt, yp, None)
        except Exception:
            continue

        delta = slice_metric - global_metric

        p = compute_p_value(
            yt, yp, y_true, y_pred,
            metric_fn, min(500, max(100, len(y_true))),
            random_state,
        )

        if p <= significance:
            scored.append((delta, p, sl))

    scored.sort(key=lambda x: x[0])
    return [sl for _, _, sl in scored[:n_slices]]