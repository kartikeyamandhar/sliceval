"""Tree-based slice discovery.

Fits a shallow decision tree that predicts model errors.
Leaf nodes become candidate slices — regions of feature space
where the model systematically fails.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..slice import Slice
from ..metrics import _get_metric_fn
from ..utils.stats import compute_p_value


def discover_tree(X, y_true, y_pred, max_depth, min_support, n_slices,
                  metric, significance, task, random_state):
    """Discover problematic slices using a decision tree on model errors.

    Args:
        X: Test features (pd.DataFrame).
        y_true: Ground truth (np.ndarray).
        y_pred: Model predictions (np.ndarray).
        max_depth: Max tree depth (controls conjunction complexity).
        min_support: Min fraction of samples a slice must cover.
        n_slices: Max number of slices to return.
        metric: Metric name used to rank slices.
        significance: p-value threshold for filtering.
        task: 'binary' | 'multiclass' | 'regression'.
        random_state: RNG seed.

    Returns:
        List of Slice objects.
    """
    min_samples_leaf = max(1, int(min_support * len(X)))

    # Step 1: Compute per-sample error signal
    if task == 'regression':
        error = np.abs(y_true - y_pred)
    else:
        error = (y_true != y_pred).astype(int)

    # Step 2: Encode categoricals for the tree
    X_encoded, col_mapping = _encode_for_tree(X)

    # Step 3: Fit decision tree on error signal
    if task == 'regression':
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    else:
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    tree.fit(X_encoded, error)

    # Step 4: Extract leaf assignments
    leaf_ids = tree.apply(X_encoded)

    # Step 5: Build Slice objects from leaves
    encoded_columns = list(X_encoded.columns)
    slices = []
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        support = mask.sum() / len(X)
        if support < min_support:
            continue
        conditions = _extract_conditions(tree, leaf_id, encoded_columns, col_mapping)
        slices.append(Slice(
            name=_conditions_to_name(conditions) + " (auto)",
            mask=mask,
            n_samples=int(mask.sum()),
            support=support,
            source='tree',
            feature_conditions=conditions,
        ))

    # Step 6: Rank by metric drop, filter by significance
    return _rank_and_filter(
        slices, y_true, y_pred, metric, significance,
        n_slices, task, random_state,
    )


def _encode_for_tree(X):
    """Encode categorical columns for sklearn tree.

    Returns:
        (X_encoded, col_mapping) where col_mapping maps encoded column names
        back to (original_col, value) for dummy columns, or (original_col, None)
        for numeric columns.
    """
    col_mapping = {}
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in numeric_cols:
        col_mapping[c] = (c, None)

    encoded = X[numeric_cols].copy()

    for c in cat_cols:
        dummies = pd.get_dummies(X[c], prefix=c, dtype=float)
        for dummy_col in dummies.columns:
            # dummy_col looks like "sensor_type_B"
            val = dummy_col[len(c) + 1:]  # strip prefix + underscore
            col_mapping[dummy_col] = (c, val)
        encoded = pd.concat([encoded, dummies], axis=1)

    return encoded, col_mapping


def _extract_conditions(tree, target_leaf, columns, col_mapping):
    """Walk root-to-leaf path and collect human-readable conditions."""
    tree_ = tree.tree_
    node = 0  # start at root
    path = _find_path_to_leaf(tree_, node, target_leaf)
    if path is None:
        return []

    conditions = []
    for i in range(len(path) - 1):
        node = path[i]
        next_node = path[i + 1]
        feature_idx = tree_.feature[node]
        threshold = tree_.threshold[node]
        encoded_col = columns[feature_idx]
        original_col, cat_val = col_mapping[encoded_col]

        went_left = (next_node == tree_.children_left[node])

        if cat_val is not None:
            # Dummy column: threshold is 0.5 for binary dummies
            if went_left:
                conditions.append(f"{original_col} != {cat_val}")
            else:
                conditions.append(f"{original_col} == {cat_val}")
        else:
            if went_left:
                conditions.append(f"{original_col} <= {threshold:.4g}")
            else:
                conditions.append(f"{original_col} > {threshold:.4g}")

    return conditions


def _find_path_to_leaf(tree_, node, target_leaf):
    """DFS to find the path from node to target_leaf. Returns list of node indices."""
    if node == target_leaf:
        return [node]

    left = tree_.children_left[node]
    right = tree_.children_right[node]

    # Leaf node that isn't target
    if left == right:
        return None

    left_path = _find_path_to_leaf(tree_, left, target_leaf)
    if left_path is not None:
        return [node] + left_path

    right_path = _find_path_to_leaf(tree_, right, target_leaf)
    if right_path is not None:
        return [node] + right_path

    return None


def _conditions_to_name(conditions):
    """Join conditions into a readable slice name."""
    if not conditions:
        return "unknown"
    return " AND ".join(conditions)


def _rank_and_filter(slices, y_true, y_pred, metric, significance,
                     n_slices, task, random_state):
    """Rank slices by metric drop, filter by significance, return top n."""
    if not slices:
        return []

    metric_fn = _get_metric_fn(metric, task)

    try:
        global_metric = metric_fn(y_true, y_pred, None)
    except Exception:
        return slices[:n_slices]

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

    # Sort by delta ascending (most negative = worst performing)
    scored.sort(key=lambda x: x[0])
    return [sl for _, _, sl in scored[:n_slices]]