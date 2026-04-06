"""SliceEvaluator — main entry point for sliceval."""

import warnings

import numpy as np
import pandas as pd

from .slice import Slice, SliceMetrics
from .metrics import compute_slice_metrics, _get_metric_fn
from .report import SliceReport
from .utils.validation import (
    validate_constructor, validate_proba_metrics, validate_slice_mask,
)
from .utils.stats import compute_p_value


class SliceEvaluator:
    """Evaluate an ML model across data subgroups (slices).

    Args:
        model: Any object with .predict(). .predict_proba() used if available.
        X: Test features as pd.DataFrame.
        y: Ground truth labels as pd.Series or 1D np.ndarray.
        task: 'binary' | 'multiclass' | 'regression'.
        metrics: List of metric names. Default: ['f1', 'precision', 'recall'].
        ci_method: 'bootstrap' | 'wilson'.
        ci_alpha: Confidence level = 1 - ci_alpha.
        n_bootstrap: Bootstrap iterations.
        average: Averaging strategy for multiclass ('macro'|'micro'|'weighted').
        random_state: RNG seed.
    """

    def __init__(self, model, X, y, task='binary', metrics=None,
                 ci_method='bootstrap', ci_alpha=0.05, n_bootstrap=1000,
                 average='macro', random_state=42):

        if metrics is None:
            if task == 'regression':
                metrics = ['rmse', 'mae']
            else:
                metrics = ['f1', 'precision', 'recall']

        y = validate_constructor(model, X, y, task, metrics)

        self.model = model
        self.X = X
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.task = task
        self.metric_names = list(metrics)
        self.ci_method = ci_method
        self.ci_alpha = ci_alpha
        self.n_bootstrap = n_bootstrap
        self.average = average
        self.random_state = random_state

        self._manual_slices = []        # list of (name, mask_or_callable)
        self._discovered_slices = []    # list of Slice objects
        self._y_pred = None
        self._y_prob = None

    def add_slice(self, name, mask):
        """Add a manually defined slice.

        Args:
            name: Human-readable label.
            mask: Boolean pd.Series/np.ndarray aligned with X rows,
                  or callable f(X) -> boolean Series/ndarray.
        """
        # Check for duplicate
        existing_names = [n for n, _ in self._manual_slices]
        if name in existing_names:
            warnings.warn(f"Slice '{name}' already exists and will be overwritten.",
                          UserWarning)
            self._manual_slices = [(n, m) for n, m in self._manual_slices if n != name]

        self._manual_slices.append((name, mask))

    def discover_slices(self, method='tree', max_depth=3, min_support=0.05,
                        metric='f1', n_slices=10, significance=0.05, **kwargs):
        """Auto-discover problematic slices.

        Args:
            method: 'tree' or 'beam'.
            max_depth: Max feature conjunctions.
            min_support: Min fraction of test set a slice must cover.
            metric: Metric used to rank slices.
            n_slices: Max slices to return.
            significance: p-value threshold.
        """
        if metric not in self.metric_names:
            raise ValueError(
                f"Discovery metric '{metric}' is not in the evaluator's "
                f"metric list: {self.metric_names}"
            )

        # Need predictions for discovery
        self._run_inference()

        if method == 'tree':
            from .discovery.tree import discover_tree
            slices = discover_tree(
                self.X, self.y.values, self._y_pred,
                max_depth=max_depth, min_support=min_support,
                n_slices=n_slices, metric=metric,
                significance=significance, task=self.task,
                random_state=self.random_state,
            )
        elif method == 'beam':
            from .discovery.beam import discover_beam
            slices = discover_beam(
                self.X, self.y.values, self._y_pred,
                max_depth=max_depth, min_support=min_support,
                n_slices=n_slices, metric=metric,
                significance=significance, task=self.task,
                random_state=self.random_state, **kwargs,
            )
        else:
            raise ValueError(f"method must be 'tree' or 'beam'. Got: '{method}'")

        self._discovered_slices.extend(slices)

    def evaluate(self):
        """Run evaluation across all slices.

        Returns:
            SliceReport
        """
        self._run_inference()

        # Check proba metrics
        proba_needed = any(m in ('ece', 'auc') for m in self.metric_names)
        if proba_needed:
            validate_proba_metrics(self.metric_names, self.model)

        y_true = self.y.values
        y_pred = self._y_pred
        y_prob = self._y_prob

        # Global metrics
        global_metrics, _, _ = compute_slice_metrics(
            y_true, y_pred, y_prob,
            self.metric_names, self.task, self.average,
            self.ci_method, self.ci_alpha, self.n_bootstrap, self.random_state,
        )

        # Resolve all slices
        all_slices = []

        # Manual slices
        for name, mask_or_callable in self._manual_slices:
            resolved = self._resolve_mask(mask_or_callable)
            validate_slice_mask(name, resolved, len(self.X))
            n_samples = int(resolved.sum())
            if n_samples < 30:
                warnings.warn(
                    f"Slice '{name}' has {n_samples} samples. "
                    f"Metrics may be unreliable. Consider increasing min_support.",
                    UserWarning,
                )
            all_slices.append(Slice(
                name=name,
                mask=resolved,
                n_samples=n_samples,
                support=n_samples / len(self.X),
                source='manual',
                feature_conditions=[],
            ))

        # Discovered slices
        all_slices.extend(self._discovered_slices)

        if len(all_slices) == 0:
            warnings.warn(
                "No slices defined. Call add_slice() or discover_slices() "
                "before evaluate().",
                UserWarning,
            )

        # Compute per-slice metrics
        slice_metrics_list = []
        for sl in all_slices:
            mask = sl.mask
            yt = y_true[mask]
            yp = y_pred[mask]
            ypr = y_prob[mask] if y_prob is not None else None

            m_vals, ci_lo, ci_hi = compute_slice_metrics(
                yt, yp, ypr,
                self.metric_names, self.task, self.average,
                self.ci_method, self.ci_alpha, self.n_bootstrap,
                self.random_state,
            )

            # Delta
            delta = {k: m_vals[k] - global_metrics[k] for k in m_vals}

            # P-value
            p_vals = {}
            for mn in self.metric_names:
                fn = _get_metric_fn(mn, self.task, self.average)
                p_vals[mn] = compute_p_value(
                    yt, yp, y_true, y_pred,
                    fn, self.n_bootstrap, self.random_state,
                )

            slice_metrics_list.append(SliceMetrics(
                slice_name=sl.name,
                n_samples=sl.n_samples,
                support=sl.support,
                metrics=m_vals,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                delta=delta,
                p_value=p_vals,
            ))

        return SliceReport(
            global_metrics=global_metrics,
            slices=all_slices,
            metrics=slice_metrics_list,
            X=self.X,
            y=self.y,
            task=self.task,
        )

    def _resolve_mask(self, mask_or_callable):
        """Convert mask input to np.ndarray[bool]."""
        if callable(mask_or_callable):
            result = mask_or_callable(self.X)
        else:
            result = mask_or_callable
        if isinstance(result, pd.Series):
            result = result.values
        return np.asarray(result, dtype=bool)

    def _run_inference(self):
        """Run model.predict and model.predict_proba once, cache results."""
        if self._y_pred is not None:
            return
        self._y_pred = self.model.predict(self.X)
        if hasattr(self.model, 'predict_proba'):
            self._y_prob = self.model.predict_proba(self.X)
