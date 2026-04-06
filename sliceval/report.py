"""SliceReport — output container for slice evaluation results."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .slice import Slice, SliceMetrics


class SliceReport:
    """Container for slice evaluation results."""

    def __init__(self, global_metrics, slices, metrics, X, y, task):
        self.global_metrics = global_metrics    # dict
        self.slices = slices                    # list[Slice]
        self.metrics = metrics                  # list[SliceMetrics]
        self.X = X                              # pd.DataFrame ref
        self.y = y                              # pd.Series ref
        self.task = task
        self.evaluated_at = datetime.now(timezone.utc)

    def worst_slices(self, n=5, metric=None, min_support=0.0):
        """Return DataFrame of worst-performing slices sorted by delta ascending.

        Args:
            n: Number of slices to return.
            metric: Metric to sort by. Default: first metric.
            min_support: Minimum support filter.

        Returns:
            pd.DataFrame with columns: slice_name, n_samples, support,
            {metric}, ci_lower, ci_upper, delta, p_value, source
        """
        if metric is None:
            metric = list(self.metrics[0].metrics.keys())[0]

        rows = []
        for sm, sl in zip(self.metrics, self.slices):
            if sm.support < min_support:
                continue
            rows.append({
                'slice_name': sm.slice_name,
                'n_samples': sm.n_samples,
                'support': sm.support,
                metric: sm.metrics.get(metric, np.nan),
                'ci_lower': sm.ci_lower.get(metric, np.nan),
                'ci_upper': sm.ci_upper.get(metric, np.nan),
                'delta': sm.delta.get(metric, np.nan),
                'p_value': sm.p_value.get(metric, np.nan),
                'source': sl.source,
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return df
        df = df.sort_values('delta', ascending=True).head(n).reset_index(drop=True)
        return df

    def to_dataframe(self):
        """Return full slice × metric DataFrame.

        First row is always [global]. One row per slice after that.
        Columns per metric: {metric}_value, {metric}_ci_lower, {metric}_ci_upper,
                            {metric}_delta, {metric}_p_value
        """
        metric_names = list(self.global_metrics.keys())

        rows = []
        # Global row
        global_row = {
            'slice_name': '[global]',
            'source': 'global',
            'n_samples': len(self.y),
            'support': 1.0,
            'feature_conditions': '[]',
        }
        for m in metric_names:
            global_row[f'{m}_value'] = self.global_metrics[m]
            global_row[f'{m}_ci_lower'] = np.nan
            global_row[f'{m}_ci_upper'] = np.nan
            global_row[f'{m}_delta'] = 0.0
            global_row[f'{m}_p_value'] = np.nan
        rows.append(global_row)

        # Slice rows
        for sm, sl in zip(self.metrics, self.slices):
            row = {
                'slice_name': sm.slice_name,
                'source': sl.source,
                'n_samples': sm.n_samples,
                'support': sm.support,
                'feature_conditions': str(sl.feature_conditions),
            }
            for m in metric_names:
                row[f'{m}_value'] = sm.metrics.get(m, np.nan)
                row[f'{m}_ci_lower'] = sm.ci_lower.get(m, np.nan)
                row[f'{m}_ci_upper'] = sm.ci_upper.get(m, np.nan)
                row[f'{m}_delta'] = sm.delta.get(m, np.nan)
                row[f'{m}_p_value'] = sm.p_value.get(m, np.nan)
            rows.append(row)

        return pd.DataFrame(rows)

    def to_mlflow(self, run_id=None, artifact_path='slice_eval'):
        """Log report as MLflow artifacts."""
        from .integrations.mlflow import log_slice_report
        log_slice_report(self, run_id=run_id, artifact_path=artifact_path)

    def plot(self, metric=None, top_n=10, figsize=(10, 6)):
        """Horizontal bar chart of worst slices.

        Returns:
            matplotlib.figure.Figure
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires: pip install sliceval[plot]")

        if metric is None:
            metric = list(self.global_metrics.keys())[0]

        df = self.worst_slices(n=top_n, metric=metric)
        if len(df) == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No slices to plot', ha='center', va='center')
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        colors = []
        for d in df['delta']:
            if d < -0.1:
                colors.append('#d9534f')  # red
            elif d < 0:
                colors.append('#f0ad4e')  # amber
            else:
                colors.append('#5cb85c')  # green

        y_pos = range(len(df))
        bars = ax.barh(y_pos, df[metric], color=colors,
                       xerr=[df[metric] - df['ci_lower'], df['ci_upper'] - df[metric]],
                       capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['slice_name'])
        ax.axvline(x=self.global_metrics[metric], color='black',
                   linestyle='--', linewidth=1, label=f'Global {metric}')
        ax.set_xlabel(metric)
        ax.set_title(f'Worst Slices by {metric}')
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
