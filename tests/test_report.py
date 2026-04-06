"""Tests for SliceReport output."""

import numpy as np
import pandas as pd
import pytest

from sliceval import SliceEvaluator


class TestSliceReport:
    def test_worst_slices_ordering(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.add_slice('majority', X['sensor_type'] == 'A')
        ev.add_slice('minority', X['sensor_type'] == 'B')
        report = ev.evaluate()
        ws = report.worst_slices(n=2)
        # Deltas should be ascending (most negative first)
        assert ws.iloc[0]['delta'] <= ws.iloc[1]['delta']

    def test_worst_slices_columns(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.add_slice('all', pd.Series(np.ones(len(X), dtype=bool)))
        report = ev.evaluate()
        ws = report.worst_slices(n=1)
        expected_cols = {'slice_name', 'n_samples', 'support', 'f1',
                         'ci_lower', 'ci_upper', 'delta', 'p_value', 'source'}
        assert set(ws.columns) == expected_cols

    def test_to_dataframe_global_row_delta_zero(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.add_slice('all', pd.Series(np.ones(len(X), dtype=bool)))
        report = ev.evaluate()
        df = report.to_dataframe()
        assert df.iloc[0]['f1_delta'] == 0.0

    def test_evaluated_at_set(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.add_slice('all', pd.Series(np.ones(len(X), dtype=bool)))
        report = ev.evaluate()
        assert report.evaluated_at is not None
