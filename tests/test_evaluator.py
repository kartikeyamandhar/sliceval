"""Tests for SliceEvaluator construction and evaluate flow."""

import numpy as np
import pandas as pd
import pytest

from sliceval import SliceEvaluator


class TestEvaluatorBasic:
    def test_construct_defaults(self, perfect_model, binary_data):
        X, y = binary_data
        ev = SliceEvaluator(perfect_model, X, y)
        assert ev.metric_names == ['f1', 'precision', 'recall']
        assert ev.task == 'binary'

    def test_construct_regression_defaults(self, regression_model, regression_data):
        X, y = regression_data
        ev = SliceEvaluator(regression_model, X, y, task='regression')
        assert ev.metric_names == ['rmse', 'mae']

    def test_evaluate_returns_report(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        ev.add_slice('all', pd.Series(np.ones(len(X), dtype=bool)))
        report = ev.evaluate()
        assert report is not None
        assert 'f1' in report.global_metrics
        assert len(report.slices) == 1
        assert len(report.metrics) == 1

    def test_callable_mask(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        ev.add_slice('night', lambda df: df['hour'] < 6)
        report = ev.evaluate()
        assert report.slices[0].name == 'night'
        assert report.slices[0].n_samples > 0

    def test_worst_slice_isolated(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'])
        ev.add_slice('minority', X['sensor_type'] == 'B')
        ev.add_slice('majority', X['sensor_type'] == 'A')
        report = ev.evaluate()
        worst = report.worst_slices(n=1)
        assert worst.iloc[0]['slice_name'] == 'minority'
        assert worst.iloc[0]['delta'] < 0


class TestMetricsComputation:
    def test_perfect_model_f1(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'], n_bootstrap=100)
        ev.add_slice('all', pd.Series(np.ones(len(X), dtype=bool)))
        report = ev.evaluate()
        assert report.global_metrics['f1'] == pytest.approx(1.0)

    def test_bootstrap_ci_coverage(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'], n_bootstrap=200)
        ev.add_slice('all', pd.Series(np.ones(len(X), dtype=bool)))
        report = ev.evaluate()
        m = report.metrics[0]
        assert m.ci_lower['f1'] <= m.metrics['f1'] <= m.ci_upper['f1']


class TestReport:
    def test_to_dataframe_has_global_row(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        ev.add_slice('all', pd.Series(np.ones(len(X), dtype=bool)))
        report = ev.evaluate()
        df = report.to_dataframe()
        assert df.iloc[0]['slice_name'] == '[global]'
        assert df.iloc[0]['source'] == 'global'
        assert 'f1_value' in df.columns

    def test_to_dataframe_shape(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1', 'precision'])
        ev.add_slice('a', X['sensor_type'] == 'A')
        ev.add_slice('b', X['sensor_type'] == 'B')
        report = ev.evaluate()
        df = report.to_dataframe()
        # 1 global + 2 slices
        assert len(df) == 3
        # Each metric gets 5 columns
        assert 'f1_value' in df.columns
        assert 'precision_delta' in df.columns

    def test_worst_slices_min_support(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'])
        ev.add_slice('majority', X['sensor_type'] == 'A')
        ev.add_slice('minority', X['sensor_type'] == 'B')
        report = ev.evaluate()
        # minority ~10% of data
        ws = report.worst_slices(n=5, min_support=0.5)
        assert all(ws['support'] >= 0.5)


class TestRegression:
    def test_regression_evaluate(self, regression_model, regression_data):
        X, y = regression_data
        np.random.seed(42)
        regression_model.fit(X, y)
        ev = SliceEvaluator(regression_model, X, y, task='regression',
                            metrics=['rmse', 'mae'], n_bootstrap=100)
        ev.add_slice('group_x', X['feature_b'] == 'X')
        report = ev.evaluate()
        assert 'rmse' in report.global_metrics
        assert 'mae' in report.global_metrics
        assert report.metrics[0].metrics['rmse'] >= 0
