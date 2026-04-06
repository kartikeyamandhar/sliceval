"""Tests for input validation."""

import numpy as np
import pandas as pd
import pytest

from sliceval import SliceEvaluator


class TestConstructorValidation:
    def test_x_must_be_dataframe(self, perfect_model):
        X = np.random.rand(100, 3)
        y = pd.Series(np.random.randint(0, 2, 100))
        with pytest.raises(TypeError, match="pd.DataFrame"):
            SliceEvaluator(perfect_model, X, y)

    def test_length_mismatch(self, perfect_model):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="same length"):
            SliceEvaluator(perfect_model, X, y)

    def test_invalid_task(self, perfect_model):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="task must be"):
            SliceEvaluator(perfect_model, X, y, task='invalid')

    def test_invalid_metric(self, perfect_model):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="Unknown metric"):
            SliceEvaluator(perfect_model, X, y, metrics=['fake_metric'])

    def test_regression_metric_on_classification(self, perfect_model):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="only valid for task='regression'"):
            SliceEvaluator(perfect_model, X, y, task='binary', metrics=['rmse'])

    def test_classification_metric_on_regression(self, perfect_model):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0.5, 1.5, 2.5])
        with pytest.raises(ValueError, match="not valid for task='regression'"):
            SliceEvaluator(perfect_model, X, y, task='regression', metrics=['f1'])

    def test_y_as_ndarray_accepted(self, perfect_model):
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = np.array([0, 1, 0])
        ev = SliceEvaluator(perfect_model, X, y)
        assert len(ev.y) == 3


class TestSliceValidation:
    def test_zero_sample_slice(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        ev.add_slice('empty', X['sensor_type'] == 'NONEXISTENT')
        with pytest.raises(ValueError, match="0 samples"):
            ev.evaluate()

    def test_wrong_length_mask(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        ev.add_slice('bad', np.ones(5, dtype=bool))
        with pytest.raises(ValueError, match="mask has length"):
            ev.evaluate()


class TestDiscoveryValidation:
    def test_discovery_metric_not_in_list(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'])
        with pytest.raises(ValueError, match="Discovery metric"):
            ev.discover_slices(metric='precision')
