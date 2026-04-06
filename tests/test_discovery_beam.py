"""Tests for beam search slice discovery."""

import numpy as np
import pandas as pd
import pytest

from sliceval import SliceEvaluator


class TestBeamDiscovery:
    def test_discovers_slices(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=2, min_support=0.05,
                           significance=1.0)
        report = ev.evaluate()
        assert len(report.slices) > 0

    def test_discovered_slices_labeled_auto(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=2, min_support=0.05,
                           significance=1.0)
        report = ev.evaluate()
        for s in report.slices:
            assert "(auto)" in s.name

    def test_source_is_beam(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=2, min_support=0.05,
                           significance=1.0)
        report = ev.evaluate()
        for s in report.slices:
            assert s.source == 'beam'

    def test_beam_finds_sensor_type(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=2, min_support=0.05,
                           significance=1.0)
        report = ev.evaluate()
        names = [s.name for s in report.slices]
        assert any('sensor_type' in n for n in names)

    def test_min_support_respected(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=2, min_support=0.3,
                           significance=1.0)
        report = ev.evaluate()
        for s in report.slices:
            assert s.support >= 0.3

    def test_n_slices_limits_output(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=2, min_support=0.05,
                           n_slices=3, significance=1.0)
        report = ev.evaluate()
        assert len(report.slices) <= 3

    def test_feature_conditions_populated(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=2, min_support=0.05,
                           significance=1.0)
        report = ev.evaluate()
        for s in report.slices:
            assert len(s.feature_conditions) > 0

    def test_depth_1_single_predicates(self, biased_model, binary_data):
        X, y = binary_data
        biased_model.fit(X, y)
        ev = SliceEvaluator(biased_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.discover_slices(method='beam', max_depth=1, min_support=0.05,
                           significance=1.0)
        report = ev.evaluate()
        for s in report.slices:
            # Depth 1 = single predicates only
            assert len(s.feature_conditions) == 1

    def test_beam_with_regression(self, regression_model, regression_data):
        X, y = regression_data
        np.random.seed(42)
        regression_model.fit(X, y)
        ev = SliceEvaluator(regression_model, X, y, task='regression',
                            metrics=['rmse', 'mae'], n_bootstrap=50)
        ev.discover_slices(method='beam', metric='rmse', max_depth=2,
                           min_support=0.05, significance=1.0)
        report = ev.evaluate()
        # Should run without error; may or may not find significant slices
        assert report is not None