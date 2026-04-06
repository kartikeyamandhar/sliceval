"""Tests for UserWarning emission."""

import numpy as np
import pandas as pd
import pytest

from sliceval import SliceEvaluator


class TestWarnings:
    def test_small_slice_warning(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        small_mask = np.zeros(len(X), dtype=bool)
        small_mask[:15] = True
        ev.add_slice('tiny', small_mask)
        with pytest.warns(UserWarning, match="15 samples"):
            ev.evaluate()

    def test_duplicate_slice_warning(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        ev.add_slice('dup', X['sensor_type'] == 'A')
        with pytest.warns(UserWarning, match="already exists"):
            ev.add_slice('dup', X['sensor_type'] == 'B')

    def test_no_slices_warning(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'])
        with pytest.warns(UserWarning, match="No slices defined"):
            ev.evaluate()
