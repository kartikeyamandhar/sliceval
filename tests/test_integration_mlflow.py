"""Tests for MLflow integration (mocked)."""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from sliceval import SliceEvaluator


class TestMLflowIntegration:
    def _make_report(self, perfect_model, binary_data):
        X, y = binary_data
        perfect_model.fit(X, y)
        ev = SliceEvaluator(perfect_model, X, y, metrics=['f1'], n_bootstrap=50)
        ev.add_slice('majority', X['sensor_type'] == 'A')
        ev.add_slice('minority', X['sensor_type'] == 'B')
        return ev.evaluate()

    def _run_with_mock_mlflow(self, report, run_id=None, artifact_path='slice_eval'):
        """Inject a mock mlflow module and call log_slice_report."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = 'test-run-123'
        mock_mlflow.active_run.return_value = mock_run
        mock_client = MagicMock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        captured_files = {}

        def capture_log(rid, local_dir, artifact_path=None):
            for fname in os.listdir(local_dir):
                with open(os.path.join(local_dir, fname), 'r') as f:
                    captured_files[fname] = f.read()

        mock_client.log_artifacts.side_effect = capture_log

        with patch.dict(sys.modules, {'mlflow': mock_mlflow, 'mlflow.tracking': mock_mlflow.tracking}):
            # Re-import to pick up the mock
            import importlib
            import sliceval.integrations.mlflow as mlflow_mod
            importlib.reload(mlflow_mod)
            mlflow_mod.log_slice_report(report, run_id=run_id, artifact_path=artifact_path)

        return mock_client, captured_files

    def test_to_mlflow_logs_artifacts(self, perfect_model, binary_data):
        report = self._make_report(perfect_model, binary_data)
        mock_client, _ = self._run_with_mock_mlflow(report)
        mock_client.log_artifacts.assert_called_once()

    def test_to_mlflow_creates_expected_files(self, perfect_model, binary_data):
        report = self._make_report(perfect_model, binary_data)
        _, files = self._run_with_mock_mlflow(report)
        assert 'slice_report.csv' in files
        assert 'worst_slices.csv' in files
        assert 'global_metrics.json' in files
        assert 'slice_summary.json' in files

    def test_global_metrics_json_content(self, perfect_model, binary_data):
        report = self._make_report(perfect_model, binary_data)
        _, files = self._run_with_mock_mlflow(report)
        gm = json.loads(files['global_metrics.json'])
        assert 'f1' in gm

    def test_slice_summary_json_content(self, perfect_model, binary_data):
        report = self._make_report(perfect_model, binary_data)
        _, files = self._run_with_mock_mlflow(report)
        s = json.loads(files['slice_summary.json'])
        assert s['n_slices'] == 2
        assert s['n_manual'] == 2
        assert s['n_discovered'] == 0
        assert s['task'] == 'binary'
        assert 'evaluated_at' in s

    def test_custom_run_id(self, perfect_model, binary_data):
        report = self._make_report(perfect_model, binary_data)
        mock_client, _ = self._run_with_mock_mlflow(report, run_id='custom-run-id')
        call_args = mock_client.log_artifacts.call_args
        assert call_args[0][0] == 'custom-run-id'

    def test_custom_artifact_path(self, perfect_model, binary_data):
        report = self._make_report(perfect_model, binary_data)
        mock_client, _ = self._run_with_mock_mlflow(report, artifact_path='custom/path')
        call_args = mock_client.log_artifacts.call_args
        ap = call_args[1].get('artifact_path') or call_args[0][2]
        assert ap == 'custom/path'

    def test_import_error_without_mlflow(self, perfect_model, binary_data):
        report = self._make_report(perfect_model, binary_data)
        with patch.dict(sys.modules, {'mlflow': None}):
            import importlib
            import sliceval.integrations.mlflow as mlflow_mod
            importlib.reload(mlflow_mod)
            with pytest.raises(ImportError, match="pip install sliceval"):
                mlflow_mod.log_slice_report(report)