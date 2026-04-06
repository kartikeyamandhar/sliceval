"""MLflow integration for sliceval."""

import json
import os
import tempfile


def log_slice_report(report, run_id=None, artifact_path='slice_eval'):
    """Log a SliceReport as MLflow artifacts.

    Logs:
        slice_report.csv   — full to_dataframe() output
        worst_slices.csv   — top 10 worst slices
        global_metrics.json — global metric values
        slice_summary.json  — metadata about the evaluation

    Args:
        report: SliceReport instance.
        run_id: MLflow run ID. Uses active run if None.
        artifact_path: Artifact subdirectory in MLflow.
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow integration requires: pip install sliceval[mlflow]")

    client = mlflow.tracking.MlflowClient()

    if run_id is None:
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError(
                "No active MLflow run. Either pass run_id or call "
                "within an mlflow.start_run() context."
            )
        active_run_id = active_run.info.run_id
    else:
        active_run_id = run_id

    with tempfile.TemporaryDirectory() as tmp:
        # slice_report.csv
        report.to_dataframe().to_csv(
            os.path.join(tmp, 'slice_report.csv'), index=False
        )

        # worst_slices.csv
        report.worst_slices(n=10).to_csv(
            os.path.join(tmp, 'worst_slices.csv'), index=False
        )

        # global_metrics.json
        with open(os.path.join(tmp, 'global_metrics.json'), 'w') as f:
            json.dump(report.global_metrics, f, indent=2)

        # slice_summary.json
        with open(os.path.join(tmp, 'slice_summary.json'), 'w') as f:
            json.dump({
                'n_slices': len(report.slices),
                'n_manual': sum(1 for s in report.slices if s.source == 'manual'),
                'n_discovered': sum(1 for s in report.slices if s.source != 'manual'),
                'evaluated_at': report.evaluated_at.isoformat(),
                'task': report.task,
            }, f, indent=2)

        # Log all files as artifacts
        client.log_artifacts(active_run_id, tmp, artifact_path=artifact_path)