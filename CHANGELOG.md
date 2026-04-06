# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-XX

### Added
- `SliceEvaluator` — main entry point wrapping any sklearn-compatible model
- Manual slice definition via boolean masks or callables
- Tree-based automatic slice discovery
- Beam search slice discovery (SliceFinder algorithm)
- 8 metrics: F1, precision, recall, accuracy, AUC, ECE, RMSE, MAE
- Bootstrap and Wilson confidence intervals
- Permutation-based significance testing
- `SliceReport` with `worst_slices()`, `to_dataframe()`, `plot()`
- MLflow artifact integration via `to_mlflow()`
- Binary, multiclass, and regression task support
- 162 tests including stress tests across 7 datasets and 13 model types
