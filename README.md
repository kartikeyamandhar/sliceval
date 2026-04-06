# sliceval

**Find where your model fails before production does.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen.svg)]()


`sliceval` breaks your test set into subgroups, measures each one,
and tells you exactly where your model is failing — with statistical proof.

---

## The Problem

```
Overall F1: 0.91 ← looks great, ship it

Actually:
  Sensor Type A (8,000 samples) — F1: 0.96  ✓
  Sensor Type B, day shift (800) — F1: 0.61  ✗
  Sensor Type B, night shift (200) — F1: 0.41  ✗✗✗

The 200-sample night group is 2% of the aggregate.
The global metric never moved. Production broke anyway.
```

This happens by default whenever subgroups have unequal sizes and different distributions. `sliceval` makes it visible.

---

## Table of Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Output Format](#output-format)
- [Manual Slices](#manual-slices)
- [Automatic Discovery](#automatic-discovery)
  - [Tree-Based Discovery](#tree-based-discovery-default)
  - [Beam Search Discovery](#beam-search-discovery)
  - [When to Use Which](#when-to-use-which)
- [Supported Metrics](#supported-metrics)
- [Confidence Intervals](#confidence-intervals)
- [MLflow Integration](#mlflow-integration)
- [Plotting](#plotting)
- [Regression Tasks](#regression-tasks)
- [Full API Reference](#full-api-reference)
  - [SliceEvaluator](#sliceevaluator)
  - [SliceReport](#slicereport)
  - [Slice](#slice)
  - [SliceMetrics](#slicemetrics)
- [Error Handling](#error-handling)
- [Development](#development)

---

## Install

```bash
pip install sliceval                  # core (numpy, pandas, scikit-learn)
pip install sliceval[mlflow]          # + MLflow artifact logging
pip install sliceval[plot]            # + matplotlib charts
pip install sliceval[all]             # everything
```

Requires Python 3.9+.

---

## Quick Start

```python
from sliceval import SliceEvaluator

# Wrap your trained model + test data
ev = SliceEvaluator(model, X_test, y_test)

# Add slices you care about
ev.add_slice('sensor_b', X_test['sensor_type'] == 'B')

# Auto-discover problematic slices
ev.discover_slices()

# Evaluate everything
report = ev.evaluate()

# See the worst performers
print(report.worst_slices())
```

That's it. Five lines from trained model to actionable diagnostics.

---

## Output Format

`report.worst_slices()` returns a pandas DataFrame:

```
   slice_name        n_samples  support    f1  ci_lower  ci_upper   delta  p_value  source
   sensor_b               100     0.10  0.41      0.33      0.49   -0.50    0.003  manual
   hour<=5 (auto)          85     0.09  0.55      0.44      0.65   -0.36    0.010  tree
```

Every row includes:

| Column | Meaning |
|---|---|
| `n_samples` | How many test samples fall in this slice |
| `support` | Fraction of the test set (n\_samples / total) |
| `{metric}` | The metric value for this slice |
| `ci_lower`, `ci_upper` | 95% confidence interval bounds |
| `delta` | Slice metric minus global metric (negative = worse) |
| `p_value` | Statistical significance vs. global performance |
| `source` | `manual`, `tree`, or `beam` |

`report.to_dataframe()` gives the full matrix across all slices and all metrics, with a `[global]` row first.

---

## Manual Slices

Define slices with boolean masks or callables:

```python
# Boolean mask (evaluated immediately)
ev.add_slice('sensor_b', X_test['sensor_type'] == 'B')
ev.add_slice('high_temp', X_test['temperature'] > 80)

# Callable (evaluated lazily at .evaluate() time)
ev.add_slice(
    'night_shift',
    lambda X: X['hour'].between(22, 23) | X['hour'].between(0, 5)
)

# Intersections — combine masks yourself
ev.add_slice(
    'sensor_b_night',
    (X_test['sensor_type'] == 'B') & (X_test['hour'] < 6)
)
```

**Safety rails:**

- Slices with **0 samples** → `ValueError` at evaluation time
- Slices with **< 30 samples** → `UserWarning` about unreliable metrics
- **Duplicate names** → overwrites previous slice with a warning

---

## Automatic Discovery

Don't know where your model fails? Let `sliceval` find out.

### Tree-Based Discovery (default)

Fits a shallow decision tree that predicts **model errors**. Each leaf becomes a candidate slice — a region of feature space where the model systematically fails.

```python
ev.discover_slices(
    method='tree',
    max_depth=3,          # max feature conjunctions (depth 2 = A AND B)
    min_support=0.05,     # slice must cover ≥5% of test set
    metric='f1',          # rank slices by this metric's drop
    n_slices=10,          # return at most 10 slices
    significance=0.05,    # drop slices with p-value > 0.05
)
```

Discovered slices are auto-labeled with human-readable conditions:

```
sensor_type == B AND hour <= 5.5 (auto)
temperature > 82.3 (auto)
```

### Beam Search Discovery

Enumerates feature conjunctions breadth-first. More exhaustive than tree-based discovery, but slower on wide feature spaces. This implements the SliceFinder algorithm (Chung et al. 2019).

```python
ev.discover_slices(
    method='beam',
    max_depth=2,
    min_support=0.05,
    beam_width=10,        # candidates kept at each depth
)
```

### When to Use Which

| Method | Speed | Best For |
|---|---|---|
| `tree` | Fast | Axis-aligned decision boundaries, first pass |
| `beam` | Slower | Exhaustive search, complex feature interactions |

**Default is `tree`.** Most users will never need `beam`.

Calling `discover_slices()` multiple times **appends** — it does not replace previously discovered slices.

---

## Supported Metrics

### Classification

| Metric | Name | Notes |
|---|---|---|
| `'f1'` | F1 Score | `average='binary'` or `'macro'` for multiclass |
| `'precision'` | Precision | Same averaging |
| `'recall'` | Recall | Same averaging |
| `'accuracy'` | Accuracy | — |
| `'auc'` | ROC AUC | Requires `model.predict_proba()` |
| `'ece'` | Expected Calibration Error | Requires `model.predict_proba()` |

### Regression

| Metric | Name |
|---|---|
| `'rmse'` | Root Mean Squared Error |
| `'mae'` | Mean Absolute Error |

**Defaults:**
- Classification: `['f1', 'precision', 'recall']`
- Regression: `['rmse', 'mae']`

Metrics that need `predict_proba` (`auc`, `ece`) will raise `ValueError` at evaluation time if the model doesn't have it. You'll know immediately, not silently.

---

## Confidence Intervals

Every metric on every slice gets a confidence interval. Two methods available:

| Method | How It Works | When to Use |
|---|---|---|
| `'bootstrap'` (default) | Resample with replacement N times, take percentiles | Always works, any metric |
| `'wilson'` | Wilson score interval for proportions | Binary classification only; faster for precision/recall/accuracy |

```python
ev = SliceEvaluator(
    model, X_test, y_test,
    ci_method='bootstrap',   # or 'wilson'
    ci_alpha=0.05,           # 95% CI (1 - alpha)
    n_bootstrap=1000,        # more = tighter CI, slower
)
```

When `ci_method='wilson'` is set but a metric doesn't support Wilson (e.g., F1, AUC), it silently falls back to bootstrap. No warning, no error — Wilson is a preference, not a hard requirement.

---

## MLflow Integration

Log slice evaluation results as MLflow artifacts in one call:

```python
import mlflow
from sliceval import SliceEvaluator

with mlflow.start_run():
    # ... your training code ...

    ev = SliceEvaluator(model, X_test, y_test,
                        metrics=['f1', 'precision', 'recall'])
    ev.add_slice('sensor_b', X_test['sensor_type'] == 'B')
    ev.discover_slices(method='tree', max_depth=2)
    report = ev.evaluate()

    report.to_mlflow()  # uses active run automatically
```

### What Gets Logged

```
artifacts/
  slice_eval/
    slice_report.csv        ← full slice × metric DataFrame
    worst_slices.csv        ← top 10 worst slices
    global_metrics.json     ← {"f1": 0.91, "precision": 0.89, ...}
    slice_summary.json      ← {"n_slices": 14, "n_manual": 2, ...}
```

### Options

```python
report.to_mlflow(
    run_id='abc123',              # explicit run ID (default: active run)
    artifact_path='my_eval',      # artifact subdirectory (default: 'slice_eval')
)
```

Requires `pip install sliceval[mlflow]`. Raises `ImportError` with install instructions if missing.

---

## Plotting

```python
fig = report.plot(metric='f1', top_n=10, figsize=(10, 6))
fig.savefig('slice_eval.png')
```

Produces a horizontal bar chart:

- **Red bars**: delta < -0.1 (significantly worse than global)
- **Amber bars**: -0.1 ≤ delta < 0 (somewhat worse)
- **Green bars**: delta ≥ 0 (at or above global)
- **Dashed line**: global metric baseline
- **Error bars**: 95% confidence intervals

Requires `pip install sliceval[plot]`. Raises `ImportError` with install instructions if missing.

---

## Regression Tasks

Everything works the same — just set `task='regression'`:

```python
ev = SliceEvaluator(
    model, X_test, y_test,
    task='regression',
    metrics=['rmse', 'mae'],
)

ev.add_slice('region_west', X_test['region'] == 'west')
ev.discover_slices(method='tree', metric='rmse', max_depth=2)

report = ev.evaluate()
print(report.worst_slices(metric='rmse'))
```

Discovery uses absolute error per sample as the error signal for regression.

---

## Full API Reference

### SliceEvaluator

```python
SliceEvaluator(
    model,                          # any object with .predict()
    X: pd.DataFrame,                # test features (must be DataFrame)
    y: pd.Series | np.ndarray,      # ground truth labels
    task: str = 'binary',           # 'binary' | 'multiclass' | 'regression'
    metrics: list = None,           # default depends on task
    ci_method: str = 'bootstrap',   # 'bootstrap' | 'wilson'
    ci_alpha: float = 0.05,         # confidence level = 1 - ci_alpha
    n_bootstrap: int = 1000,        # bootstrap iterations
    average: str = 'macro',         # multiclass averaging strategy
    random_state: int = 42,
)
```

#### `add_slice(name, mask)`

```python
ev.add_slice(
    name: str,                      # human-readable label
    mask,                           # pd.Series[bool] | np.ndarray[bool] | callable
)
```

If `mask` is a callable, it receives `X` and must return a boolean Series/array. Evaluated lazily at `.evaluate()` time.

#### `discover_slices(method, **kwargs)`

```python
ev.discover_slices(
    method: str = 'tree',           # 'tree' | 'beam'
    max_depth: int = 3,             # max feature conjunctions
    min_support: float = 0.05,      # min fraction of test set
    metric: str = 'f1',             # metric to rank by
    n_slices: int = 10,             # max slices to return
    significance: float = 0.05,     # p-value threshold
)
```

Must be called **before** `.evaluate()`. Calling multiple times appends.

#### `evaluate() → SliceReport`

Runs inference once, computes all metrics on all slices, returns a `SliceReport`.

---

### SliceReport

Returned by `ev.evaluate()`.

| Attribute | Type | Description |
|---|---|---|
| `global_metrics` | `dict` | `{'f1': 0.91, ...}` |
| `slices` | `list[Slice]` | All evaluated slices |
| `metrics` | `list[SliceMetrics]` | Per-slice results (same order as `slices`) |
| `task` | `str` | Task type |
| `evaluated_at` | `datetime` | UTC timestamp |

#### `worst_slices(n=5, metric=None, min_support=0.0) → pd.DataFrame`

Returns `n` worst slices sorted by `delta` ascending (most negative first). Filterable by minimum support.

#### `to_dataframe() → pd.DataFrame`

Full slice × metric matrix. First row is always `[global]`. Columns per metric: `{m}_value`, `{m}_ci_lower`, `{m}_ci_upper`, `{m}_delta`, `{m}_p_value`.

#### `to_mlflow(run_id=None, artifact_path='slice_eval')`

Logs CSV and JSON artifacts to MLflow. Uses active run if `run_id` is None.

#### `plot(metric=None, top_n=10, figsize=(10, 6)) → Figure`

Horizontal bar chart with CI error bars and global baseline.

---

### Slice

Dataclass. Accessible via `report.slices`.

```python
@dataclass
class Slice:
    name: str                       # human-readable label
    mask: np.ndarray                # boolean, shape (n_test_samples,)
    n_samples: int                  # count of True entries
    support: float                  # n_samples / len(X_test)
    source: str                     # 'manual' | 'tree' | 'beam'
    feature_conditions: list        # e.g. ['sensor_type == B', 'hour < 6']
```

### SliceMetrics

Dataclass. Accessible via `report.metrics`.

```python
@dataclass
class SliceMetrics:
    slice_name: str
    n_samples: int
    support: float
    metrics: dict                   # {'f1': 0.41, ...}
    ci_lower: dict                  # {'f1': 0.33, ...}
    ci_upper: dict                  # {'f1': 0.49, ...}
    delta: dict                     # {'f1': -0.50, ...} (slice - global)
    p_value: dict                   # {'f1': 0.003, ...}
```

---

## Error Handling

`sliceval` fails loudly with descriptive messages. No silent corruption.

| Situation | Exception | What You See |
|---|---|---|
| `X` is not a DataFrame | `TypeError` | `X must be a pd.DataFrame, got ndarray` |
| `len(X) != len(y)` | `ValueError` | `X and y must have the same length. Got X: 500, y: 400` |
| Invalid task string | `ValueError` | `task must be 'binary', 'multiclass', or 'regression'. Got: 'classify'` |
| Unknown metric name | `ValueError` | `Unknown metric 'f2'. Valid metrics: [...]` |
| `auc`/`ece` without `predict_proba` | `ValueError` | `Metric 'auc' requires model.predict_proba()` |
| Slice mask wrong length | `ValueError` | `Slice 'my_slice' mask has length 50, expected 1000` |
| Empty slice | `ValueError` | `Slice 'my_slice' has 0 samples. Check your mask condition.` |
| Discovery metric not in evaluator | `ValueError` | `Discovery metric 'auc' is not in the evaluator's metric list` |
| MLflow not installed | `ImportError` | `MLflow integration requires: pip install sliceval[mlflow]` |
| matplotlib not installed | `ImportError` | `Plotting requires: pip install sliceval[plot]` |

### Warnings (non-fatal)

| Situation | Warning |
|---|---|
| Slice with < 30 samples | `Slice 'x' has 15 samples. Metrics may be unreliable.` |
| Duplicate slice name | `Slice 'x' already exists and will be overwritten.` |
| No slices defined before evaluate | `No slices defined. Call add_slice() or discover_slices() before evaluate().` |

---

## Development

```bash
git clone https://github.com/kartikeyamandhar/sliceval.git
cd sliceval
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
pytest tests/ -v
```

### Project Structure

```
sliceval/
├── __init__.py                 # public API re-exports
├── evaluator.py                # SliceEvaluator (main entry point)
├── slice.py                    # Slice, SliceMetrics dataclasses
├── metrics.py                  # metric computation + confidence intervals
├── report.py                   # SliceReport output container
├── discovery/
│   ├── tree.py                 # decision tree-based discovery
│   └── beam.py                 # beam search discovery (SliceFinder)
├── integrations/
│   └── mlflow.py               # MLflow artifact export
└── utils/
    ├── stats.py                # bootstrap CI, Wilson CI, permutation tests
    └── validation.py           # input validation, error messages

tests/
├── conftest.py                 # shared fixtures
├── test_evaluator.py           # evaluator construction + evaluate flow
├── test_metrics.py             # metric computation, CI, p-values
├── test_discovery_tree.py      # tree discovery
├── test_discovery_beam.py      # beam search discovery
├── test_report.py              # report output format
├── test_integration_mlflow.py  # MLflow logging (mocked)
├── test_validation.py          # all ValueError/TypeError cases
└── test_warnings.py            # all UserWarning cases
```

### Design Principles

- **Non-invasive**: wraps any sklearn-compatible model, no training code changes
- **Composable**: each component usable independently
- **Statistically honest**: every metric has CI, sample size, and significance test
- **Lazy evaluation**: `model.predict()` called once, all slices reuse cached predictions
- **Zero mandatory dependencies beyond sklearn**: `mlflow`, `scipy`, `matplotlib` are optional

---

## License

MIT