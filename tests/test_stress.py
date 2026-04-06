"""Stress test: run sliceval against many real sklearn datasets and model types."""

import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    load_digits,
    load_diabetes,
    load_linnerud,
    fetch_california_housing,
    make_classification,
    make_regression,
    make_moons,
    make_circles,
    make_blobs,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sliceval import SliceEvaluator

# ─── Helpers ───

def _make_df(X, feature_names=None):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names)


def _run_sliceval(model, X_train, X_test, y_train, y_test, task, metrics,
                  discover=True, method='tree'):
    """Full sliceval run. Returns report or raises."""
    model.fit(X_train, y_train)

    ev = SliceEvaluator(
        model, X_test, y_test,
        task=task,
        metrics=metrics,
        n_bootstrap=50,  # fast for stress testing
        random_state=42,
    )

    # Manual slices: pick a feature with variance
    col = None
    for c in X_test.columns:
        if X_test[c].nunique() > 2:
            col = c
            break
    if col is None:
        col = X_test.columns[0]

    q25 = X_test[col].quantile(0.25)
    q75 = X_test[col].quantile(0.75)

    if (X_test[col] <= q25).sum() > 0:
        ev.add_slice('low_f0', X_test[col] <= q25)
    if (X_test[col] >= q75).sum() > 0:
        ev.add_slice('high_f0', X_test[col] >= q75)

    # Callable slice — use >= to handle median ties
    median_val = X_test[col].median()
    if (X_test[col] > median_val).sum() > 0:
        ev.add_slice('above_median_f0', lambda X, m=median_val, c=col: X[c] > m)
    elif (X_test[col] >= median_val).sum() > 0:
        ev.add_slice('at_or_above_median_f0', lambda X, m=median_val, c=col: X[c] >= m)

    if discover:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev.discover_slices(method=method, max_depth=2,
                               min_support=0.05, significance=1.0, n_slices=5,
                               metric=metrics[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ev.evaluate()

    # Validate report structure
    assert report is not None
    assert len(report.global_metrics) == len(metrics)
    assert report.evaluated_at is not None

    df = report.to_dataframe()
    assert df.iloc[0]['slice_name'] == '[global]'
    assert len(df) >= 1  # at least global row

    worst = report.worst_slices(n=3, metric=metrics[0])
    assert isinstance(worst, pd.DataFrame)

    return report

# ─── Binary Classification Datasets ───

BINARY_DATASETS = []

def _add_binary(name, loader):
    BINARY_DATASETS.append(pytest.param(loader, id=name))

_add_binary("breast_cancer", lambda: load_breast_cancer(return_X_y=True))
_add_binary("make_moons", lambda: make_moons(n_samples=500, noise=0.3, random_state=42))
_add_binary("make_circles", lambda: make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=42))
_add_binary("make_clf_100x10", lambda: make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42))
_add_binary("make_clf_100x20", lambda: make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42))
_add_binary("make_clf_imbalanced", lambda: make_classification(n_samples=500, n_features=10, weights=[0.9, 0.1], random_state=42))
_add_binary("make_clf_noisy", lambda: make_classification(n_samples=500, n_features=10, flip_y=0.2, random_state=42))


BINARY_MODELS = [
    pytest.param(lambda: RandomForestClassifier(n_estimators=30, random_state=42), id="rf"),
    pytest.param(lambda: GradientBoostingClassifier(n_estimators=30, random_state=42), id="gbm"),
    pytest.param(lambda: LogisticRegression(max_iter=1000, random_state=42), id="lr"),
    pytest.param(lambda: DecisionTreeClassifier(max_depth=5, random_state=42), id="dt"),
    pytest.param(lambda: KNeighborsClassifier(n_neighbors=5), id="knn"),
    pytest.param(lambda: GaussianNB(), id="nb"),
    pytest.param(lambda: SVC(probability=True, random_state=42), id="svc"),
]


@pytest.mark.parametrize("data_loader", BINARY_DATASETS)
@pytest.mark.parametrize("model_factory", BINARY_MODELS)
def test_binary_classification(data_loader, model_factory):
    X_raw, y_raw = data_loader()
    X = _make_df(X_raw)
    y = pd.Series(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = model_factory()
    _run_sliceval(model, X_train, X_test, y_train, y_test,
                  task='binary', metrics=['f1', 'precision', 'recall'])


# ─── Multiclass Classification Datasets ───

MULTI_DATASETS = []

def _add_multi(name, loader):
    MULTI_DATASETS.append(pytest.param(loader, id=name))

_add_multi("iris", lambda: load_iris(return_X_y=True))
_add_multi("wine", lambda: load_wine(return_X_y=True))
_add_multi("digits_subset", lambda: _digits_subset())
_add_multi("make_blobs_5", lambda: make_blobs(n_samples=500, centers=5, random_state=42))

def _digits_subset():
    X, y = load_digits(return_X_y=True)
    # Use only first 500 samples for speed
    return X[:500], y[:500]


MULTI_MODELS = [
    pytest.param(lambda: RandomForestClassifier(n_estimators=30, random_state=42), id="rf"),
    pytest.param(lambda: GradientBoostingClassifier(n_estimators=30, random_state=42), id="gbm"),
    pytest.param(lambda: LogisticRegression(max_iter=1000, random_state=42), id="lr"),
    pytest.param(lambda: DecisionTreeClassifier(max_depth=5, random_state=42), id="dt"),
    pytest.param(lambda: KNeighborsClassifier(n_neighbors=5), id="knn"),
]


@pytest.mark.parametrize("data_loader", MULTI_DATASETS)
@pytest.mark.parametrize("model_factory", MULTI_MODELS)
def test_multiclass_classification(data_loader, model_factory):
    X_raw, y_raw = data_loader()
    X = _make_df(X_raw)
    y = pd.Series(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = model_factory()
    _run_sliceval(model, X_train, X_test, y_train, y_test,
                  task='multiclass', metrics=['f1', 'accuracy'])


# ─── Regression Datasets ───

REG_DATASETS = []

def _add_reg(name, loader):
    REG_DATASETS.append(pytest.param(loader, id=name))

_add_reg("diabetes", lambda: load_diabetes(return_X_y=True))
_add_reg("california_housing", lambda: _california_subset())
_add_reg("make_reg_10", lambda: make_regression(n_samples=500, n_features=10, noise=10, random_state=42))
_add_reg("make_reg_20", lambda: make_regression(n_samples=500, n_features=20, noise=20, random_state=42))

def _california_subset():
    X, y = fetch_california_housing(return_X_y=True)
    # Subset for speed
    return X[:800], y[:800]


REG_MODELS = [
    pytest.param(lambda: RandomForestRegressor(n_estimators=30, random_state=42), id="rf"),
    pytest.param(lambda: GradientBoostingRegressor(n_estimators=30, random_state=42), id="gbm"),
    pytest.param(lambda: Ridge(random_state=42), id="ridge"),
    pytest.param(lambda: Lasso(random_state=42), id="lasso"),
    pytest.param(lambda: DecisionTreeRegressor(max_depth=5, random_state=42), id="dt"),
    pytest.param(lambda: KNeighborsRegressor(n_neighbors=5), id="knn"),
]


@pytest.mark.parametrize("data_loader", REG_DATASETS)
@pytest.mark.parametrize("model_factory", REG_MODELS)
def test_regression(data_loader, model_factory):
    X_raw, y_raw = data_loader()
    X = _make_df(X_raw)
    y = pd.Series(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = model_factory()
    _run_sliceval(model, X_train, X_test, y_train, y_test,
                  task='regression', metrics=['rmse', 'mae'])


# ─── Discovery Method Variants ───

@pytest.mark.parametrize("method", ['tree', 'beam'])
@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_discovery_variants(method, max_depth):
    X_raw, y_raw = load_breast_cancer(return_X_y=True)
    X = _make_df(X_raw)
    y = pd.Series(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)

    ev = SliceEvaluator(model, X_test, y_test, task='binary',
                        metrics=['f1'], n_bootstrap=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.discover_slices(method=method, max_depth=max_depth,
                           min_support=0.05, significance=1.0)
    report = ev.evaluate()
    assert report is not None
    assert len(report.to_dataframe()) >= 1


# ─── CI Method Variants ───

@pytest.mark.parametrize("ci_method", ['bootstrap', 'wilson'])
def test_ci_methods(ci_method):
    X_raw, y_raw = load_breast_cancer(return_X_y=True)
    X = _make_df(X_raw)
    y = pd.Series(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    ev = SliceEvaluator(model, X_test, y_test, task='binary',
                        metrics=['f1', 'precision', 'recall', 'accuracy'],
                        ci_method=ci_method, n_bootstrap=50)
    ev.add_slice('low', X_test.iloc[:, 0] <= X_test.iloc[:, 0].median())
    report = ev.evaluate()
    m = report.metrics[0]
    for metric in ['f1', 'precision', 'recall', 'accuracy']:
            assert m.ci_lower[metric] <= m.metrics[metric] + 1e-9
            assert m.metrics[metric] <= m.ci_upper[metric] + 1e-9


# ─── Edge Cases ───

def test_single_feature_dataset():
    X = pd.DataFrame({'x': np.random.normal(0, 1, 200)})
    y = pd.Series((X['x'] > 0).astype(int))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(random_state=42)
    _run_sliceval(model, X_train, X_test, y_train, y_test,
                  task='binary', metrics=['f1'])


def test_many_features():
    X_raw, y_raw = make_classification(n_samples=300, n_features=50,
                                        n_informative=10, random_state=42)
    X = _make_df(X_raw)
    y = pd.Series(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    _run_sliceval(model, X_train, X_test, y_train, y_test,
                  task='binary', metrics=['f1'], method='tree')


def test_perfectly_separable():
    np.random.seed(42)
    X = pd.DataFrame({'x': np.concatenate([np.ones(100), np.zeros(100)])})
    y = pd.Series(np.concatenate([np.ones(100), np.zeros(100)]).astype(int))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    _run_sliceval(model, X_train, X_test, y_train, y_test,
                  task='binary', metrics=['f1', 'accuracy'], discover=False)


def test_all_same_prediction():
    """Model that predicts the same class for everything."""
    X = pd.DataFrame({'x': np.random.normal(0, 1, 200)})
    y = pd.Series(np.ones(200, dtype=int))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    class AlwaysOne:
        def predict(self, X): return np.ones(len(X), dtype=int)
    
    ev = SliceEvaluator(AlwaysOne(), X_test, y_test, task='binary',
                        metrics=['accuracy'], n_bootstrap=50)
    ev.add_slice('half', X_test['x'] > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = ev.evaluate()
    assert report.global_metrics['accuracy'] == 1.0


def test_high_cardinality_feature():
    """Dataset with a column that has many unique values."""
    np.random.seed(42)
    X = pd.DataFrame({
        'id': np.arange(300),
        'feature': np.random.normal(0, 1, 300),
    })
    y = pd.Series(np.random.randint(0, 2, 300))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    _run_sliceval(model, X_train, X_test, y_train, y_test,
                  task='binary', metrics=['f1'])


def test_report_plot_does_not_crash():
    """Ensure plot() runs without error when matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
    except ImportError:
        pytest.skip("matplotlib not installed")

    X_raw, y_raw = load_breast_cancer(return_X_y=True)
    X = _make_df(X_raw)
    y = pd.Series(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)

    ev = SliceEvaluator(model, X_test, y_test, task='binary',
                        metrics=['f1'], n_bootstrap=50)
    ev.add_slice('low', X_test.iloc[:, 0] <= X_test.iloc[:, 0].median())
    report = ev.evaluate()
    fig = report.plot(metric='f1', top_n=5)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)