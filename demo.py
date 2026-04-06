"""sliceval demo — real dataset (sklearn breast cancer)."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sliceval import SliceEvaluator

# ─── 1. Load real data ───
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='malignant')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ─── 2. Train a real model ───
model = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

# ─── 3. The misleading global number ───
y_pred = model.predict(X_test)
print("=" * 65)
print(f"GLOBAL F1: {f1_score(y_test, y_pred):.4f}")
print("  ^ Looks excellent. Ship it?")
print("=" * 65)

# ─── 4. sliceval — what's actually happening ───
print("\n>>> Running sliceval...\n")

ev = SliceEvaluator(
    model, X_test, y_test,
    task='binary',
    metrics=['f1', 'precision', 'recall', 'auc'],
    n_bootstrap=500,
)

# Manual slices based on domain knowledge:
# Tumor size features
ev.add_slice('large_radius', X_test['mean radius'] > X_test['mean radius'].quantile(0.75))
ev.add_slice('small_radius', X_test['mean radius'] < X_test['mean radius'].quantile(0.25))
ev.add_slice('high_concavity', X_test['mean concavity'] > X_test['mean concavity'].quantile(0.75))
ev.add_slice('low_texture', X_test['mean texture'] < X_test['mean texture'].quantile(0.25))

# Borderline cases: mid-range features where the model is least certain
ev.add_slice(
    'borderline_radius',
    X_test['mean radius'].between(
        X_test['mean radius'].quantile(0.4),
        X_test['mean radius'].quantile(0.6)
    )
)

# Auto-discover
ev.discover_slices(method='tree', max_depth=2, min_support=0.05,
                   n_slices=10, significance=1.0)

report = ev.evaluate()

# ─── 5. Results ───
print("─── Global Metrics ───")
for k, v in report.global_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n─── Worst 8 Slices (by F1) ───")
worst = report.worst_slices(n=8, metric='f1')
print(worst.to_string(index=False))

print("\n─── Full Report ───")
df = report.to_dataframe()
cols = ['slice_name', 'source', 'n_samples', 'f1_value',
        'f1_ci_lower', 'f1_ci_upper', 'f1_delta', 'f1_p_value']
print(df[cols].to_string(index=False))

print("\n─── Recall Breakdown (missed diagnoses matter) ───")
worst_recall = report.worst_slices(n=5, metric='recall')
print(worst_recall.to_string(index=False))

print("\n" + "=" * 65)
print("In cancer diagnosis, a missed malignant case (low recall) kills.")
print("The global metric can't show you where that's happening.")
print("sliceval can.")
print("=" * 65)