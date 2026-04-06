"""sliceval demo — see it work end to end."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sliceval import SliceEvaluator

# ─── 1. Synthetic dataset with a hidden failure mode ───
np.random.seed(42)
n = 2000

sensor_type = np.random.choice(['A', 'B'], size=n, p=[0.85, 0.15])
hour = np.random.randint(0, 24, size=n)
temperature = np.random.normal(65, 15, size=n)
vibration = np.random.normal(50, 10, size=n)

# Ground truth: failure depends on sensor type and hour
# Type A: predictable failures (high temp + high vibration)
# Type B at night: chaotic — model will struggle here
failure = np.zeros(n, dtype=int)
for i in range(n):
    if sensor_type[i] == 'A':
        if temperature[i] > 80 and vibration[i] > 60:
            failure[i] = 1
    else:
        if hour[i] < 6:
            # Night shift type B: noisy labels, hard to learn
            failure[i] = int(np.random.random() > 0.4)
        elif temperature[i] > 75:
            failure[i] = 1

X = pd.DataFrame({
    'sensor_type': sensor_type,
    'hour': hour,
    'temperature': temperature,
    'vibration': vibration,
})
y = pd.Series(failure, name='failure')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ─── 2. Train a model ───
# Encode categoricals for sklearn
X_train_enc = pd.get_dummies(X_train, dtype=float)
X_test_enc = pd.get_dummies(X_test, dtype=float)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_enc, y_train)

# Wrap model so it works with original DataFrame
class WrappedModel:
    def __init__(self, model):
        self._model = model
    def predict(self, X):
        return self._model.predict(pd.get_dummies(X, dtype=float))
    def predict_proba(self, X):
        return self._model.predict_proba(pd.get_dummies(X, dtype=float))

wrapped = WrappedModel(model)

# ─── 3. Standard evaluation — the misleading number ───
from sklearn.metrics import f1_score
y_pred = wrapped.predict(X_test)
print("=" * 60)
print(f"GLOBAL F1: {f1_score(y_test, y_pred):.3f}")
print("  ^ This is what gets reported. Looks fine.")
print("=" * 60)

# ─── 4. sliceval — the real picture ───
print("\n>>> Running sliceval...\n")

ev = SliceEvaluator(
    wrapped, X_test, y_test,
    task='binary',
    metrics=['f1', 'precision', 'recall'],
    n_bootstrap=500,
)

# Manual slices: things you suspect
ev.add_slice('sensor_b', X_test['sensor_type'] == 'B')
ev.add_slice('night_shift', X_test['hour'] < 6)
ev.add_slice('sensor_b_night',
             (X_test['sensor_type'] == 'B') & (X_test['hour'] < 6))

# Auto-discover: things you don't suspect
ev.discover_slices(method='tree', max_depth=2, min_support=0.05, significance=1.0)

report = ev.evaluate()

# ─── 5. Results ───
print("─── Global Metrics ───")
for k, v in report.global_metrics.items():
    print(f"  {k}: {v:.3f}")

print("\n─── Worst Slices (by F1) ───")
worst = report.worst_slices(n=5, metric='f1')
print(worst.to_string(index=False))

print("\n─── Full Report ───")
df = report.to_dataframe()
print(df[['slice_name', 'source', 'n_samples', 'f1_value', 'f1_delta', 'f1_p_value']].to_string(index=False))

print("\n" + "=" * 60)
print("The global F1 hid these failures. sliceval surfaced them.")
print("=" * 60)