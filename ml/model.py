"""Train, persist, evaluate, and slice the Census classifier."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from ml.data import CATEGORICAL_FEATURES, clean_data, process_data


def train_model(X_train, y_train):
    """Fit a deterministic forest with bounded artifact size."""
    model = RandomForestClassifier(
        n_estimators=150, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=1,
    )
    return model.fit(X_train, y_train)


def inference(model, X):
    """Predict binary income labels for encoded rows."""
    return model.predict(X)


def compute_model_metrics(y, preds):
    """Return precision, recall, F1 for >50K; undefined metrics are zero."""
    return (
        precision_score(y, preds, zero_division=0),
        recall_score(y, preds, zero_division=0),
        f1_score(y, preds, zero_division=0),
    )


def save_model(model, path):
    """Persist a model or encoder. Only load trusted project artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path):
    """Load a trusted, locally produced joblib artifact."""
    return joblib.load(path)


def performance_on_categorical_slice(
    data, model, encoder, lb, feature, label="salary",
):
    """Evaluate every observed value of one feature on held-out rows."""
    if feature not in CATEGORICAL_FEATURES:
        raise ValueError(f"Not a categorical feature: {feature}")
    data = clean_data(data)
    X, y, _, _ = process_data(
        data, label=label, training=False, encoder=encoder, lb=lb,
    )
    predictions = inference(model, X)
    results = []
    values = data[feature].fillna("?")
    for value in sorted(values.unique()):
        mask = (values == value).to_numpy()
        precision, recall, f1 = compute_model_metrics(y[mask], predictions[mask])
        results.append({
            "feature": feature, "value": value, "count": int(mask.sum()),
            "positive_count": int(np.sum(y[mask])),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1),
        })
    return results
