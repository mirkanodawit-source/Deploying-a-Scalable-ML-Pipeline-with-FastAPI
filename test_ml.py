"""Behavioral tests for preprocessing, training, persistence, and slice metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import CATEGORICAL_FEATURES, clean_data, process_data
from ml.model import (
    compute_model_metrics, inference, load_model,
    performance_on_categorical_slice, save_model, train_model,
)

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def bundle():
    data = clean_data(pd.read_csv(ROOT / "data/census.csv"))
    _, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["salary"],
    )
    return (
        test, load_model(ROOT / "model/model.pkl"),
        load_model(ROOT / "model/encoder.pkl"), load_model(ROOT / "model/lb.pkl"),
    )


def test_metrics_known_values():
    precision, recall, f1 = compute_model_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (precision, recall, f1) == (0.5, 0.5, 0.5)
    assert compute_model_metrics([0, 0], [0, 0]) == (0, 0, 0)


def test_training_learns_separable_data():
    X = np.array([[0], [0], [0], [1], [1], [1]] * 10)
    y = np.array([0, 0, 0, 1, 1, 1] * 10)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    np.testing.assert_array_equal(inference(model, X), y)


def test_model_round_trip_preserves_predictions(bundle, tmp_path):
    data, model, encoder, lb = bundle
    X, _, _, _ = process_data(data, training=False, encoder=encoder)
    for name, artifact in [("model", model), ("encoder", encoder), ("lb", lb)]:
        save_model(artifact, tmp_path / f"{name}.pkl")
    restored = load_model(tmp_path / "model.pkl")
    np.testing.assert_array_equal(inference(restored, X), inference(model, X))
    restored_encoder = load_model(tmp_path / "encoder.pkl")
    X_again, _, _, _ = process_data(data, training=False, encoder=restored_encoder)
    np.testing.assert_array_equal(X_again, X)


def test_inference_never_refits_and_handles_unknown_categories(bundle):
    data, _, encoder, lb = bundle
    before = [values.copy() for values in encoder.categories_]
    row = data.iloc[:1].copy()
    row["workclass"] = "New-category"
    X, y, returned_encoder, returned_lb = process_data(
        row, label="salary", training=False, encoder=encoder, lb=lb,
    )
    assert X.shape == (1, 6 + sum(len(values) for values in before))
    assert y.shape == (1,)
    assert returned_encoder is encoder and returned_lb is lb
    for old, new in zip(before, encoder.categories_):
        np.testing.assert_array_equal(old, new)


def test_column_order_does_not_change_features(bundle):
    data, _, encoder, _ = bundle
    X, _, _, _ = process_data(data, training=False, encoder=encoder)
    reverse, _, _, _ = process_data(
        data[data.columns[::-1]], training=False, encoder=encoder,
    )
    np.testing.assert_array_equal(X, reverse)


def test_all_slice_counts_and_metrics_match_manual_calculation(bundle):
    data, model, encoder, lb = bundle
    for feature in CATEGORICAL_FEATURES:
        slices = performance_on_categorical_slice(data, model, encoder, lb, feature)
        assert sum(row["count"] for row in slices) == len(data)
        assert {row["value"] for row in slices} == set(data[feature].unique())
        row = slices[0]
        subset = data[data[feature] == row["value"]]
        X, y, _, _ = process_data(
            subset, label="salary", training=False, encoder=encoder, lb=lb,
        )
        expected = compute_model_metrics(y, inference(model, X))
        assert (row["precision"], row["recall"], row["f1"]) == expected


def test_missing_encoder_rejected(bundle):
    with pytest.raises(ValueError, match="fitted encoder"):
        process_data(bundle[0], training=False)


def test_missing_feature_rejected(bundle):
    with pytest.raises(ValueError, match="Missing features"):
        process_data(bundle[0].drop(columns="age"))


def test_nonfinite_numeric_rejected(bundle):
    data = bundle[0].iloc[:2].copy()
    data["age"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        process_data(data)


def test_unknown_label_rejected(bundle):
    data = bundle[0].iloc[:2].copy()
    data["salary"] = "invalid"
    with pytest.raises(ValueError, match="Labels"):
        process_data(data, label="salary")
