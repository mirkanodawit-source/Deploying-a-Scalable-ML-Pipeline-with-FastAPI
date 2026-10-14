"""Shared Census preprocessing; fit encoders on training rows only."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
NUMERIC_FEATURES = [
    "age", "fnlgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]
FEATURES = [
    "age", "workclass", "fnlgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]


def clean_data(data):
    """Strip surrounding whitespace; keep '?' as an explicit unknown category."""
    result = data.copy()
    result.columns = result.columns.str.strip()
    for column in result.select_dtypes(include=["object", "string"]):
        result[column] = result[column].str.strip()
    return result


def process_data(
    X, categorical_features=None, label=None, training=True,
    encoder=None, lb=None,
):
    """Return encoded features, labels, and fitted reusable encoders.

    The fixed feature order prevents request/dataset column order from changing
    predictions. Unseen categories are encoded as all-zero indicator groups.
    """
    data = clean_data(X)
    categorical_features = (
        CATEGORICAL_FEATURES if categorical_features is None
        else list(categorical_features)
    )
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    if categorical_features != CATEGORICAL_FEATURES:
        raise ValueError("Use the canonical Census categorical feature order")
    numeric = data[NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite")
    categorical = data[categorical_features].fillna("?")
    if training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(categorical)
    else:
        if encoder is None:
            raise ValueError("A fitted encoder is required for inference")
        encoded = encoder.transform(categorical)
    y = np.array([])
    if label is not None:
        labels = data[label].str.rstrip(".")
        if not labels.isin(["<=50K", ">50K"]).all():
            raise ValueError("Labels must be <=50K or >50K")
        if training:
            lb = LabelBinarizer()
            y = lb.fit_transform(labels).ravel()
            if len(lb.classes_) != 2:
                raise ValueError("Training data must contain both income classes")
        else:
            if lb is None:
                raise ValueError("A fitted label binarizer is required")
            y = lb.transform(labels).ravel()
    features = np.concatenate([numeric.to_numpy(), encoded], axis=1)
    return features, y, encoder, lb
