"""Run the reproducible train/test pipeline: python train_model.py."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import CATEGORICAL_FEATURES, clean_data, process_data
from ml.model import (
    compute_model_metrics, inference, performance_on_categorical_slice,
    save_model, train_model,
)

ROOT = Path(__file__).resolve().parent


def run_pipeline(data_path=ROOT / "data/census.csv", output_dir=ROOT / "model"):
    """Fit only on training rows and write artifacts and held-out metrics."""
    data_path, output_dir = Path(data_path), Path(output_dir)
    data = clean_data(pd.read_csv(data_path))
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["salary"],
    )
    X_train, y_train, encoder, lb = process_data(train, label="salary")
    X_test, y_test, _, _ = process_data(
        test, label="salary", training=False, encoder=encoder, lb=lb,
    )
    model = train_model(X_train, y_train)
    precision, recall, f1 = compute_model_metrics(y_test, inference(model, X_test))
    for name, artifact in [("model", model), ("encoder", encoder), ("lb", lb)]:
        save_model(artifact, output_dir / f"{name}.pkl")
        print(f"Saved {output_dir / (name + '.pkl')}")
    slices = []
    for feature in CATEGORICAL_FEATURES:
        slices.extend(performance_on_categorical_slice(
            test, model, encoder, lb, feature,
        ))
    metrics = {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "train_rows": len(train), "test_rows": len(test),
        "positive_test_rows": int(y_test.sum()), "random_state": 42,
        "data_sha256": hashlib.sha256(
            data_path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "slices": slices,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
    )
    lines = ["Held-out test slices; positive class >50K; undefined metrics = 0."]
    for row in slices:
        lines.append(
            f"{row['feature']}: {row['value']}, Count: {row['count']}, "
            f"Positive count: {row['positive_count']}\n"
            f"Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | "
            f"F1: {row['f1']:.4f}"
        )
    (output_dir.parent / "slice_output.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Training rows: {len(train)} | Test rows: {len(test)}")
    print(f"Evaluated {len(slices)} categorical slices")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/census.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model")
    args = parser.parse_args()
    run_pipeline(args.data, args.output_dir)
