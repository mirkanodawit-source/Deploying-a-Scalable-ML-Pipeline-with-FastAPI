"""Load the trusted artifact bundle once during application startup."""

import pandas as pd

from ml.data import process_data
from ml.model import inference, load_model


class ModelService:
    def __init__(self, model_dir):
        self.model = load_model(model_dir / "model.pkl")
        self.encoder = load_model(model_dir / "encoder.pkl")
        self.lb = load_model(model_dir / "lb.pkl")
        if self.model.n_features_in_ != 6 + sum(
            len(values) for values in self.encoder.categories_
        ):
            raise ValueError("Model and encoder have incompatible feature counts")
        if list(self.lb.classes_) != ["<=50K", ">50K"]:
            raise ValueError("Unexpected label encoder classes")

    def predict(self, record):
        frame = pd.DataFrame([record.model_dump(by_alias=True)])
        X, _, _, _ = process_data(frame, training=False, encoder=self.encoder)
        prediction = inference(self.model, X)
        return str(self.lb.inverse_transform(prediction)[0])
