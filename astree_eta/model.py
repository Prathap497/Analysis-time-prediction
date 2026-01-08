from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from astree_eta.features import FeatureSpec, build_preprocessor
from astree_eta.intervals import build_residual_interval, train_quantile_models


@dataclass
class ModelBundle:
    feature_spec: FeatureSpec
    preprocessor: object
    base_model: GradientBoostingRegressor
    interval_type: str
    interval_payload: object

    def predict_log(self, frame) -> np.ndarray:
        X = self.preprocessor.transform(frame)
        return self.base_model.predict(X)

    def predict_intervals(self, frame, quantiles: Iterable[float]) -> Dict[float, np.ndarray]:
        X = self.preprocessor.transform(frame)
        if self.interval_type == "quantile":
            models = self.interval_payload
            return {q: models[q].predict(X) for q in quantiles}
        base_pred = self.base_model.predict(X)
        return self.interval_payload.predict(base_pred, quantiles)


def train_model(train_df, random_state: int = 42) -> ModelBundle:
    preprocessor, spec = build_preprocessor()
    X = train_df[spec.numeric_features + spec.categorical_features]
    y = np.log1p(train_df["runtime_sec"].values)

    base_model = GradientBoostingRegressor(random_state=random_state)
    transformed = preprocessor.fit_transform(X)
    base_model.fit(transformed, y)
    interval_training = train_quantile_models(transformed, y)

    if interval_training["type"] == "quantile":
        interval_payload = interval_training["models"]
        interval_type = "quantile"
    else:
        base_pred = base_model.predict(transformed)
        interval_payload = build_residual_interval(base_pred, y, (0.1, 0.5, 0.9))
        interval_type = "residual"

    return ModelBundle(
        feature_spec=spec,
        preprocessor=preprocessor,
        base_model=base_model,
        interval_type=interval_type,
        interval_payload=interval_payload,
    )


def save_model(bundle: ModelBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "model.joblib")


def load_model(model_dir: Path) -> ModelBundle:
    return joblib.load(model_dir / "model.joblib")
