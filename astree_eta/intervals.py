from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class ResidualIntervalModel:
    residual_quantiles: Dict[float, float]

    def predict(self, base_pred: np.ndarray, quantiles: Iterable[float]) -> Dict[float, np.ndarray]:
        outputs: Dict[float, np.ndarray] = {}
        for q in quantiles:
            offset = self.residual_quantiles[q]
            outputs[q] = base_pred + offset
        return outputs


def train_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    min_samples: int = 30,
) -> Dict[str, object]:
    if len(y) < min_samples:
        return {"type": "residual"}

    models: Dict[float, GradientBoostingRegressor] = {}
    for q in quantiles:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=42)
        model.fit(X, y)
        models[q] = model
    return {"type": "quantile", "models": models}


def build_residual_interval(base_pred: np.ndarray, y: np.ndarray, quantiles: Iterable[float]) -> ResidualIntervalModel:
    residuals = y - base_pred
    quantile_map = {float(q): float(np.quantile(residuals, q)) for q in quantiles}
    return ResidualIntervalModel(residual_quantiles=quantile_map)
