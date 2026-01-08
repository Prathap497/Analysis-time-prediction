"""Runtime prediction modeling and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class IntervalResult:
    prediction: float
    lower: float
    upper: float


class RuntimeModel:
    """Linear regression model for log-runtime prediction."""

    def __init__(self) -> None:
        self._model = LinearRegression()
        self._residuals: np.ndarray | None = None

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._model.fit(features, target)
        predicted = self._model.predict(features)
        self._residuals = target.to_numpy() - predicted

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._model.predict(features)

    def predict_with_interval(
        self,
        features: pd.DataFrame,
        lower_q: float = 0.1,
        upper_q: float = 0.9,
    ) -> list[IntervalResult]:
        if self._residuals is None:
            raise ValueError("Model must be trained before computing intervals.")
        predictions = self.predict(features)
        lower_offset = np.quantile(self._residuals, lower_q)
        upper_offset = np.quantile(self._residuals, upper_q)
        results = []
        for pred in predictions:
            results.append(
                IntervalResult(
                    prediction=float(np.expm1(pred)),
                    lower=float(np.expm1(pred + lower_offset)),
                    upper=float(np.expm1(pred + upper_offset)),
                )
            )
        return results
