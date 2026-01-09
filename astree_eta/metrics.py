from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from astree_eta.features import prepare_training_frame
from astree_eta.model import train_model


@dataclass
class MetricsResult:
    mae_hours: float
    mdape: float
    coverage_p10_p90: float


def evaluate(history_df: pd.DataFrame, output_path: Path) -> MetricsResult:
    history_df = prepare_training_frame(history_df)
    history_df = history_df.sort_values("start_time")

    tscv = TimeSeriesSplit(n_splits=3)
    errors = []
    ape = []
    coverage = []

    for train_index, test_index in tscv.split(history_df):
        train_df = history_df.iloc[train_index]
        test_df = history_df.iloc[test_index]
        model = train_model(train_df)
        pred_log = model.predict_log(test_df[model.feature_spec.numeric_features + model.feature_spec.categorical_features])
        preds = np.expm1(pred_log)
        actuals = test_df["runtime_sec"].values
        errors.extend(np.abs(preds - actuals) / 3600)
        ape.extend(np.abs(preds - actuals) / np.maximum(actuals, 1))
        interval_logs = model.predict_intervals(
            test_df[model.feature_spec.numeric_features + model.feature_spec.categorical_features],
            (0.1, 0.9),
        )
        lower = np.expm1(interval_logs[0.1])
        upper = np.expm1(interval_logs[0.9])
        coverage.extend(((actuals >= lower) & (actuals <= upper)).astype(float))

    result = MetricsResult(
        mae_hours=float(np.mean(errors)) if errors else 0.0,
        mdape=float(np.median(ape)) if ape else 0.0,
        coverage_p10_p90=float(np.mean(coverage)) if coverage else 0.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, indent=2)
    return result
