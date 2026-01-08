"""Data extraction and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from astree_pipeline.splunk_client import SplunkClient


@dataclass
class PipelineConfig:
    outlier_quantile: float = 0.99


def extract_runs(client: SplunkClient, status: str) -> pd.DataFrame:
    """Extract runs from Splunk into a DataFrame."""
    records = list(client.fetch_runs(status))
    return pd.DataFrame.from_records(records)


def preprocess_runs(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Drop failed/aborted runs, remove outliers, and normalize fields."""
    if df.empty:
        return df.copy()

    cleaned = df.copy()
    cleaned = cleaned[cleaned.get("status", "completed") == "completed"]
    cleaned = cleaned.dropna(subset=["total_runtime_sec", "loc", "modules"])

    cutoff = cleaned["total_runtime_sec"].quantile(config.outlier_quantile)
    cleaned = cleaned[cleaned["total_runtime_sec"] <= cutoff]

    cleaned["log_loc"] = np.log1p(cleaned["loc"].astype(float))
    cleaned["log_runtime"] = np.log1p(cleaned["total_runtime_sec"].astype(float))

    return cleaned


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for modeling."""
    features = df[["log_loc", "modules", "astree_version", "config_profile"]].copy()
    features = pd.get_dummies(features, columns=["astree_version", "config_profile"], drop_first=True)
    target = df["log_runtime"].copy()
    return features, target
