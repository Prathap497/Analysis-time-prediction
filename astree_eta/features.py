from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


NUMERIC_FEATURES = ["loc", "num_files", "cpu_avg"]
CATEGORICAL_FEATURES = ["project_id", "astree_version", "config_profile", "host"]


@dataclass
class FeatureSpec:
    numeric_features: List[str]
    categorical_features: List[str]


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    clipped = np.where(values < 0, 0, values)
    return np.log1p(clipped)


def build_preprocessor(
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
) -> Tuple[ColumnTransformer, FeatureSpec]:
    numeric = numeric_features or NUMERIC_FEATURES
    categorical = categorical_features or CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("log1p", FunctionTransformer(_safe_log1p, feature_names_out="one-to-one")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )
    return preprocessor, FeatureSpec(numeric_features=numeric, categorical_features=categorical)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "loc" not in df.columns:
        df["loc"] = np.nan
    if "kloc" not in df.columns:
        df["kloc"] = np.nan
    if "num_files" not in df.columns:
        df["num_files"] = np.nan
    if "num_tu" not in df.columns:
        df["num_tu"] = np.nan
    if "cpu_avg" not in df.columns:
        df["cpu_avg"] = np.nan
    if "server_load_avg" not in df.columns:
        df["server_load_avg"] = np.nan

    df["loc"] = df["loc"].where(pd.notnull(df["loc"]), df["kloc"] * 1000)
    df["num_files"] = df["num_files"].where(pd.notnull(df["num_files"]), df["num_tu"])
    df["cpu_avg"] = df["cpu_avg"].where(pd.notnull(df["cpu_avg"]), df["server_load_avg"])
    return df


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_features(df)
    df = df.copy()
    df = df[df["status"] == "COMPLETED"]
    df = df[df["runtime_sec"] > 0]
    lower, upper = df["runtime_sec"].quantile([0.01, 0.99])
    df = df[(df["runtime_sec"] >= lower) & (df["runtime_sec"] <= upper)]
    df = df.dropna(subset=["run_id", "project_id", "start_time", "end_time", "status"])
    return df


def prepare_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_features(df)
    df = df.copy()
    df = df[df["status"].isin(["RUNNING", "COMPLETED", "FAILED", "ABORTED"])]
    return df
