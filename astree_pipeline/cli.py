"""Command-line interface for the pipeline."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from astree_pipeline.data_pipeline import PipelineConfig, prepare_features, preprocess_runs
from astree_pipeline.live import AnomalyDetector, LiveEstimator
from astree_pipeline.modeling import RuntimeModel


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def cmd_train(args: argparse.Namespace) -> None:
    data = load_csv(args.training_data)
    processed = preprocess_runs(data, PipelineConfig())
    features, target = prepare_features(processed)

    model = RuntimeModel()
    model.train(features, target)

    with open(args.model_output, "w", encoding="utf-8") as handle:
        payload = {
            "coefficients": model._model.coef_.tolist(),
            "intercept": model._model.intercept_.item(),
            "columns": features.columns.tolist(),
            "residuals": model._residuals.tolist() if model._residuals is not None else [],
        }
        json.dump(payload, handle, indent=2)


def cmd_predict(args: argparse.Namespace) -> None:
    model_payload = json.loads(open(args.model_path, "r", encoding="utf-8").read())
    features = load_csv(args.features)

    model = RuntimeModel()
    model._model.coef_ = pd.Series(model_payload["coefficients"]).to_numpy()
    model._model.intercept_ = model_payload["intercept"]
    model._residuals = pd.Series(model_payload["residuals"]).to_numpy()

    results = model.predict_with_interval(features)
    for result in results:
        print(
            json.dumps(
                {
                    "prediction": result.prediction,
                    "lower": result.lower,
                    "upper": result.upper,
                }
            )
        )


def cmd_live(args: argparse.Namespace) -> None:
    estimator = LiveEstimator()
    estimate = estimator.estimate(args.predicted_total, args.elapsed)

    detector = AnomalyDetector(args.p90_threshold)
    anomaly = detector.check_elapsed(args.elapsed)

    payload = {
        "predicted_total_sec": estimate.predicted_total_sec,
        "remaining_sec": estimate.remaining_sec,
        "completion_ratio": estimate.completion_ratio,
        "anomaly": anomaly.is_anomaly,
        "anomaly_reason": anomaly.reason,
    }
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astrée runtime prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a baseline regression model")
    train_parser.add_argument("training_data", help="CSV file of historical runs")
    train_parser.add_argument("model_output", help="Path to save trained model JSON")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Predict runtime with intervals")
    predict_parser.add_argument("model_path", help="Path to saved model JSON")
    predict_parser.add_argument("features", help="CSV file with prepared features")
    predict_parser.set_defaults(func=cmd_predict)

    live_parser = subparsers.add_parser("live", help="Estimate remaining runtime")
    live_parser.add_argument("predicted_total", type=float, help="Predicted total runtime in seconds")
    live_parser.add_argument("elapsed", type=float, help="Elapsed runtime in seconds")
    live_parser.add_argument("p90_threshold", type=float, help="P90 runtime threshold in seconds")
    live_parser.set_defaults(func=cmd_live)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
