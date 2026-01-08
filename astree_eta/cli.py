from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from astree_eta.features import prepare_inference_frame, prepare_training_frame
from astree_eta.metrics import evaluate as evaluate_metrics
from astree_eta.model import load_model, save_model, train_model
from astree_eta.notifier import EmailConfig, EmailNotifier, render_notification
from astree_eta.rules import should_notify, update_state_from_events
from astree_eta.schemas import PredictionRecord
from astree_eta.splunk_client import (
    SplunkClient,
    SplunkConfig,
    build_completed_runs_query,
    build_running_runs_query,
)
from astree_eta.store import NotificationStore

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _splunk_config() -> SplunkConfig:
    return SplunkConfig(
        base_url=os.environ["SPLUNK_BASE_URL"],
        token=os.environ["SPLUNK_TOKEN"],
        verify_tls=os.environ.get("SPLUNK_VERIFY_TLS", "true").lower() == "true",
        app=os.environ.get("SPLUNK_APP", ""),
        index=os.environ.get("SPLUNK_INDEX", ""),
        sourcetype=os.environ.get("SPLUNK_SOURCETYPE", ""),
        host_filter=os.environ.get("SPLUNK_HOST_FILTER"),
    )


def _splunk_link(run_id: str, config: SplunkConfig) -> str:
    return f"{config.base_url}/app/{config.app}/search?q=search%20run_id%3D{run_id}"


def extract_history(args: argparse.Namespace) -> None:
    config = _splunk_config()
    client = SplunkClient(config)
    search = build_completed_runs_query(config)
    earliest = f"-{args.months}mon@mon"
    latest = "now"
    rows = list(client.export_search(search, earliest=earliest, latest=latest))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    logger.info("history extracted", extra={"rows": len(df), "out": args.out})


def train(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    df = df.copy()
    df["runtime_sec"] = (pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])) / pd.Timedelta(seconds=1)
    df = prepare_training_frame(df)
    model = train_model(df)
    save_model(model, Path(args.out))
    logger.info("model trained", extra={"rows": len(df), "out": args.out})


def predict_live(args: argparse.Namespace) -> None:
    config = _splunk_config()
    client = SplunkClient(config)
    search = build_running_runs_query(config)
    earliest = f"-{args.interval_min}m"
    latest = "now"
    rows = list(client.export_search(search, earliest=earliest, latest=latest))
    df = pd.DataFrame(rows)
    df = prepare_inference_frame(df)
    if df.empty:
        df.to_parquet(args.out, index=False)
        logger.info("no running runs", extra={"out": args.out})
        return

    df["start_time"] = pd.to_datetime(df["start_time"])
    now = datetime.utcnow()
    df["elapsed_sec"] = (now - df["start_time"]).dt.total_seconds()

    model = load_model(Path(args.model_dir))
    features = df[model.feature_spec.numeric_features + model.feature_spec.categorical_features]
    interval_logs = model.predict_intervals(features, (0.1, 0.5, 0.9))
    totals = {q: np.expm1(interval_logs[q]) for q in interval_logs}

    df["total_p10_sec"] = totals[0.1]
    df["total_p50_sec"] = totals[0.5]
    df["total_p90_sec"] = totals[0.9]
    df["remaining_sec"] = np.maximum(df["total_p50_sec"] - df["elapsed_sec"], 0)
    df["progress"] = np.minimum(df["elapsed_sec"] / np.maximum(df["total_p50_sec"], 1), 0.999)
    df["splunk_url"] = df["run_id"].apply(lambda run_id: _splunk_link(run_id, config))
    df.to_parquet(args.out, index=False)
    logger.info("live predictions", extra={"rows": len(df), "out": args.out})


def notify(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    if df.empty:
        logger.info("no predictions to notify")
        return

    config = _splunk_config()
    if not config.host_filter:
        logger.warning("host filter not set; refusing to send notifications")
        return
    store = NotificationStore(args.db_path)
    email_config = EmailConfig(
        host=os.environ["SMTP_HOST"],
        port=int(os.environ.get("SMTP_PORT", "25")),
        user=os.environ.get("SMTP_USER", ""),
        password=os.environ.get("SMTP_PASS", ""),
        sender=os.environ["EMAIL_FROM"],
        recipients=os.environ["EMAIL_TO"].split(","),
    )
    notifier = EmailNotifier(email_config)

    now = datetime.utcnow()
    for _, row in df.iterrows():
        prediction = PredictionRecord(**row.to_dict())
        state = store.get_state(prediction.run_id)
        if not store.can_send(state, now, throttle_minutes=args.throttle_min):
            continue
        events = should_notify(
            prediction,
            state,
            now,
            throttle_minutes=args.throttle_min,
            allowed_host=config.host_filter,
        )
        if not events:
            continue
        event = events[0]
        body = render_notification(event.payload)
        notifier.send(event.subject, body)
        updated = update_state_from_events(state, [event], prediction.status, now)
        store.update_state(updated.run_id, updated.last_status, updated.last_email_ts, updated.milestones)


def evaluate(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.input)
    df = df.copy()
    df["runtime_sec"] = (pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])) / pd.Timedelta(seconds=1)
    evaluate_metrics(df, Path(args.out))
    logger.info("metrics written", extra={"out": args.out})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astrée ETA + notifications")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-history")
    extract.add_argument("--months", type=int, required=True)
    extract.add_argument("--out", required=True)
    extract.set_defaults(func=extract_history)

    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--in", dest="input", required=True)
    train_cmd.add_argument("--out", required=True)
    train_cmd.set_defaults(func=train)

    predict = sub.add_parser("predict-live")
    predict.add_argument("--interval-min", type=int, required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--model-dir", default="models")
    predict.set_defaults(func=predict_live)

    notify_cmd = sub.add_parser("notify")
    notify_cmd.add_argument("--in", dest="input", required=True)
    notify_cmd.add_argument("--db-path", default="data/notifications.db")
    notify_cmd.add_argument("--throttle-min", type=int, default=15)
    notify_cmd.set_defaults(func=notify)

    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument("--in", dest="input", required=True)
    evaluate_cmd.add_argument("--out", required=True)
    evaluate_cmd.set_defaults(func=evaluate)

    return parser


def main() -> None:
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
