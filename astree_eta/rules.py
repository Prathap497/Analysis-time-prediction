from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from astree_eta.schemas import PredictionRecord
from astree_eta.store import NotificationState


@dataclass
class NotificationEvent:
    run_id: str
    event_type: str
    subject: str
    payload: dict


def _milestone_crossed(progress: float, threshold: float) -> bool:
    return progress >= threshold


def should_notify(
    prediction: PredictionRecord,
    state: NotificationState,
    now: datetime,
    throttle_minutes: int,
    allowed_host: Optional[str] = None,
) -> List[NotificationEvent]:
    events: List[NotificationEvent] = []
    if allowed_host and prediction.host != allowed_host:
        return events

    if state.last_status != prediction.status and prediction.status in {"COMPLETED", "FAILED", "ABORTED"}:
        events.append(_event("DONE", prediction))

    if prediction.status in {"RUNNING", "PROCESSING"} and prediction.elapsed_sec > prediction.total_p90_sec:
        events.append(_event("DELAY", prediction))

    milestones = {"90": 0.9, "75": 0.75, "50": 0.5}
    for key, threshold in milestones.items():
        if (
            prediction.status in {"RUNNING", "PROCESSING"}
            and state.milestones.get(key, 0) == 0
            and _milestone_crossed(prediction.progress, threshold)
        ):
            events.append(_event(f"MILESTONE_{key}", prediction))

    if not state.last_status and prediction.status in {"RUNNING", "PROCESSING"}:
        events.append(_event("START", prediction))

    if prediction.anomaly_reasons:
        events.append(_event("ANOMALY", prediction))

    return events


def _event(event_type: str, prediction: PredictionRecord) -> NotificationEvent:
    subject = f"Astrée ETA Update [{event_type}] run {prediction.run_id}"
    payload = {
        "run_id": prediction.run_id,
        "build_number": prediction.build_number,
        "analysis_name": prediction.analysis_name,
        "project_id": prediction.project_id,
        "host": prediction.host,
        "astree_version": prediction.astree_version,
        "config_profile": prediction.config_profile,
        "status": prediction.status,
        "elapsed_hours": prediction.elapsed_sec / 3600,
        "total_p10_hours": prediction.total_p10_sec / 3600,
        "total_p50_hours": prediction.total_p50_sec / 3600,
        "total_p90_hours": prediction.total_p90_sec / 3600,
        "remaining_hours": prediction.remaining_sec / 3600,
        "progress_pct": prediction.progress * 100,
        "queue_wait_hours": (prediction.queue_wait_sec or 0) / 3600 if prediction.queue_wait_sec is not None else None,
        "anomaly_reasons": prediction.anomaly_reasons,
        "splunk_url": prediction.splunk_url,
    }
    return NotificationEvent(run_id=prediction.run_id, event_type=event_type, subject=subject, payload=payload)


def update_state_from_events(state: NotificationState, events: Iterable[NotificationEvent], status: str, sent_at: datetime) -> NotificationState:
    milestones = dict(state.milestones)
    for event in events:
        if event.event_type.startswith("MILESTONE_"):
            milestone_key = event.event_type.split("_")[1]
            milestones[milestone_key] = 1
    return NotificationState(run_id=state.run_id, last_status=status, last_email_ts=sent_at, milestones=milestones)
