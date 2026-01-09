from datetime import datetime

from astree_eta.rules import should_notify
from astree_eta.schemas import PredictionRecord
from astree_eta.store import NotificationState


def test_should_notify_start_and_milestone():
    prediction = PredictionRecord(
        run_id="run-1",
        project_id="proj",
        host="linux-aaas",
        status="RUNNING",
        start_time=datetime.utcnow(),
        elapsed_sec=3600,
        total_p10_sec=4000,
        total_p50_sec=5000,
        total_p90_sec=6000,
        remaining_sec=1000,
        progress=0.6,
        astree_version="1.0",
        config_profile="default",
    )
    state = NotificationState(run_id="run-1", last_status=None, last_email_ts=None, milestones={"50": 0, "75": 0, "90": 0})
    events = should_notify(prediction, state, datetime.utcnow(), 15, allowed_host="linux-aaas")
    assert events
    assert events[-1].event_type == "START"


def test_should_notify_delay():
    prediction = PredictionRecord(
        run_id="run-2",
        project_id="proj",
        host="linux-aaas",
        status="RUNNING",
        start_time=datetime.utcnow(),
        elapsed_sec=7000,
        total_p10_sec=4000,
        total_p50_sec=5000,
        total_p90_sec=6000,
        remaining_sec=0,
        progress=0.99,
        astree_version="1.0",
        config_profile="default",
    )
    state = NotificationState(run_id="run-2", last_status="RUNNING", last_email_ts=None, milestones={"50": 1, "75": 1, "90": 1})
    events = should_notify(prediction, state, datetime.utcnow(), 15, allowed_host="linux-aaas")
    assert events
    assert events[0].event_type == "DELAY"
