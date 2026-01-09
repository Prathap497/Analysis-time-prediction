from datetime import datetime, timedelta

from astree_eta.store import NotificationStore


def test_notification_store_roundtrip(tmp_path):
    db_path = tmp_path / "notify.db"
    store = NotificationStore(str(db_path))
    state = store.get_state("run-1")
    assert state.run_id == "run-1"
    now = datetime.utcnow()
    store.update_state("run-1", "RUNNING", now, {"50": 1, "75": 0, "90": 0})
    fetched = store.get_state("run-1")
    assert fetched.last_status == "RUNNING"
    assert fetched.milestones["50"] == 1


def test_throttle(tmp_path):
    db_path = tmp_path / "notify.db"
    store = NotificationStore(str(db_path))
    now = datetime.utcnow()
    store.update_state("run-2", "RUNNING", now, {"50": 0, "75": 0, "90": 0})
    state = store.get_state("run-2")
    assert store.can_send(state, now + timedelta(minutes=16), 15)
    assert not store.can_send(state, now + timedelta(minutes=5), 15)


def test_store_creates_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "notify.db"
    NotificationStore(str(db_path))
    assert db_path.exists()
