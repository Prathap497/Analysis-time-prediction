from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional


@dataclass
class NotificationState:
    run_id: str
    last_status: Optional[str]
    last_email_ts: Optional[datetime]
    milestones: Dict[str, int]


class NotificationStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notification_state (
                    run_id TEXT PRIMARY KEY,
                    last_status TEXT,
                    last_email_ts TEXT,
                    milestone_50 INTEGER DEFAULT 0,
                    milestone_75 INTEGER DEFAULT 0,
                    milestone_90 INTEGER DEFAULT 0
                )
                """
            )

    def get_state(self, run_id: str) -> NotificationState:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT run_id, last_status, last_email_ts, milestone_50, milestone_75, milestone_90 "
                "FROM notification_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return NotificationState(run_id=run_id, last_status=None, last_email_ts=None, milestones={"50": 0, "75": 0, "90": 0})
        last_email_ts = datetime.fromisoformat(row[2]) if row[2] else None
        milestones = {"50": row[3], "75": row[4], "90": row[5]}
        return NotificationState(run_id=row[0], last_status=row[1], last_email_ts=last_email_ts, milestones=milestones)

    def update_state(
        self,
        run_id: str,
        last_status: Optional[str],
        last_email_ts: Optional[datetime],
        milestones: Dict[str, int],
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO notification_state (run_id, last_status, last_email_ts, milestone_50, milestone_75, milestone_90)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    last_status=excluded.last_status,
                    last_email_ts=excluded.last_email_ts,
                    milestone_50=excluded.milestone_50,
                    milestone_75=excluded.milestone_75,
                    milestone_90=excluded.milestone_90
                """,
                (
                    run_id,
                    last_status,
                    last_email_ts.isoformat() if last_email_ts else None,
                    milestones.get("50", 0),
                    milestones.get("75", 0),
                    milestones.get("90", 0),
                ),
            )

    def can_send(self, state: NotificationState, now: datetime, throttle_minutes: int) -> bool:
        if not state.last_email_ts:
            return True
        return now - state.last_email_ts >= timedelta(minutes=throttle_minutes)
