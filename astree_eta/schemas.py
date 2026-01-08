from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RunRecord(BaseModel):
    run_id: str
    project_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str
    astree_version: str
    config_profile: str
    host: str
    loc: Optional[float] = None
    kloc: Optional[float] = None
    num_files: Optional[float] = None
    num_tu: Optional[float] = None
    cpu_avg: Optional[float] = None
    server_load_avg: Optional[float] = None
    last_log_time: Optional[datetime] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        allowed = {"RUNNING", "COMPLETED", "FAILED", "ABORTED"}
        if value not in allowed:
            raise ValueError(f"Invalid status: {value}")
        return value

    def effective_loc(self) -> Optional[float]:
        if self.kloc is not None:
            return self.kloc * 1000
        return self.loc

    def effective_num_files(self) -> Optional[float]:
        return self.num_files or self.num_tu


class PredictionRecord(BaseModel):
    run_id: str
    project_id: str
    host: str
    status: str
    start_time: datetime
    elapsed_sec: float
    total_p10_sec: float
    total_p50_sec: float
    total_p90_sec: float
    remaining_sec: float
    progress: float
    astree_version: str
    config_profile: str
    splunk_url: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class NotificationRecord(BaseModel):
    run_id: str
    event_type: str
    sent_at: datetime
    recipient: str
