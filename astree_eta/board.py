from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd

from astree_eta.schemas import AnalysisStatus, ServerSnapshot


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_server_snapshot(rows: Iterable[dict]) -> Optional[ServerSnapshot]:
    rows_list = list(rows)
    if not rows_list:
        return None
    row = rows_list[0]
    timestamp = _parse_datetime(row.get("timestamp")) or _parse_datetime(row.get("_time"))
    if not timestamp:
        timestamp = datetime.utcnow()
    return ServerSnapshot(
        processing_count=_parse_int(row.get("processing_count")) or 0,
        queued_count=_parse_int(row.get("queued_count")) or 0,
        total_mem_used_gb=_parse_float(row.get("total_mem_used_gb")) or 0.0,
        free_mem_gb=_parse_float(row.get("free_mem_gb")) or 0.0,
        timestamp=timestamp,
    )


def parse_analysis_statuses(rows: Iterable[dict], status: str) -> List[AnalysisStatus]:
    statuses: List[AnalysisStatus] = []
    for row in rows:
        statuses.append(
            AnalysisStatus(
                build_number=str(row.get("build_number") or "").strip() or None,
                analysis_name=str(row.get("analysis_name") or "").strip() or "unknown",
                status=status,
                used_memory_gb=_parse_float(row.get("used_memory_gb")),
                duration_hours=_parse_float(row.get("duration_hours")),
                queued_timestamp=_parse_datetime(row.get("queued_timestamp")),
            )
        )
    return statuses


@dataclass
class RuntimeHistory:
    global_median_sec: float
    by_analysis_name: Dict[str, float]
    by_config_profile: Dict[str, float]
    by_project_id: Dict[str, float]

    def lookup(self, analysis_name: Optional[str], config_profile: Optional[str], project_id: Optional[str]) -> float:
        if analysis_name and analysis_name in self.by_analysis_name:
            return self.by_analysis_name[analysis_name]
        if config_profile and config_profile in self.by_config_profile:
            return self.by_config_profile[config_profile]
        if project_id and project_id in self.by_project_id:
            return self.by_project_id[project_id]
        return self.global_median_sec


def build_runtime_history(history_df: Optional[pd.DataFrame]) -> RuntimeHistory:
    if history_df is None or history_df.empty:
        return RuntimeHistory(global_median_sec=3600.0, by_analysis_name={}, by_config_profile={}, by_project_id={})
    history_df = history_df.copy()
    if "runtime_sec" not in history_df.columns and {"start_time", "end_time"}.issubset(history_df.columns):
        history_df["runtime_sec"] = (
            pd.to_datetime(history_df["end_time"]) - pd.to_datetime(history_df["start_time"])
        ).dt.total_seconds()
    history_df = history_df.dropna(subset=["runtime_sec"])
    global_median = float(history_df["runtime_sec"].median()) if not history_df.empty else 3600.0

    def _median_by(column: str) -> Dict[str, float]:
        if column not in history_df.columns:
            return {}
        grouped = history_df.dropna(subset=[column]).groupby(column)["runtime_sec"].median()
        return {str(key): float(value) for key, value in grouped.items()}

    return RuntimeHistory(
        global_median_sec=global_median,
        by_analysis_name=_median_by("analysis_name"),
        by_config_profile=_median_by("config_profile"),
        by_project_id=_median_by("project_id"),
    )


@dataclass
class MemoryThresholds:
    free_mem_threshold_gb: float
    total_mem_spike_gb: float
    mem_growth_rate_gb_per_hour: float


def estimate_queue_wait_sec(
    analysis: AnalysisStatus,
    snapshot: Optional[ServerSnapshot],
    history: RuntimeHistory,
    thresholds: MemoryThresholds,
) -> float:
    base_runtime = history.lookup(analysis.analysis_name, None, None)
    if not snapshot:
        return base_runtime
    slots = max(snapshot.processing_count, 1)
    queue_pressure = snapshot.queued_count / slots
    mem_penalty = 1.0
    if snapshot.free_mem_gb < thresholds.free_mem_threshold_gb and thresholds.free_mem_threshold_gb > 0:
        mem_penalty += (thresholds.free_mem_threshold_gb - snapshot.free_mem_gb) / thresholds.free_mem_threshold_gb
    return base_runtime * (1.0 + queue_pressure) * mem_penalty


def estimate_total_runtime_sec(
    analysis: AnalysisStatus,
    history: RuntimeHistory,
    model_p50_sec: Optional[float],
) -> float:
    historical = history.lookup(analysis.analysis_name, None, None)
    candidates = [historical]
    if model_p50_sec:
        candidates.append(model_p50_sec)
    elapsed = (analysis.duration_hours or 0.0) * 3600
    if elapsed:
        candidates.append(elapsed)
    return max(candidates)


def detect_anomalies(
    analysis: AnalysisStatus,
    snapshot: Optional[ServerSnapshot],
    thresholds: MemoryThresholds,
) -> List[str]:
    reasons: List[str] = []
    if snapshot:
        if snapshot.free_mem_gb < thresholds.free_mem_threshold_gb:
            reasons.append("Free memory below threshold; potential slowdown risk.")
        if snapshot.total_mem_used_gb > thresholds.total_mem_spike_gb:
            reasons.append("Total memory usage spike detected on server.")
    if analysis.status == "PROCESSING" and analysis.used_memory_gb and analysis.duration_hours:
        rate = analysis.used_memory_gb / max(analysis.duration_hours, 0.1)
        if rate > thresholds.mem_growth_rate_gb_per_hour:
            reasons.append("Used memory growth rate exceeds threshold.")
    return reasons
