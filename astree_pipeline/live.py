"""Live prediction logic and anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveEstimate:
    predicted_total_sec: float
    remaining_sec: float
    completion_ratio: float


class LiveEstimator:
    """Estimate remaining runtime based on predictions."""

    def estimate(self, predicted_total_sec: float, elapsed_sec: float) -> LiveEstimate:
        remaining = max(predicted_total_sec - elapsed_sec, 0.0)
        completion_ratio = min(elapsed_sec / predicted_total_sec, 1.0) if predicted_total_sec else 0.0
        return LiveEstimate(
            predicted_total_sec=predicted_total_sec,
            remaining_sec=remaining,
            completion_ratio=completion_ratio,
        )


@dataclass
class AnomalyResult:
    is_anomaly: bool
    reason: str


class AnomalyDetector:
    """Simple anomaly detection based on thresholds."""

    def __init__(self, p90_threshold_sec: float) -> None:
        self._p90_threshold_sec = p90_threshold_sec

    def check_elapsed(self, elapsed_sec: float) -> AnomalyResult:
        if elapsed_sec > self._p90_threshold_sec:
            return AnomalyResult(is_anomaly=True, reason="Elapsed time exceeded P90 prediction.")
        return AnomalyResult(is_anomaly=False, reason="Elapsed time within expected range.")
