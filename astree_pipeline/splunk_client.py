"""Splunk REST client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import requests


@dataclass
class SplunkConfig:
    base_url: str
    token: str
    index: str
    timeout_sec: int = 30


class SplunkClient:
    """Minimal Splunk REST API wrapper for searches."""

    def __init__(self, config: SplunkConfig) -> None:
        self._config = config

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.token}"}

    def search(self, query: str) -> dict:
        """Execute a search query and return the JSON payload."""
        url = f"{self._config.base_url}/services/search/jobs/export"
        response = requests.post(
            url,
            data={"search": query, "output_mode": "json"},
            headers=self._headers(),
            timeout=self._config.timeout_sec,
        )
        response.raise_for_status()
        return response.json()

    def build_query(self, status: str) -> str:
        """Build a query for completed or running runs."""
        return (
            f"search index={self._config.index} status={status} "
            "| table project_id, astree_version, config_profile, loc, modules, "
            "start_time, end_time, total_runtime_sec, current_elapsed_time"
        )

    def fetch_runs(self, status: str) -> Iterable[dict]:
        """Fetch runs of a given status from Splunk."""
        query = self.build_query(status)
        payload = self.search(query)
        return payload.get("results", [])
