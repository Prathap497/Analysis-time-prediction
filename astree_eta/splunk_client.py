from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class SplunkConfig:
    base_url: str
    token: str
    verify_tls: bool
    app: str
    index: str
    sourcetype: str
    host_filter: Optional[str] = None


class SplunkClient:
    def __init__(self, config: SplunkConfig, timeout: int = 60) -> None:
        self.config = config
        self.timeout = timeout
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Splunk {self.config.token}"}

    def export_search(self, search: str, earliest: str, latest: str) -> Iterator[Dict[str, str]]:
        url = f"{self.config.base_url}/services/search/jobs/export"
        payload = {
            "search": search,
            "earliest_time": earliest,
            "latest_time": latest,
            "output_mode": "json",
        }
        response = self.session.post(
            url,
            headers=self._headers(),
            data=payload,
            timeout=self.timeout,
            verify=self.config.verify_tls,
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            if "result" in data:
                yield data["result"]

    def build_base_search(self) -> str:
        base = f"search index={self.config.index} sourcetype={self.config.sourcetype}"
        if self.config.app:
            base += f" app={self.config.app}"
        if self.config.host_filter:
            base += f" host={self.config.host_filter}"
        return base


def build_completed_runs_query(config: SplunkConfig) -> str:
    base = SplunkClient(config).build_base_search()
    return (
        f"{base} status=COMPLETED "
        "| stats latest(project_id) as project_id "
        "latest(build_number) as build_number "
        "latest(analysis_name) as analysis_name "
        "latest(astree_version) as astree_version "
        "latest(config_profile) as config_profile "
        "latest(host) as host "
        "latest(loc) as loc "
        "latest(kloc) as kloc "
        "latest(num_files) as num_files "
        "latest(num_tu) as num_tu "
        "latest(cpu_avg) as cpu_avg "
        "latest(server_load_avg) as server_load_avg "
        "min(start_time) as start_time "
        "max(end_time) as end_time "
        "latest(status) as status "
        "by run_id"
    )


def build_running_runs_query(config: SplunkConfig) -> str:
    base = SplunkClient(config).build_base_search()
    return (
        f"{base} status=RUNNING "
        "| stats latest(project_id) as project_id "
        "latest(build_number) as build_number "
        "latest(analysis_name) as analysis_name "
        "latest(astree_version) as astree_version "
        "latest(config_profile) as config_profile "
        "latest(host) as host "
        "latest(loc) as loc "
        "latest(kloc) as kloc "
        "latest(num_files) as num_files "
        "latest(num_tu) as num_tu "
        "latest(cpu_avg) as cpu_avg "
        "latest(server_load_avg) as server_load_avg "
        "min(start_time) as start_time "
        "latest(_time) as last_log_time "
        "latest(status) as status "
        "by run_id"
    )


def build_status_transition_query(config: SplunkConfig) -> str:
    base = SplunkClient(config).build_base_search()
    return (
        f"{base} (status=COMPLETED OR status=FAILED OR status=ABORTED) "
        "| stats latest(project_id) as project_id "
        "latest(build_number) as build_number "
        "latest(analysis_name) as analysis_name "
        "latest(astree_version) as astree_version "
        "latest(config_profile) as config_profile "
        "latest(host) as host "
        "latest(status) as status "
        "max(end_time) as end_time "
        "by run_id"
    )
