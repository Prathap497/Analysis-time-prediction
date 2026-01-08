from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str
    recipients: Iterable[str]


class EmailNotifier:
    def __init__(self, config: EmailConfig) -> None:
        self.config = config

    def send(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self.config.sender
        message["To"] = ", ".join(self.config.recipients)
        message["Subject"] = subject
        message.set_content(body)

        with smtplib.SMTP(self.config.host, self.config.port) as server:
            if self.config.user and self.config.password:
                server.starttls()
                server.login(self.config.user, self.config.password)
            server.send_message(message)
        logger.info("Email sent", extra={"subject": subject})


def render_notification(payload: dict) -> str:
    return (
        f"Run ID: {payload['run_id']}\n"
        f"Project: {payload['project_id']}\n"
        f"Host: {payload['host']}\n"
        f"Version: {payload['astree_version']}\n"
        f"Config: {payload['config_profile']}\n"
        f"Status: {payload['status']}\n"
        f"Elapsed: {payload['elapsed_hours']:.2f} h\n"
        f"Predicted total (P50): {payload['total_p50_hours']:.2f} h\n"
        f"Interval (P10-P90): {payload['total_p10_hours']:.2f} - {payload['total_p90_hours']:.2f} h\n"
        f"Remaining: {payload['remaining_hours']:.2f} h\n"
        f"Progress: {payload['progress_pct']:.1f}%\n"
        f"Splunk: {payload.get('splunk_url', 'n/a')}\n"
    )
