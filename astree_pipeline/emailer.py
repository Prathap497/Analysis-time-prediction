"""Email notification utilities."""

from __future__ import annotations

from dataclasses import dataclass
import smtplib
from email.message import EmailMessage


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    sender: str
    recipients: list[str]


class EmailNotifier:
    """Send notification emails about analysis status."""

    def __init__(self, config: EmailConfig) -> None:
        self._config = config

    def send_status(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self._config.sender
        message["To"] = ", ".join(self._config.recipients)
        message.set_content(body)

        with smtplib.SMTP(self._config.smtp_host, self._config.smtp_port) as smtp:
            smtp.send_message(message)
