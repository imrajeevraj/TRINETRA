import json
import logging
import re
import ssl
import threading
from typing import Dict, Any

import paho.mqtt.client as mqtt
import httpx

from backend.app.core.config import settings

logger = logging.getLogger("AlertDispatch")

# S-10: Telegram bot-token must match official format before we accept it.
_TELEGRAM_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{35,}$")


def _validate_telegram_token(token: str) -> bool:
    """S-10: Return True only if the token matches the Telegram Bot API format."""
    return bool(token and _TELEGRAM_TOKEN_RE.match(token))


class AlertDispatchService:
    # R-08: Exponential back-off parameters for MQTT reconnect.
    _MQTT_BACKOFF_BASE: float = 2.0  # seconds
    _MQTT_BACKOFF_MAX: float = 120.0  # seconds cap

    def __init__(self):
        self.mqtt_client = None
        self.mqtt_connected = False
        self._mqtt_backoff = self._MQTT_BACKOFF_BASE
        self._mqtt_backoff_lock = threading.Lock()

        # S-10: Validate Telegram token at startup so misconfigured tokens fail fast.
        if settings.TELEGRAM_ENABLED:
            if not _validate_telegram_token(settings.TELEGRAM_BOT_TOKEN):
                raise RuntimeError(
                    "TELEGRAM_BOT_TOKEN is invalid or missing. "
                    "Expected format: <bot_id>:<35+ char key>. "
                    "Set TELEGRAM_ENABLED=false to skip."
                )
            logger.info("Telegram bot-token validated successfully.")

        if settings.MQTT_ENABLED:
            self._init_mqtt()

    def _init_mqtt(self):
        try:
            try:
                from paho.mqtt.enums import CallbackAPIVersion

                self.mqtt_client = mqtt.Client(
                    CallbackAPIVersion.VERSION2, client_id="ibvap_dispatcher"
                )
            except ImportError:
                self.mqtt_client = mqtt.Client(client_id="ibvap_dispatcher")

            if settings.MQTT_USERNAME and settings.MQTT_PASSWORD:
                self.mqtt_client.username_pw_set(
                    settings.MQTT_USERNAME, settings.MQTT_PASSWORD
                )

            # S-09: Enable TLS when MQTT_TLS_ENABLED is set.
            if getattr(settings, "MQTT_TLS_ENABLED", False):
                ca_cert = getattr(settings, "MQTT_CA_CERT", None) or None
                self.mqtt_client.tls_set(
                    ca_certs=ca_cert,  # None = use system CA bundle
                    tls_version=ssl.PROTOCOL_TLS_CLIENT,
                )
                logger.info("MQTT TLS enabled (CA: %s)", ca_cert or "system bundle")

            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

            self.mqtt_client.connect_async(
                settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, 60
            )
            self.mqtt_client.loop_start()
        except Exception as e:
            logger.error("Failed to initialize MQTT client: %s", e)

    def _on_mqtt_connect(self, client, userdata, flags, rc, *args):
        if rc == 0:
            self.mqtt_connected = True
            with self._mqtt_backoff_lock:
                self._mqtt_backoff = self._MQTT_BACKOFF_BASE  # reset on success
            logger.info("Connected to MQTT broker at %s", settings.MQTT_BROKER_HOST)
        else:
            logger.error("MQTT connection failed with code %d", rc)

    def _on_mqtt_disconnect(self, client, userdata, rc, *args):
        self.mqtt_connected = False
        if rc != 0:
            # R-08: Exponential back-off with 25% jitter so multiple instances
            # don't reconnect simultaneously after a broker restart.
            import random

            with self._mqtt_backoff_lock:
                delay = self._mqtt_backoff * (1.0 + 0.25 * random.random())
                self._mqtt_backoff = min(self._mqtt_backoff * 2, self._MQTT_BACKOFF_MAX)
            logger.warning(
                "Unexpected MQTT disconnection (rc=%d). Reconnecting in %.1fs.",
                rc,
                delay,
            )
            threading.Timer(delay, self._reconnect_mqtt).start()

    def _reconnect_mqtt(self):
        """R-08: Attempt a single reconnect; paho loop_start handles retries."""
        try:
            if self.mqtt_client:
                self.mqtt_client.reconnect()
        except Exception as e:
            logger.error("MQTT reconnect attempt failed: %s", e)

    def dispatch_event(self, event_dict: Dict[str, Any]):
        """Evaluate and dispatch high-severity events to external systems."""
        severity = event_dict.get("severity", "LOW")
        if severity not in ["HIGH", "CRITICAL"]:
            return

        payload = self._format_payload(event_dict)

        # 1. MQTT dispatch
        if settings.MQTT_ENABLED and self.mqtt_client and self.mqtt_connected:
            try:
                topic = (
                    f"{settings.MQTT_TOPIC_PREFIX}"
                    f"/{severity.lower()}"
                    f"/{event_dict.get('event_type', 'unknown').lower()}"
                )
                self.mqtt_client.publish(topic, json.dumps(payload), qos=1)
                logger.info("Dispatched event via MQTT to %s", topic)
            except Exception as e:
                logger.error("MQTT dispatch failed: %s", e)

        # 2. Telegram dispatch
        if (
            settings.TELEGRAM_ENABLED
            and settings.TELEGRAM_BOT_TOKEN
            and settings.TELEGRAM_CHAT_ID
        ):
            self._dispatch_telegram(payload, severity)

    def _format_payload(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "alert_id": event_dict.get("id"),
            "severity": event_dict.get("severity"),
            "type": event_dict.get("event_type"),
            "camera": event_dict.get("camera_id"),
            "timestamp": event_dict.get("timestamp"),
            "reasons": event_dict.get("risk_reasons", []),
            "score": event_dict.get("risk_score"),
        }

    def _dispatch_telegram(self, payload: Dict[str, Any], severity: str):
        try:
            url = (
                f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
            )
            icon = "🚨" if severity == "CRITICAL" else "⚠️"
            reasons = ", ".join(payload.get("reasons", []))
            text = (
                f"{icon} *TRINETRA ALERT: {severity}*\n"
                f"*Type:* {payload.get('type')}\n"
                f"*Camera:* {payload.get('camera')}\n"
                f"*Reasons:* {reasons}\n"
                f"*Time:* {payload.get('timestamp')}"
            )
            data = {
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            }
            threading.Thread(
                target=self._send_http, args=(url, data), daemon=True
            ).start()
        except Exception as e:
            logger.error("Failed to prep Telegram dispatch: %s", e)

    def _send_http(self, url: str, data: Dict[str, Any]):
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=data)
                response.raise_for_status()
            logger.info("Dispatched event via Telegram webhook")
        except Exception as e:
            logger.error("HTTP Webhook dispatch failed: %s", e)


alert_dispatch = AlertDispatchService()
