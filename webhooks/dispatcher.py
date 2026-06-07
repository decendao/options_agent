"""
webhooks/dispatcher.py — Alert Dispatcher
==========================================
Dispatches RiskAlert objects to:
  - Generic webhook (configurable URL + HMAC signing)
  - Slack Incoming Webhook (Block Kit format)
  - Stdout (dry-run mode)

All dispatches are async and non-blocking. Failures are logged
but never propagate to the risk loop.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Optional

import aiohttp

from options_agent.core.schemas import RiskAlert, RiskLevel
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)


class AlertDispatcher:
    """
    Unified alert dispatcher. Routes alerts based on risk level and config.
    Dry-run mode: prints to stdout, no HTTP calls.
    """

    def __init__(
        self,
        dry_run: bool = True,
        webhook_url: str = "",
        webhook_secret: str = "",
        slack_webhook_url: str = "",
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.dry_run = dry_run
        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.slack_webhook_url = slack_webhook_url
        self._session = session
        self._owns_session = session is None
        self._dispatch_count = 0

    async def connect(self) -> None:
        if self._owns_session and self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0)
            )

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def dispatch(self, alert: RiskAlert) -> bool:
        """
        Dispatch an alert through all configured channels.
        Returns True if at least one channel succeeded.
        """
        self._dispatch_count += 1
        success = False

        # Always log
        self._log_alert(alert)

        if self.dry_run:
            self._print_dry_run(alert)
            return True

        # Generic webhook
        if self.webhook_url:
            ok = await self._send_webhook(alert)
            success = success or ok

        # Slack
        if self.slack_webhook_url:
            ok = await self._send_slack(alert)
            success = success or ok

        return success

    # ------------------------------------------------------------------
    # Channel implementations
    # ------------------------------------------------------------------

    async def _send_webhook(self, alert: RiskAlert) -> bool:
        payload = alert.model_dump(mode="json", exclude={"raw_matrix"})
        payload_bytes = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-Alert-ID": alert.alert_id,
            "X-Risk-Level": alert.risk_level.value,
        }

        if self.webhook_secret:
            sig = hmac.new(
                self.webhook_secret.encode(),
                payload_bytes,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={sig}"

        try:
            async with self._session.post(
                self.webhook_url,
                data=payload_bytes,
                headers=headers,
            ) as resp:
                if resp.status < 300:
                    logger.info(f"Webhook dispatched: {alert.alert_id} [{alert.risk_level}]")
                    return True
                else:
                    logger.warning(f"Webhook returned {resp.status} for {alert.alert_id}")
                    return False
        except Exception as e:
            logger.error(f"Webhook dispatch failed for {alert.alert_id}: {e}")
            return False

    async def _send_slack(self, alert: RiskAlert) -> bool:
        try:
            blocks = alert.to_slack_blocks()
            payload = {
                "text": f"[{alert.risk_level}] {alert.ticker}: {alert.headline}",
                "blocks": blocks,
            }
            async with self._session.post(
                self.slack_webhook_url,
                json=payload,
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Slack alert sent: {alert.alert_id}")
                    return True
                else:
                    body = await resp.text()
                    logger.warning(f"Slack returned {resp.status}: {body}")
                    return False
        except Exception as e:
            logger.error(f"Slack dispatch failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Logging / dry-run
    # ------------------------------------------------------------------

    def _log_alert(self, alert: RiskAlert) -> None:
        level_map = {
            RiskLevel.LOW: logger.info,
            RiskLevel.MEDIUM: logger.warning,
            RiskLevel.HIGH: logger.error,
            RiskLevel.CRITICAL: logger.critical,
        }
        log_fn = level_map.get(alert.risk_level, logger.warning)
        log_fn(
            f"ALERT [{alert.risk_level}] {alert.ticker} | {alert.headline} | "
            f"Rules: {alert.triggered_rules}"
        )

    def _print_dry_run(self, alert: RiskAlert) -> None:
        border = "=" * 70
        print(f"\n{border}")
        print(f"  [DRY-RUN ALERT] {alert.risk_level} -- {alert.ticker}")
        print(f"  {alert.headline}")
        print(f"  {border}")
        print(f"  NARRATIVE:\n  {alert.narrative}")
        print(f"  ACTION: {alert.recommended_action}")
        if alert.spot_price:
            print(f"  Spot: ${alert.spot_price:.2f}", end="")
        if alert.iv_rank:
            print(f"  | IV Rank: {alert.iv_rank:.1%}", end="")
        if alert.gamma_wall_strike:
            print(f"  | Gamma Wall: ${alert.gamma_wall_strike:.2f}", end="")
        print(f"\n  Rules triggered: {', '.join(alert.triggered_rules)}")
        print(f"  Time: {alert.timestamp_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{border}\n")

    @property
    def total_dispatched(self) -> int:
        return self._dispatch_count
