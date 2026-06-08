"""
agents/scout_agent.py — 前哨兵 Scout Agent
==========================================
Real-time Unusual Whales WebSocket options flow listener.

Architecture role:
  Runs as an INDEPENDENT asyncio.Task — NOT part of the 30s batch polling cycle.
  Filters for premium > min_premium_usd (default $100k).
  Pushes qualifying UnusualFlowSignal onto scout_queue for downstream consumers.
  Broadcasts each signal to WebSocket frontend clients immediately.

Reconnect strategy:
  Exponential backoff: delay × 2^n (capped at 120s) + 10% jitter.
  Unlimited reconnects by default (max_reconnect_attempts=0).
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from options_agent.config import Settings
from options_agent.core.schemas import ScoutAgentStatus, UnusualFlowSignal
from options_agent.harness.logger import get_logger

if TYPE_CHECKING:
    from options_agent.api_server import APIRegistry

logger = get_logger(__name__)


class ScoutAgent:
    """
    前哨兵 Scout Agent — Unusual Whales realtime flow sentinel.

    Runs perpetually as a parallel asyncio.Task alongside the main
    A→B→C polling pipeline.  Each qualifying signal is:
      1. Pushed onto scout_queue (for future Analytica Agent consumption)
      2. Broadcast to all connected WebSocket clients (immediate frontend push)
      3. Cached in APIRegistry (accessible via GET /flow)
    """

    def __init__(
        self,
        settings: Settings,
        scout_queue: asyncio.Queue,
        api_registry: Optional[APIRegistry] = None,
    ):
        self.settings = settings
        self.scout_queue = scout_queue
        self._api_registry = api_registry
        self._shutdown = False

        # Counters (written only by this task — no locking needed)
        self._reconnect_count = 0
        self._signals_today = 0
        self._total_signals = 0
        self._last_signal_at: Optional[datetime] = None
        self._last_error: str = ""
        self._is_connected = False

        uw = settings.unusual_whales
        self._min_premium = uw.min_premium_usd
        self._reconnect_delay_base = uw.reconnect_delay_seconds
        self._max_reconnects = uw.max_reconnect_attempts
        self._tickers = settings.get_enabled_tickers()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def status(self) -> ScoutAgentStatus:
        return ScoutAgentStatus(
            is_connected=self._is_connected,
            last_signal_at=self._last_signal_at,
            signals_today=self._signals_today,
            total_signals=self._total_signals,
            reconnect_count=self._reconnect_count,
            last_error=self._last_error,
        )

    async def run(self) -> None:
        """Persistent loop: connect → stream → reconnect on any failure."""
        logger.info(
            f"[Scout] Starting | "
            f"filter=${self._min_premium:,.0f} premium | "
            f"tickers={self._tickers[:6]}{'...' if len(self._tickers) > 6 else ''}"
        )

        provider = self._build_provider()
        await provider.connect()

        try:
            while not self._shutdown:
                try:
                    self._set_connected(True)
                    self._sync_status()

                    async for signal in provider.stream():
                        if self._shutdown:
                            return
                        if signal.premium_usd >= self._min_premium:
                            await self._dispatch(signal)

                    logger.warning("[Scout] Stream ended cleanly — scheduling reconnect")

                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    self._last_error = str(exc)
                    self._set_connected(False)
                    self._sync_status()
                    logger.error(f"[Scout] Stream error: {exc}", exc_info=True)

                if self._max_reconnects and self._reconnect_count >= self._max_reconnects:
                    logger.error(
                        f"[Scout] Exhausted {self._max_reconnects} reconnect attempts — stopping"
                    )
                    return

                # Exponential backoff with ±10% jitter, capped at 120s
                delay = min(
                    self._reconnect_delay_base * (2 ** min(self._reconnect_count, 6)),
                    120.0,
                )
                delay += random.uniform(-delay * 0.1, delay * 0.1)
                self._reconnect_count += 1
                logger.info(f"[Scout] Reconnect #{self._reconnect_count} in {delay:.1f}s")
                self._sync_status()
                await asyncio.sleep(delay)

        finally:
            self._set_connected(False)
            self._sync_status()
            await provider.close()
            logger.info(
                f"[Scout] Stopped | signals={self._total_signals} | "
                f"reconnects={self._reconnect_count}"
            )

    async def shutdown(self) -> None:
        self._shutdown = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_provider(self):
        uw = self.settings.unusual_whales
        use_mock = self.settings.use_mock_data or uw.use_mock or not uw.api_token

        if use_mock:
            logger.info("[Scout] No API token — using MockUnusualWhalesProvider")
            from options_agent.data.unusual_whales_provider import MockUnusualWhalesProvider
            return MockUnusualWhalesProvider(
                tickers_filter=self._tickers,
                emit_interval_seconds=3.0,
            )

        from options_agent.data.unusual_whales_provider import UnusualWhalesProvider
        logger.info(f"[Scout] Connecting to live feed → {uw.ws_url}")
        return UnusualWhalesProvider(
            api_token=uw.api_token,
            ws_url=uw.ws_url,
            tickers_filter=self._tickers,
            timeout_seconds=uw.timeout_seconds,
        )

    async def _dispatch(self, signal: UnusualFlowSignal) -> None:
        """Enqueue signal, push to WebSocket clients, cache in registry."""
        self._total_signals += 1
        self._signals_today += 1
        self._last_signal_at = datetime.utcnow()
        self._last_error = ""

        # ── Queue (non-blocking, drop oldest on overflow) ────────────
        try:
            self.scout_queue.put_nowait(signal)
        except asyncio.QueueFull:
            try:
                self.scout_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.scout_queue.put_nowait(signal)

        _type_label = "CALL" if signal.option_type.value == "call" else "PUT"
        _flags = (
            ("SWEEP " if signal.is_sweep else "") +
            ("BLOCK " if signal.is_block else "")
        ).strip()

        logger.info(
            f"[Scout] ✦ {signal.ticker} {_type_label} "
            f"${signal.strike:.0f} {signal.expiration} | "
            f"${signal.premium_usd:,.0f} | "
            f"{_flags + ' | ' if _flags else ''}"
            f"{signal.sentiment.value.upper()} | "
            f"Δ={signal.delta or 0:.2f} IV={signal.implied_volatility or 0:.0%}"
        )

        # ── WebSocket broadcast ──────────────────────────────────────
        if self._api_registry:
            asyncio.create_task(
                self._api_registry.ws_manager.broadcast({
                    "type": "flow_signal",
                    "signal_id": signal.signal_id,
                    "ticker": signal.ticker,
                    "expiration": signal.expiration.isoformat(),
                    "strike": signal.strike,
                    "option_type": signal.option_type.value,
                    "premium_usd": signal.premium_usd,
                    "volume": signal.volume,
                    "open_interest": signal.open_interest,
                    "spot_price_at_trade": signal.spot_price_at_trade,
                    "implied_volatility": signal.implied_volatility,
                    "delta": signal.delta,
                    "sentiment": signal.sentiment.value,
                    "is_sweep": signal.is_sweep,
                    "is_block": signal.is_block,
                    "exchange": signal.exchange,
                    "timestamp_utc": signal.timestamp_utc.isoformat(),
                    "source": signal.source,
                })
            )
            self._api_registry.register_flow_signal(signal)

        self._sync_status()

    def _set_connected(self, connected: bool) -> None:
        self._is_connected = connected

    def _sync_status(self) -> None:
        if self._api_registry:
            self._api_registry.register_scout_status(self.status)
