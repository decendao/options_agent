"""
data/unusual_whales_provider.py — Unusual Whales WebSocket Provider
====================================================================
Real-time options flow feed from Unusual Whales.

WebSocket API:
  wss://api.unusualwhales.com/socket/option-activity
  Auth:  { "action": "auth", "key": "<API_TOKEN>" }
  Feed:  JSON messages with option activity events

Field reference (Unusual Whales v2 schema):
  ticker, expiry_date, strike, put_call, option_activity_type,
  premium (USD), size (contracts), open_interest, underlying_price,
  iv, delta, sentiment, exchange, created_at
"""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import date, datetime, timedelta
from typing import AsyncIterator, Optional

import aiohttp

from options_agent.core.schemas import FlowSentiment, OptionType, UnusualFlowSignal
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_WS_URL = "wss://api.unusualwhales.com/socket/option-activity"


# ---------------------------------------------------------------------------
# Real provider
# ---------------------------------------------------------------------------

class UnusualWhalesProvider:
    """
    Unusual Whales WebSocket client.

    Lifecycle:
        await provider.connect()
        async for signal in provider.stream():
            ...  # yields UnusualFlowSignal
        await provider.close()
    """

    def __init__(
        self,
        api_token: str,
        ws_url: str = _DEFAULT_WS_URL,
        tickers_filter: Optional[list[str]] = None,
        timeout_seconds: float = 10.0,
    ):
        self.api_token = api_token
        self.ws_url = ws_url
        self.tickers_filter = {t.upper() for t in (tickers_filter or [])}
        self._timeout = aiohttp.ClientTimeout(total=None, connect=timeout_seconds)
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def stream(self) -> AsyncIterator[UnusualFlowSignal]:
        """Yield UnusualFlowSignal objects from the WebSocket feed."""
        if self._session is None:
            await self.connect()

        headers = {"Authorization": f"Bearer {self.api_token}"}

        async with self._session.ws_connect(self.ws_url, headers=headers) as ws:
            logger.info(f"[UnusualWhales] WebSocket connected → {self.ws_url}")

            # Some API versions require an explicit auth message
            await ws.send_json({"action": "auth", "key": self.api_token})

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                        signal = _parse_message(payload, self.tickers_filter)
                        if signal is not None:
                            yield signal
                    except json.JSONDecodeError as e:
                        logger.debug(f"[UnusualWhales] JSON parse error: {e}")
                    except Exception as e:
                        logger.warning(f"[UnusualWhales] Message parse error: {e}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[UnusualWhales] WS error: {ws.exception()}")
                    break

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    logger.warning("[UnusualWhales] WebSocket closed by server")
                    break


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------

def _parse_message(
    payload: dict,
    tickers_filter: set[str],
) -> Optional[UnusualFlowSignal]:
    """Parse a raw Unusual Whales WebSocket message."""
    msg_type = payload.get("type", "")

    # Skip control frames
    if msg_type in ("authenticated", "connected", "subscribed", "heartbeat", "pong"):
        logger.debug(f"[UnusualWhales] Control: {msg_type}")
        return None

    # Data lives in "data" key or directly in payload
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return None

    ticker = (data.get("ticker") or data.get("symbol") or "").strip().upper()
    if not ticker:
        return None

    if tickers_filter and ticker not in tickers_filter:
        return None

    return _build_signal(data, ticker)


def _build_signal(data: dict, ticker: str) -> Optional[UnusualFlowSignal]:
    """Convert raw flow dict to UnusualFlowSignal; returns None on bad data."""
    try:
        # ── Option type ──────────────────────────────────────────────
        raw_pc = (data.get("put_call") or data.get("option_type") or "").upper()
        if raw_pc not in ("CALL", "PUT"):
            return None
        option_type = OptionType.CALL if raw_pc == "CALL" else OptionType.PUT

        # ── Expiration ───────────────────────────────────────────────
        expiry_raw = (
            data.get("expiry_date") or data.get("expiry") or
            data.get("expiration") or data.get("exp_date") or ""
        )
        expiry = _parse_date(str(expiry_raw))

        # ── Strike ───────────────────────────────────────────────────
        strike = float(data.get("strike") or data.get("strike_price") or 0)
        if strike <= 0:
            return None

        # ── Premium (UW stores in USD, but some feeds use cents) ────
        premium_raw = float(data.get("premium") or data.get("cost_basis") or 0)
        # Heuristic: values > 10_000_000 are likely stored in cents
        premium_usd = premium_raw / 100.0 if premium_raw > 10_000_000 else premium_raw

        # ── Volume & OI ──────────────────────────────────────────────
        volume = int(data.get("size") or data.get("volume") or 0)
        oi = int(data.get("open_interest") or data.get("oi") or 0)

        # ── Spot price ───────────────────────────────────────────────
        spot = float(
            data.get("underlying_price") or data.get("stock_price") or
            data.get("spot") or data.get("reference_price") or 0
        )
        if spot <= 0:
            spot = strike  # last resort fallback

        # ── Greeks ───────────────────────────────────────────────────
        iv_raw = data.get("iv") or data.get("implied_volatility")
        delta_raw = data.get("delta")

        # ── Sentiment ────────────────────────────────────────────────
        raw_sent = (data.get("sentiment") or "").upper()
        sentiment = {
            "BULLISH": FlowSentiment.BULLISH,
            "BEARISH": FlowSentiment.BEARISH,
        }.get(raw_sent, FlowSentiment.NEUTRAL)

        # ── Trade flags ──────────────────────────────────────────────
        activity = (
            data.get("option_activity_type") or data.get("trade_type") or ""
        ).upper()
        is_sweep = "SWEEP" in activity
        is_block = "BLOCK" in activity

        exchange = data.get("exchange") or data.get("venue") or ""

        # ── Timestamp ────────────────────────────────────────────────
        ts_raw = data.get("created_at") or data.get("timestamp") or data.get("time")
        ts = _parse_ts(str(ts_raw)) if ts_raw else datetime.utcnow()

        signal_id = str(data.get("id") or data.get("uid") or uuid.uuid4().hex[:12])

        return UnusualFlowSignal(
            signal_id=signal_id,
            ticker=ticker,
            expiration=expiry,
            strike=strike,
            option_type=option_type,
            premium_usd=premium_usd,
            volume=volume,
            open_interest=oi,
            spot_price_at_trade=spot,
            implied_volatility=float(iv_raw) if iv_raw is not None else None,
            delta=float(delta_raw) if delta_raw is not None else None,
            sentiment=sentiment,
            is_sweep=is_sweep,
            is_block=is_block,
            exchange=exchange,
            timestamp_utc=ts,
            source="unusual_whales",
        )
    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"[UnusualWhales] Could not build signal for {ticker}: {e}")
        return None


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse expiry date: {s!r}")


def _parse_ts(s: str) -> datetime:
    try:
        ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return ts.replace(tzinfo=None)
    except ValueError:
        return datetime.utcnow()


# ---------------------------------------------------------------------------
# Mock provider for offline / CI testing
# ---------------------------------------------------------------------------

class MockUnusualWhalesProvider:
    """
    Synthetic Unusual Whales feed.

    Emits realistic UnusualFlowSignal objects at a configurable interval.
    Signals span a mix of premiums — some above $100k (pass the filter),
    some below (should be dropped by ScoutAgent).
    """

    _BASE_PRICES: dict[str, float] = {
        "SPY": 520.0, "QQQ": 440.0, "AAPL": 188.0, "NVDA": 875.0,
        "TSLA": 178.0, "META": 495.0, "MSFT": 418.0, "AMZN": 187.0,
        "AMD": 155.0, "GOOGL": 175.0,
    }
    _EXCHANGES = ["CBOE", "PHLX", "BOX", "MIAX", "ISE", "C2"]

    def __init__(
        self,
        tickers_filter: Optional[list[str]] = None,
        emit_interval_seconds: float = 3.0,
    ):
        self._filter = [t.upper() for t in (tickers_filter or list(self._BASE_PRICES))]
        self._interval = emit_interval_seconds
        self._running = False

    async def connect(self) -> None:
        self._running = True
        logger.info(
            f"[MockUnusualWhales] Started | "
            f"tickers={self._filter[:5]}{'...' if len(self._filter) > 5 else ''} | "
            f"interval={self._interval}s"
        )

    async def close(self) -> None:
        self._running = False

    async def stream(self) -> AsyncIterator[UnusualFlowSignal]:
        self._running = True
        while self._running:
            ticker = random.choice(self._filter)
            yield self._make_signal(ticker)
            await asyncio.sleep(self._interval * (0.6 + random.random() * 0.8))

    def _make_signal(self, ticker: str) -> UnusualFlowSignal:
        spot = self._BASE_PRICES.get(ticker, 100.0) * (1 + random.gauss(0, 0.004))

        option_type = random.choice([OptionType.CALL, OptionType.PUT])

        # Strike near ATM
        step = max(round(spot * 0.005 / 2.5) * 2.5, 1.0)
        strike = round(spot + random.choice([-2, -1, 0, 1, 2]) * step, 2)

        # Expiry: 7–60 days out, anchored to nearest Friday
        expiry = date.today() + timedelta(days=random.randint(7, 60))
        expiry += timedelta(days=(4 - expiry.weekday()) % 7)

        # Premium distribution: ~30% below $100k (filtered), ~70% above
        premium_bucket = random.random()
        if premium_bucket < 0.30:
            premium_usd = random.uniform(20_000, 99_000)
        elif premium_bucket < 0.70:
            premium_usd = random.uniform(100_000, 500_000)
        else:
            premium_usd = random.uniform(500_000, 3_000_000)

        price_per = max(spot * 0.01, 0.05)
        volume = max(1, round(premium_usd / (price_per * 100)))

        iv = random.uniform(0.15, 0.85)
        delta = (1 if option_type == OptionType.CALL else -1) * random.uniform(0.25, 0.75)

        bearish = option_type == OptionType.PUT and random.random() < 0.65
        bullish = option_type == OptionType.CALL and random.random() < 0.65
        sentiment = (
            FlowSentiment.BEARISH if bearish
            else FlowSentiment.BULLISH if bullish
            else FlowSentiment.NEUTRAL
        )

        is_sweep = random.random() < 0.35
        is_block = (not is_sweep) and random.random() < 0.20

        return UnusualFlowSignal(
            signal_id=uuid.uuid4().hex[:12],
            ticker=ticker,
            expiration=expiry,
            strike=round(strike, 2),
            option_type=option_type,
            premium_usd=round(premium_usd, 2),
            volume=volume,
            open_interest=random.randint(volume, volume * 15),
            spot_price_at_trade=round(spot, 2),
            implied_volatility=round(iv, 4),
            delta=round(delta, 4),
            sentiment=sentiment,
            is_sweep=is_sweep,
            is_block=is_block,
            exchange=random.choice(self._EXCHANGES),
            timestamp_utc=datetime.utcnow(),
            source="mock_unusual_whales",
        )
