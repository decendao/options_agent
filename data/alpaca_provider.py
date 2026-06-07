"""
data/alpaca_provider.py — Alpaca Markets Data Provider
=======================================================
Implements BaseMarketDataProvider for Alpaca's REST + WebSocket APIs.
Handles:
  - Stock quotes via /v2/stocks/{ticker}/quotes/latest
  - Options chains via /v2/options/snapshots
  - WebSocket quote streaming via wss://stream.data.alpaca.markets/v2/iex
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from typing import AsyncIterator, Optional

import aiohttp

from options_agent.core.schemas import (
    DataQuality,
    OptionContract,
    OptionsChain,
    OptionType,
    SpotQuote,
)
from options_agent.data.base_provider import BaseMarketDataProvider
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)


class AlpacaProvider(BaseMarketDataProvider):
    """
    Alpaca Markets data provider.
    Requires ALPACA_API_KEY and ALPACA_API_SECRET environment variables.
    Supports both paper and live endpoints.
    """

    DATA_BASE_URL  = "https://data.alpaca.markets/v2"
    TRADE_BASE_URL = "https://paper-api.alpaca.markets/v2"
    WS_URL         = "wss://stream.data.alpaca.markets/v2/iex"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = DATA_BASE_URL,
        ws_url: str = WS_URL,
        lookahead_days: int = 45,
        moneyness_range_pct: float = 0.15,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__("alpaca", session)
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base_url   = base_url.rstrip("/")
        self.ws_url     = ws_url
        self.lookahead_days     = lookahead_days
        self.moneyness_range_pct = moneyness_range_pct

        self._auth_headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

    # ------------------------------------------------------------------
    # Spot quotes
    # ------------------------------------------------------------------

    async def fetch_spot_quote(self, ticker: str) -> SpotQuote:
        url = f"{self.base_url}/stocks/{ticker}/quotes/latest"
        data = await self._get(url, headers=self._auth_headers)

        q = data.get("quote", {})
        bars = data.get("bar", {})  # fallback last price from bars

        # Alpaca quote fields: ap (ask price), bp (bid price), t (timestamp)
        ask   = float(q.get("ap", 0) or 0)
        bid   = float(q.get("bp", 0) or 0)
        last  = float(bars.get("c", 0) or (bid + ask) / 2 or 0)
        ts_str = q.get("t", datetime.utcnow().isoformat())

        return SpotQuote(
            ticker=ticker,
            spot_price=last or (bid + ask) / 2,
            bid=bid,
            ask=ask,
            last=last,
            volume=int(bars.get("v", 0) or 0),
            timestamp_utc=datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None),
            source="alpaca",
        )

    # ------------------------------------------------------------------
    # Options chain
    # ------------------------------------------------------------------

    async def fetch_available_expirations(self, ticker: str) -> list[date]:
        url = f"{self.base_url}/options/contracts"
        params = {
            "underlying_symbol": ticker,
            "expiration_date_gte": date.today().isoformat(),
            "expiration_date_lte": (date.today() + timedelta(days=self.lookahead_days)).isoformat(),
            "limit": 1000,
        }
        data = await self._get(url, params=params, headers=self._auth_headers)
        contracts = data.get("option_contracts", [])
        expirations = sorted({
            date.fromisoformat(c["expiration_date"]) for c in contracts
        })
        return expirations

    async def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[date] = None,
        moneyness_range_pct: float = 0.15,
    ) -> list[OptionsChain]:
        # First get spot to filter strikes
        try:
            spot = await self.fetch_spot_quote(ticker)
            S = spot.spot_price
        except Exception:
            S = 0.0  # can't filter strikes — fetch all

        exp_gte = date.today().isoformat()
        exp_lte = (
            expiration.isoformat() if expiration
            else (date.today() + timedelta(days=self.lookahead_days)).isoformat()
        )

        url = f"{self.base_url}/options/snapshots"
        params: dict = {
            "underlying_symbols": ticker,
            "expiration_date_gte": exp_gte,
            "expiration_date_lte": exp_lte,
            "limit": 1000,
        }
        if S > 0 and moneyness_range_pct > 0:
            params["strike_price_gte"] = str(round(S * (1 - moneyness_range_pct), 2))
            params["strike_price_lte"] = str(round(S * (1 + moneyness_range_pct), 2))

        data = await self._get(url, params=params, headers=self._auth_headers)
        snapshots = data.get("snapshots", {})

        # Group by expiration
        chains_by_expiry: dict[date, list[OptionContract]] = {}
        now = datetime.utcnow()

        for symbol, snap in snapshots.items():
            details = snap.get("latestQuote", {})
            greeks  = snap.get("greeks", {})
            meta    = snap.get("details", {})

            try:
                exp = date.fromisoformat(meta.get("expiration_date", ""))
                strike = float(meta.get("strike_price", 0))
                opt_type = OptionType.CALL if meta.get("type", "").lower() == "call" else OptionType.PUT
            except (ValueError, KeyError):
                continue

            contract = OptionContract(
                ticker=ticker,
                expiration=exp,
                strike=strike,
                option_type=opt_type,
                bid=float(details.get("bp", 0) or 0),
                ask=float(details.get("ap", 0) or 0),
                last=float(snap.get("latestTrade", {}).get("p", 0) or 0),
                volume=int(snap.get("dailyBar", {}).get("v", 0) or 0),
                open_interest=int(meta.get("open_interest", 0) or 0),
                implied_volatility=float(greeks.get("impliedVolatility", 0) or 0) or None,
                delta=float(greeks.get("delta", 0) or 0) or None,
                gamma=float(greeks.get("gamma", 0) or 0) or None,
                theta=float(greeks.get("theta", 0) or 0) or None,
                vega=float(greeks.get("vega", 0)  or 0) or None,
                timestamp_utc=now,
            )
            chains_by_expiry.setdefault(exp, []).append(contract)

        # Build OptionsChain objects
        chains = []
        for exp, contracts in chains_by_expiry.items():
            quality = DataQuality.GOOD if len(contracts) >= 10 else DataQuality.SPARSE
            chains.append(
                OptionsChain(
                    ticker=ticker,
                    expiration=exp,
                    spot_price=S,
                    contracts=contracts,
                    timestamp_utc=now,
                    data_quality=quality,
                )
            )
        return sorted(chains, key=lambda c: c.expiration)

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    async def stream_quotes(self, tickers: list[str]) -> AsyncIterator[SpotQuote]:
        """
        Stream real-time quotes via Alpaca WebSocket.
        Yields SpotQuote objects as they arrive.
        """
        auth_msg = json.dumps({
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret,
        })
        subscribe_msg = json.dumps({
            "action": "subscribe",
            "quotes": tickers,
        })

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url) as ws:
                # Auth handshake
                await ws.send_str(auth_msg)
                await asyncio.sleep(0.5)
                await ws.send_str(subscribe_msg)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        events = json.loads(msg.data)
                        for event in events:
                            if event.get("T") == "q":  # quote event
                                try:
                                    yield SpotQuote(
                                        ticker=event["S"],
                                        spot_price=(float(event.get("ap", 0)) + float(event.get("bp", 0))) / 2,
                                        bid=float(event.get("bp", 0)),
                                        ask=float(event.get("ap", 0)),
                                        last=float(event.get("ap", 0)),
                                        volume=int(event.get("bs", 0)),
                                        timestamp_utc=datetime.utcnow(),
                                        source="alpaca_ws",
                                    )
                                except Exception as e:
                                    logger.warning(f"Alpaca WS parse error: {e}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Alpaca WebSocket error: {ws.exception()}")
                        break

    async def health_check(self) -> bool:
        try:
            url = f"https://api.alpaca.markets/v2/account"
            data = await self._get(url, headers=self._auth_headers)
            return "id" in data
        except Exception:
            return False
