"""
data/alpaca_provider.py — Alpaca Markets Data Provider
=====================================================
Implements BaseMarketDataProvider for Alpaca's REST + WebSocket APIs.

Separates spot (quotes) and options (chains) into independent HTTP clients
with their own rate limiters to respect Alpaca's 15 req/s tier limit.

Batch Strategy (Alpaca Free Tier: 15 req/s):
  - Each batch: 5 tickers × (1 spot + 1 options) = 10 req
  - 10 req / 15 req/s ≈ 0.67s per batch
  - With 1.25s batch_interval → ample headroom below rate limit
  - 20 tickers = 4 batches = ~5s total (fits in 5s cycle)

Data flows:
  Spot API:    /v2/stocks/{ticker}/quotes/latest
  Options API: /v2/options/snapshots
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


# ---------------------------------------------------------------------------
# AlpacaProvider
# ---------------------------------------------------------------------------

class AlpacaProvider(BaseMarketDataProvider):
    """
    Alpaca Markets data provider with SEPARATE spot and options HTTP sessions.

    Each data type has its own:
      - aiohttp.ClientSession
      - AsyncRateLimiter
      - base_url (can point to different endpoints if needed)

    This ensures spot quotes and options chains don't compete for the same
    15 req/s rate limit bucket.
    """

    SPOT_BASE_URL   = "https://data.alpaca.markets/v2"
    OPTIONS_BASE_URL = "https://data.alpaca.markets/v2"
    TRADE_BASE_URL   = "https://paper-api.alpaca.markets/v2"
    WS_URL           = "wss://stream.data.alpaca.markets/v2/iex"

    def __init__(
        self,
        spot_api_key: str = "",
        spot_api_secret: str = "",
        options_api_key: str = "",
        options_api_secret: str = "",
        spot_base_url: str = "",
        options_base_url: str = "",
        spot_rate_limit_per_minute: int = 900,
        options_rate_limit_per_minute: int = 900,
        lookahead_days: int = 45,
        moneyness_range_pct: float = 0.15,
        batch_interval_seconds: float = 1.25,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__("alpaca", session)

        self._spot_api_key    = spot_api_key
        self._spot_api_secret = spot_api_secret
        self._options_api_key    = options_api_key
        self._options_api_secret = options_api_secret

        self._spot_base_url    = (spot_base_url    or self.SPOT_BASE_URL).rstrip("/")
        self._options_base_url = (options_base_url or self.OPTIONS_BASE_URL).rstrip("/")

        self._spot_rate_limit        = spot_rate_limit_per_minute
        self._options_rate_limit    = options_rate_limit_per_minute
        self._lookahead_days        = lookahead_days
        self._moneyness_range_pct   = moneyness_range_pct
        self._batch_interval        = batch_interval_seconds

        # Separate sessions (initialized in connect())
        self._spot_session:    Optional[aiohttp.ClientSession] = None
        self._options_session:  Optional[aiohttp.ClientSession] = None

        # Separate rate limiters
        self._spot_limiter:    Optional[asyncio.Semaphore] = None
        self._options_limiter: Optional[asyncio.Semaphore] = None

        self._spot_auth_headers = {
            "APCA-API-KEY-ID":     self._spot_api_key,
            "APCA-API-SECRET-KEY": self._spot_api_secret,
        }
        self._options_auth_headers = {
            "APCA-API-KEY-ID":     self._options_api_key,
            "APCA-API-SECRET-KEY": self._options_api_secret,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await super().connect()
        # Create separate sessions for spot and options
        self._spot_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15.0),
            connector=aiohttp.TCPConnector(limit=20, limit_per_host=10),
        )
        self._options_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15.0),
            connector=aiohttp.TCPConnector(limit=20, limit_per_host=10),
        )
        # Semaphore-based rate limiter: 1 token = 1 request
        self._spot_limiter    = asyncio.Semaphore(self._spot_rate_limit // 60)
        self._options_limiter = asyncio.Semaphore(self._options_rate_limit // 60)
        logger.info(
            f"AlpacaProvider: separate spot/options sessions initialized. "
            f"Spot rate={self._spot_rate_limit}/min, Options rate={self._options_rate_limit}/min"
        )

    async def close(self) -> None:
        await super().close()
        if self._spot_session and not self._spot_session.closed:
            await self._spot_session.close()
        if self._options_session and not self._options_session.closed:
            await self._options_session.close()

    # ------------------------------------------------------------------
    # Rate-limited HTTP GET
    # ------------------------------------------------------------------

    async def _spot_get(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> dict:
        """GET from spot endpoint with rate limiting."""
        if self._spot_limiter is None:
            raise RuntimeError("Provider not connected. Call connect() first.")
        async with self._spot_limiter:
            async with self._spot_session.get(url, params=params, headers=self._spot_auth_headers) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _options_get(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> dict:
        """GET from options endpoint with rate limiting."""
        if self._options_limiter is None:
            raise RuntimeError("Provider not connected. Call connect() first.")
        async with self._options_limiter:
            async with self._options_session.get(url, params=params, headers=self._options_auth_headers) as resp:
                resp.raise_for_status()
                return await resp.json()

    # ------------------------------------------------------------------
    # Spot quotes — BATCHED
    # ------------------------------------------------------------------

    async def fetch_spot_quote(self, ticker: str) -> SpotQuote:
        """
        Fetch a single ticker's spot quote.
        For batched fetching across multiple tickers, use fetch_spot_quotes_batch().
        """
        result = await self.fetch_spot_quotes_batch([ticker])
        if ticker in result:
            return result[ticker]
        raise ValueError(f"No quote returned for {ticker}")

    async def fetch_spot_quotes_batch(
        self,
        tickers: list[str],
        batch_interval: Optional[float] = None,
    ) -> dict[str, SpotQuote]:
        """
        Fetch spot quotes for multiple tickers in BATCHES with rate limiting.

        Strategy:
          - Semaphore allows up to rate_limit_per_minute/60 req/s continuously
          - 20 tickers → 20 req → at 15 req/s = ~1.3s
          - With batch_interval=1.25s there's safe headroom

        Returns: {ticker: SpotQuote}
        """
        if not tickers:
            return {}
        interval = batch_interval if batch_interval is not None else self._batch_interval
        quotes: dict[str, SpotQuote] = {}

        for i, ticker in enumerate(tickers):
            try:
                quote = await self._fetch_spot_single(ticker)
                if quote:
                    quotes[ticker] = quote
            except Exception as e:
                logger.warning(f"[Alpaca] Spot fetch failed for {ticker}: {e}")

            # Rate limit: 1 request per tick (rate_limit/60 per second)
            # Also inter-batch pause
            if i < len(tickers) - 1:
                # Control burst: (1 req) / (rate_limit/60) seconds
                pause = 1.0 / (self._spot_rate_limit / 60.0)
                await asyncio.sleep(pause)

        return quotes

    async def _fetch_spot_single(self, ticker: str) -> Optional[SpotQuote]:
        url = f"{self._spot_base_url}/stocks/{ticker}/quotes/latest"
        data = await self._spot_get(url)
        q = data.get("quote", {})
        bars = data.get("bar", {})

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
            source="alpaca_spot",
        )

    # ------------------------------------------------------------------
    # Options chains — BATCHED
    # ------------------------------------------------------------------

    async def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[date] = None,
        moneyness_range_pct: float = 0.15,
    ) -> list[OptionsChain]:
        """
        Fetch options chains for a single ticker.
        For batched fetching, use fetch_options_chains_batch().
        """
        result = await self.fetch_options_chains_batch([ticker], expiration, moneyness_range_pct)
        if ticker in result:
            return result[ticker]
        return []

    async def fetch_options_chains_batch(
        self,
        tickers: list[str],
        expiration: Optional[date] = None,
        moneyness_range_pct: Optional[float] = None,
        batch_interval: Optional[float] = None,
    ) -> dict[str, list[OptionsChain]]:
        """
        Fetch options chains for multiple tickers in BATCHES.

        Strategy:
          - Alpaca /v2/options/snapshots accepts multiple underlying_symbols in one call
          - But to stay within rate limits, we batch per ticker and pause between calls
          - Each ticker: 1 OPTIONS req → 1 req per pause cycle

        Returns: {ticker: [OptionsChain]}
        """
        if not tickers:
            return {}
        interval = batch_interval if batch_interval is not None else self._batch_interval
        moneyness = moneyness_range_pct if moneyness_range_pct is not None else self._moneyness_range_pct

        all_chains: dict[str, list[OptionsChain]] = {t: [] for t in tickers}

        for i, ticker in enumerate(tickers):
            try:
                spot_url = f"{self._spot_base_url}/stocks/{ticker}/quotes/latest"
                spot_data = await self._spot_get(spot_url)
                S = self._extract_spot_price(spot_data, ticker)
            except Exception as e:
                logger.warning(f"[Alpaca] Spot fetch failed for chain filter {ticker}: {e}")
                S = 0.0

            try:
                chains = await self._fetch_options_chain_single(
                    ticker, S, expiration, moneyness
                )
                if chains:
                    all_chains[ticker] = chains
            except Exception as e:
                logger.warning(f"[Alpaca] Options chain fetch failed for {ticker}: {e}")

            # Inter-batch pause
            if i < len(tickers) - 1:
                pause = 1.0 / (self._options_rate_limit / 60.0)
                await asyncio.sleep(pause)

        return all_chains

    def _extract_spot_price(self, data: dict, ticker: str) -> float:
        q = data.get("quote", {})
        bars = data.get("bar", {})
        ask = float(q.get("ap", 0) or 0)
        bid = float(q.get("bp", 0) or 0)
        return float(bars.get("c", 0) or (bid + ask) / 2 or 0)

    async def _fetch_options_chain_single(
        self,
        ticker: str,
        spot_price: float,
        expiration: Optional[date],
        moneyness_range_pct: float,
    ) -> list[OptionsChain]:
        exp_gte = date.today().isoformat()
        exp_lte = (
            expiration.isoformat()
            if expiration
            else (date.today() + timedelta(days=self._lookahead_days)).isoformat()
        )

        params: dict = {
            "underlying_symbols": ticker,
            "expiration_date_gte": exp_gte,
            "expiration_date_lte": exp_lte,
            "limit": 1000,
        }
        if spot_price > 0 and moneyness_range_pct > 0:
            params["strike_price_gte"] = str(round(spot_price * (1 - moneyness_range_pct), 2))
            params["strike_price_lte"] = str(round(spot_price * (1 + moneyness_range_pct), 2))

        url = f"{self._options_base_url}/options/snapshots"
        data = await self._options_get(url, params=params)
        snapshots = data.get("snapshots", {})

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
                vega=float(greeks.get("vega", 0) or 0) or None,
                timestamp_utc=now,
            )
            chains_by_expiry.setdefault(exp, []).append(contract)

        chains = []
        for exp, contracts in chains_by_expiry.items():
            quality = DataQuality.GOOD if len(contracts) >= 10 else DataQuality.SPARSE
            chains.append(
                OptionsChain(
                    ticker=ticker,
                    expiration=exp,
                    spot_price=spot_price,
                    contracts=contracts,
                    timestamp_utc=now,
                    data_quality=quality,
                )
            )
        return sorted(chains, key=lambda c: c.expiration)

    # ------------------------------------------------------------------
    # Available expirations
    # ------------------------------------------------------------------

    async def fetch_available_expirations(self, ticker: str) -> list[date]:
        url = f"{self._options_base_url}/options/contracts"
        params = {
            "underlying_symbol": ticker,
            "expiration_date_gte": date.today().isoformat(),
            "expiration_date_lte": (date.today() + timedelta(days=self._lookahead_days)).isoformat(),
            "limit": 1000,
        }
        data = await self._options_get(url, params=params)
        contracts = data.get("option_contracts", [])
        return sorted({date.fromisoformat(c["expiration_date"]) for c in contracts})

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------

    async def stream_quotes(self, tickers: list[str]) -> AsyncIterator[SpotQuote]:
        auth_msg = json.dumps({
            "action": "auth",
            "key": self._spot_api_key,
            "secret": self._spot_api_secret,
        })
        subscribe_msg = json.dumps({
            "action": "subscribe",
            "quotes": tickers,
        })

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.WS_URL) as ws:
                await ws.send_str(auth_msg)
                await asyncio.sleep(0.5)
                await ws.send_str(subscribe_msg)
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        events = json.loads(msg.data)
                        for event in events:
                            if event.get("T") == "q":
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
            headers = {
                "APCA-API-KEY-ID": self._spot_api_key,
                "APCA-API-SECRET-KEY": self._spot_api_secret,
            }
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, headers=headers) as resp:
                    data = await resp.json()
                    return "id" in data
        except Exception:
            return False
