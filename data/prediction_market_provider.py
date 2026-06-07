"""
data/prediction_market_provider.py — Prediction Market Data Provider
=====================================================================
Fetches binary event contract data from Polymarket and Kalshi.
Maps prediction market questions to underlying equity tickers
for cross-market EV comparison.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Optional

import aiohttp

from options_agent.core.schemas import PredictionMarketContract
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Polymarket Provider
# ---------------------------------------------------------------------------

class PolymarketProvider:
    """
    Fetches contracts from Polymarket's CLOB (Central Limit Order Book) API.
    Public API — no auth required for reads.
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None

    async def connect(self) -> None:
        if self._owns_session and self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15.0)
            )

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def fetch_markets(
        self,
        ticker_keywords: Optional[list[str]] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[PredictionMarketContract]:
        """
        Fetch markets from Polymarket CLOB.
        Optionally filter by ticker keywords (e.g. ["SPY", "S&P", "500"]).
        """
        url = f"{self.base_url}/markets"
        params: dict = {"limit": limit}
        if active_only:
            params["active"] = "true"

        try:
            async with self._session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:
            logger.warning(f"Polymarket fetch failed: {e}")
            return []

        markets = data if isinstance(data, list) else data.get("data", [])
        contracts = []

        for market in markets:
            question = market.get("question", "")
            if ticker_keywords:
                if not any(kw.upper() in question.upper() for kw in ticker_keywords):
                    continue

            ticker_ref = _infer_ticker_from_question(question)
            if not ticker_ref:
                continue

            tokens = market.get("tokens", [])
            yes_price = no_price = 0.5
            for token in tokens:
                outcome = token.get("outcome", "").lower()
                price = float(token.get("price", 0.5) or 0.5)
                if outcome == "yes":
                    yes_price = price
                elif outcome == "no":
                    no_price = price

            try:
                contracts.append(
                    PredictionMarketContract(
                        market_name="polymarket",
                        contract_id=str(market.get("condition_id", market.get("id", ""))),
                        question=question,
                        ticker_ref=ticker_ref,
                        yes_price=yes_price,
                        no_price=no_price,
                        volume_24h=float(market.get("volume24hr", 0) or 0),
                        timestamp_utc=datetime.utcnow(),
                    )
                )
            except Exception as e:
                logger.debug(f"Polymarket contract parse error: {e}")

        return contracts


# ---------------------------------------------------------------------------
# Kalshi Provider
# ---------------------------------------------------------------------------

class KalshiProvider:
    """
    Fetches markets from Kalshi's public API v2.
    Kalshi requires email+password auth for trading; read-only market data is public.
    """

    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

    def __init__(
        self,
        base_url: str = BASE_URL,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._owns_session = session is None

    async def connect(self) -> None:
        if self._owns_session and self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15.0)
            )

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def fetch_markets(
        self,
        ticker_keywords: Optional[list[str]] = None,
        series_ticker: Optional[str] = None,
        limit: int = 200,
    ) -> list[PredictionMarketContract]:
        """Fetch active markets from Kalshi."""
        url = f"{self.base_url}/markets"
        params: dict = {"limit": limit, "status": "open"}
        if series_ticker:
            params["series_ticker"] = series_ticker

        try:
            async with self._session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:
            logger.warning(f"Kalshi fetch failed: {e}")
            return []

        markets = data.get("markets", [])
        contracts = []

        for market in markets:
            title = market.get("title", "")
            if ticker_keywords:
                if not any(kw.upper() in title.upper() for kw in ticker_keywords):
                    continue

            ticker_ref = _infer_ticker_from_question(title)
            if not ticker_ref:
                continue

            yes_bid = float(market.get("yes_bid", 50) or 50) / 100.0
            yes_ask = float(market.get("yes_ask", 50) or 50) / 100.0
            yes_mid = (yes_bid + yes_ask) / 2.0

            try:
                contracts.append(
                    PredictionMarketContract(
                        market_name="kalshi",
                        contract_id=str(market.get("ticker", "")),
                        question=title,
                        ticker_ref=ticker_ref,
                        yes_price=yes_mid,
                        no_price=1.0 - yes_mid,
                        volume_24h=float(market.get("volume", 0) or 0),
                        timestamp_utc=datetime.utcnow(),
                    )
                )
            except Exception as e:
                logger.debug(f"Kalshi contract parse error: {e}")

        return contracts


# ---------------------------------------------------------------------------
# Unified Prediction Market Aggregator
# ---------------------------------------------------------------------------

class PredictionMarketAggregator:
    """
    Aggregates contracts from multiple prediction markets.
    Deduplicates by (ticker_ref, question) and prefers highest-volume contract.
    """

    def __init__(
        self,
        providers: Optional[list] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self._session = session
        self.providers = providers or []
        if not self.providers:
            self.providers = [
                PolymarketProvider(session=session),
                KalshiProvider(session=session),
            ]

    async def connect(self) -> None:
        for p in self.providers:
            await p.connect()

    async def close(self) -> None:
        for p in self.providers:
            await p.close()

    async def fetch_all(
        self,
        watch_tickers: list[str],
    ) -> list[PredictionMarketContract]:
        """
        Fetch and merge contracts from all providers for the given tickers.
        Returns contracts sorted by volume descending.
        """
        tasks = []
        for provider in self.providers:
            tasks.append(
                provider.fetch_markets(ticker_keywords=watch_tickers)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_contracts: list[PredictionMarketContract] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Prediction market provider error: {result}")
                continue
            all_contracts.extend(result)

        # Deduplicate: keep highest-volume per (ticker_ref, market_name, contract_id)
        seen: dict[tuple, PredictionMarketContract] = {}
        for c in all_contracts:
            key = (c.ticker_ref, c.market_name, c.contract_id)
            if key not in seen or c.volume_24h > seen[key].volume_24h:
                seen[key] = c

        return sorted(seen.values(), key=lambda c: c.volume_24h, reverse=True)


# ---------------------------------------------------------------------------
# Ticker inference from question text
# ---------------------------------------------------------------------------

# Map of keywords → canonical ticker
_KEYWORD_TICKER_MAP: dict[str, str] = {
    "S&P 500": "SPY",
    "SPY": "SPY",
    "S&P500": "SPY",
    "NASDAQ": "QQQ",
    "QQQ": "QQQ",
    "DOW JONES": "DIA",
    "DJIA": "DIA",
    "APPLE": "AAPL",
    "AAPL": "AAPL",
    "NVIDIA": "NVDA",
    "NVDA": "NVDA",
    "TESLA": "TSLA",
    "TSLA": "TSLA",
    "MICROSOFT": "MSFT",
    "MSFT": "MSFT",
    "AMAZON": "AMZN",
    "AMZN": "AMZN",
    "FED": "MACRO",
    "FOMC": "MACRO",
    "FEDERAL RESERVE": "MACRO",
    "CPI": "MACRO",
    "INTEREST RATE": "MACRO",
    "VIX": "VIX",
}


def _infer_ticker_from_question(question: str) -> Optional[str]:
    """
    Heuristically infer the underlying ticker from a question string.
    Returns the matched canonical ticker or None if no match.
    """
    q_upper = question.upper()
    for keyword, ticker in sorted(_KEYWORD_TICKER_MAP.items(), key=lambda x: -len(x[0])):
        if keyword in q_upper:
            return ticker

    # Fallback: look for 1-5 letter all-caps token that looks like a ticker
    tokens = re.findall(r'\b[A-Z]{1,5}\b', question)
    if tokens:
        # Filter out common English words
        stopwords = {"WILL", "THE", "AND", "FOR", "ARE", "NOT", "BY", "AT", "OR", "TO", "IN", "ON"}
        candidates = [t for t in tokens if t not in stopwords and len(t) >= 2]
        if candidates:
            return candidates[0]

    return None
