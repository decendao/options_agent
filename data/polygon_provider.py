"""
data/polygon_provider.py — Polygon.io Data Provider
=====================================================
Supplementary provider used primarily for:
  - Options chain snapshots (/v3/snapshot/options/{ticker})
  - Historical IV data for IV Rank computation
  - Tick-level trade data for volume analysis
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional

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


class PolygonProvider(BaseMarketDataProvider):
    """
    Polygon.io data provider. Requires POLYGON_API_KEY.
    Polygon options snapshot returns pre-computed Greeks and IV.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        lookahead_days: int = 45,
        moneyness_range_pct: float = 0.15,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__("polygon", session)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.lookahead_days = lookahead_days
        self.moneyness_range_pct = moneyness_range_pct

    # ------------------------------------------------------------------
    # Spot quotes
    # ------------------------------------------------------------------

    async def fetch_spot_quote(self, ticker: str) -> SpotQuote:
        url = f"{self.base_url}/v2/last/nbbo/{ticker}"
        data = await self._get(url, params={"apiKey": self.api_key})

        result = data.get("results", {})
        bid = float(result.get("P", 0) or 0)
        ask = float(result.get("p", 0) or 0)
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0

        # Fallback to previous close if NBBO unavailable
        if mid == 0:
            url2 = f"{self.base_url}/v2/aggs/ticker/{ticker}/prev"
            data2 = await self._get(url2, params={"apiKey": self.api_key})
            result2 = (data2.get("results") or [{}])[0]
            mid = float(result2.get("c", 0) or 0)
            bid = float(result2.get("l", mid))
            ask = float(result2.get("h", mid))

        return SpotQuote(
            ticker=ticker,
            spot_price=mid,
            bid=bid,
            ask=ask,
            last=mid,
            volume=int(result.get("s", 0) or 0),
            timestamp_utc=datetime.utcnow(),
            source="polygon",
        )

    # ------------------------------------------------------------------
    # Options chain via /v3/snapshot/options
    # ------------------------------------------------------------------

    async def fetch_available_expirations(self, ticker: str) -> list[date]:
        # Polygon doesn't have a dedicated expirations endpoint; derive from snapshot
        chains = await self.fetch_options_chain(ticker)
        return sorted({c.expiration for c in chains})

    async def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[date] = None,
        moneyness_range_pct: float = 0.15,
    ) -> list[OptionsChain]:
        url = f"{self.base_url}/v3/snapshot/options/{ticker}"
        params: dict = {
            "apiKey": self.api_key,
            "limit": 250,
        }
        if expiration:
            params["expiration_date"] = expiration.isoformat()
        else:
            params["expiration_date.gte"] = date.today().isoformat()
            params["expiration_date.lte"] = (
                date.today() + timedelta(days=self.lookahead_days)
            ).isoformat()

        all_results = []
        next_url: Optional[str] = None

        # Paginate through all results
        while True:
            fetch_url = next_url or url
            fetch_params = None if next_url else params
            data = await self._get(fetch_url, params=fetch_params)
            results = data.get("results", [])
            all_results.extend(results)
            next_url = data.get("next_url")
            if not next_url or not results:
                break

        # Get spot for moneyness filtering
        try:
            spot = await self.fetch_spot_quote(ticker)
            S = spot.spot_price
        except Exception:
            S = 0.0

        now = datetime.utcnow()
        chains_by_expiry: dict[date, list[OptionContract]] = {}

        for snap in all_results:
            details = snap.get("details", {})
            greeks  = snap.get("greeks", {})
            day     = snap.get("day", {})

            try:
                exp    = date.fromisoformat(details.get("expiration_date", ""))
                strike = float(details.get("strike_price", 0))
                opt_type = OptionType.CALL if details.get("contract_type", "").lower() == "call" \
                           else OptionType.PUT
            except (ValueError, TypeError, KeyError):
                continue

            # Moneyness filter
            if S > 0 and moneyness_range_pct > 0:
                if not (S * (1 - moneyness_range_pct) <= strike <= S * (1 + moneyness_range_pct)):
                    continue

            iv_raw = snap.get("implied_volatility") or greeks.get("vega") and None
            try:
                iv = float(iv_raw) if iv_raw is not None else None
            except (TypeError, ValueError):
                iv = None

            contract = OptionContract(
                ticker=ticker,
                expiration=exp,
                strike=strike,
                option_type=opt_type,
                bid=float((snap.get("last_quote") or {}).get("bid", 0) or 0),
                ask=float((snap.get("last_quote") or {}).get("ask", 0) or 0),
                last=float((snap.get("last_trade") or {}).get("price", 0) or 0),
                volume=int(day.get("volume", 0) or 0),
                open_interest=int(snap.get("open_interest", 0) or 0),
                implied_volatility=iv,
                delta=_safe_float(greeks.get("delta")),
                gamma=_safe_float(greeks.get("gamma")),
                theta=_safe_float(greeks.get("theta")),
                vega=_safe_float(greeks.get("vega")),
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
                    spot_price=S,
                    contracts=contracts,
                    timestamp_utc=now,
                    data_quality=quality,
                )
            )
        return sorted(chains, key=lambda c: c.expiration)

    # ------------------------------------------------------------------
    # Historical IV series (for IV Rank computation)
    # ------------------------------------------------------------------

    async def fetch_historical_iv(
        self,
        ticker: str,
        days: int = 252,
    ) -> list[float]:
        """
        Approximate IV history using historical daily close prices via
        Polygon's aggregate endpoint. Returns realized vol as a proxy
        when IV history is unavailable.
        """
        to_date   = date.today()
        from_date = to_date - timedelta(days=days + 10)
        url = (
            f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day"
            f"/{from_date.isoformat()}/{to_date.isoformat()}"
        )
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 365,
        }
        data = await self._get(url, params=params)
        results = data.get("results", [])
        if len(results) < 20:
            return []

        closes = [float(r["c"]) for r in results]
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]

        # 21-day rolling realized vol as IV proxy
        rv_series = []
        window = 21
        for i in range(window, len(returns)):
            window_returns = returns[i - window:i]
            rv = (sum(r**2 for r in window_returns) / window) ** 0.5 * (252**0.5)
            rv_series.append(rv)

        return rv_series

    async def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/v2/aggs/ticker/AAPL/prev"
            data = await self._get(url, params={"apiKey": self.api_key})
            return data.get("status") == "OK"
        except Exception:
            return False


def _safe_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
