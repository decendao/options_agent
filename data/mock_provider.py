"""
data/mock_provider.py — Deterministic Mock Data Provider
=========================================================
Generates realistic synthetic options chain data for:
  - Local development and testing without API keys
  - CI/CD pipeline validation
  - System integration testing

Data is deterministically generated using Black-Scholes with
configurable spot, IV, and Greeks perturbations.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from typing import Optional

from options_agent.core.schemas import (
    DataQuality,
    OptionContract,
    OptionsChain,
    OptionType,
    PredictionMarketContract,
    SpotQuote,
)
from options_agent.core.greeks_engine import bs_delta, bs_gamma, bs_theta, bs_vega
from options_agent.data.base_provider import BaseMarketDataProvider
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)

# Realistic mock spot prices for common tickers
_MOCK_SPOTS: dict[str, float] = {
    "SPY": 578.50,
    "QQQ": 495.20,
    "AAPL": 213.40,
    "NVDA": 134.20,
    "TSLA": 285.00,
    "MSFT": 455.80,
    "AMZN": 220.50,
    "META": 630.10,
    "DEFAULT": 100.00,
}

# Realistic mock IV levels
_MOCK_ATM_IV: dict[str, float] = {
    "SPY": 0.14,
    "QQQ": 0.17,
    "AAPL": 0.22,
    "NVDA": 0.45,
    "TSLA": 0.52,
    "MSFT": 0.20,
    "DEFAULT": 0.25,
}


class MockMarketDataProvider(BaseMarketDataProvider):
    """
    Fully offline mock provider. Generates synthetic but structurally
    correct options chains with pre-computed Greeks.
    Useful for validating Agent B logic without live API calls.
    """

    def __init__(
        self,
        seed: int = 42,
        spot_override: Optional[dict[str, float]] = None,
        iv_override: Optional[dict[str, float]] = None,
        noise_pct: float = 0.002,   # ±0.2% price noise per cycle
    ):
        super().__init__("mock")
        self._rng = random.Random(seed)
        self._spot_override = spot_override or {}
        self._iv_override = iv_override or {}
        self._noise_pct = noise_pct
        self._cycle = 0
        self.r = 0.0525  # risk-free rate

    async def connect(self) -> None:
        logger.info("MockMarketDataProvider connected (offline mode)")

    async def close(self) -> None:
        pass

    def _get_spot(self, ticker: str) -> float:
        base = self._spot_override.get(ticker) or _MOCK_SPOTS.get(ticker, _MOCK_SPOTS["DEFAULT"])
        # Small deterministic walk each cycle
        noise = 1 + self._rng.gauss(0, self._noise_pct)
        return round(base * noise, 2)

    def _get_atm_iv(self, ticker: str) -> float:
        base = self._iv_override.get(ticker) or _MOCK_ATM_IV.get(ticker, _MOCK_ATM_IV["DEFAULT"])
        noise = 1 + self._rng.gauss(0, 0.02)
        return max(0.05, base * noise)

    async def fetch_spot_quote(self, ticker: str) -> SpotQuote:
        S = self._get_spot(ticker)
        spread = S * 0.0002
        return SpotQuote(
            ticker=ticker,
            spot_price=S,
            bid=round(S - spread, 2),
            ask=round(S + spread, 2),
            last=S,
            volume=self._rng.randint(5_000_000, 80_000_000),
            timestamp_utc=datetime.utcnow(),
            source="mock",
        )

    async def fetch_available_expirations(self, ticker: str) -> list[date]:
        today = date.today()
        expirations = []
        for days_out in [7, 14, 21, 30, 45, 60, 90]:
            exp = today + timedelta(days=days_out)
            # Roll to nearest Friday
            while exp.weekday() != 4:
                exp += timedelta(days=1)
            expirations.append(exp)
        return sorted(set(expirations))

    async def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[date] = None,
        moneyness_range_pct: float = 0.15,
    ) -> list[OptionsChain]:
        self._cycle += 1
        S = self._get_spot(ticker)
        atm_iv = self._get_atm_iv(ticker)
        expirations = await self.fetch_available_expirations(ticker)

        if expiration:
            expirations = [e for e in expirations if e == expiration] or [expirations[0]]

        chains = []
        for exp in expirations:
            T = max((exp - date.today()).days, 1) / 365.0
            chain = self._generate_chain(ticker, S, T, exp, atm_iv, moneyness_range_pct)
            chains.append(chain)

        return chains

    def _generate_chain(
        self,
        ticker: str,
        S: float,
        T: float,
        expiration: date,
        atm_iv: float,
        moneyness_range_pct: float,
    ) -> OptionsChain:
        """Generate a synthetic options chain with realistic strike spacing."""
        # Strike spacing: ~$1 for cheap stocks, $5 for mid, $10+ for expensive
        if S < 50:
            step = 1.0
        elif S < 200:
            step = 2.5
        elif S < 500:
            step = 5.0
        else:
            step = 10.0

        strike_lo = S * (1 - moneyness_range_pct)
        strike_hi = S * (1 + moneyness_range_pct)

        # Round to nearest step
        start = math.floor(strike_lo / step) * step
        end   = math.ceil(strike_hi / step) * step
        strikes = []
        k = start
        while k <= end + step / 2:
            strikes.append(round(k, 2))
            k += step

        contracts: list[OptionContract] = []
        now = datetime.utcnow()

        for strike in strikes:
            moneyness = strike / S

            # Volatility smile: higher IV for OTM options (realistic skew)
            # Put skew: OTM puts have higher IV
            if moneyness < 1.0:
                skew_adj = 0.03 * (1.0 - moneyness) * 4  # steeper for puts
            else:
                skew_adj = 0.01 * (moneyness - 1.0) * 2  # mild for calls

            iv = max(0.01, atm_iv + skew_adj + self._rng.gauss(0, 0.005))

            for opt_type in [OptionType.CALL, OptionType.PUT]:
                from options_agent.core.greeks_engine import bs_price, implied_vol_newton
                price = bs_price(S, strike, T, self.r, iv, opt_type)
                spread = max(0.01, price * 0.015)

                delta = bs_delta(S, strike, T, self.r, iv, opt_type)
                gamma = bs_gamma(S, strike, T, self.r, iv)
                theta = bs_theta(S, strike, T, self.r, iv, opt_type)
                vega  = bs_vega(S, strike, T, self.r, iv)

                # OI peaks near ATM, tapers off for deep OTM
                oi_base = 10_000
                dist = abs(strike - S) / S
                oi = max(100, int(oi_base * math.exp(-15 * dist) * self._rng.uniform(0.5, 1.5)))
                vol = max(10, int(oi * 0.1 * self._rng.uniform(0.3, 2.0)))

                contracts.append(
                    OptionContract(
                        ticker=ticker,
                        expiration=expiration,
                        strike=strike,
                        option_type=opt_type,
                        bid=round(max(0.01, price - spread / 2), 2),
                        ask=round(price + spread / 2, 2),
                        last=round(price, 2),
                        volume=vol,
                        open_interest=oi,
                        implied_volatility=iv,
                        delta=delta,
                        gamma=gamma,
                        theta=theta,
                        vega=vega,
                        timestamp_utc=now,
                    )
                )

        return OptionsChain(
            ticker=ticker,
            expiration=expiration,
            spot_price=S,
            contracts=contracts,
            timestamp_utc=now,
            data_quality=DataQuality.GOOD,
        )

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Mock Prediction Market Provider
# ---------------------------------------------------------------------------

class MockPredictionMarketProvider:
    """
    Generates synthetic prediction market contracts for testing.
    Creates one YES/NO market per watch ticker.
    """

    def __init__(self, seed: int = 99):
        self._rng = random.Random(seed)

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def fetch_markets(
        self,
        ticker_keywords: Optional[list[str]] = None,
    ) -> list[PredictionMarketContract]:
        if not ticker_keywords:
            return []

        contracts = []
        for ticker in ticker_keywords:
            spot = _MOCK_SPOTS.get(ticker, 100.0)
            # Probability that the stock closes above current spot + 5% by expiry
            yes_price = round(self._rng.uniform(0.20, 0.55), 3)

            contracts.append(
                PredictionMarketContract(
                    market_name="mock_polymarket",
                    contract_id=f"mock_{ticker}_above_{spot * 1.05:.0f}_30d",
                    question=f"Will {ticker} close above ${spot * 1.05:.0f} within 30 days?",
                    ticker_ref=ticker,
                    yes_price=yes_price,
                    no_price=round(1.0 - yes_price, 3),
                    volume_24h=round(self._rng.uniform(50_000, 2_000_000), 0),
                    timestamp_utc=datetime.utcnow(),
                )
            )

        return contracts
