"""
agents/agent_a.py — Data Provider & Context Assembler
======================================================
Responsibilities:
  - Poll/stream equity spot prices and option chains for all watch tickers
  - Fetch prediction market contracts
  - Normalize raw API responses into canonical Pydantic schemas
  - Measure data latency and log sanity metrics
  - Publish AgentAOutput onto the A→B async queue
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Optional

from options_agent.config import Settings
from options_agent.core.schemas import (
    AgentAOutput,
    CatalystSignal,
    DataQuality,
    OptionsChain,
    SpotQuote,
)
from options_agent.core.greeks_engine import enrich_chain_greeks
from options_agent.data.base_provider import BaseMarketDataProvider
from options_agent.data.prediction_market_provider import PredictionMarketAggregator
from options_agent.harness.circuit_breaker import (
    AsyncRateLimiter,
    CircuitBreaker,
    DataSanityGuard,
)
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)


class AgentA:
    """
    Data Provider & Context Assembler.

    Each cycle it:
    1. Fetches spot quotes for all watch_tickers (parallel)
    2. Fetches options chains for each ticker (parallel)
    3. Enriches chains with Greeks via BS engine
    4. Fetches prediction market contracts
    5. Resolves catalyst signals from the calendar
    6. Publishes AgentAOutput onto the output_queue
    """

    def __init__(
        self,
        settings: Settings,
        market_provider: BaseMarketDataProvider,
        prediction_aggregator: PredictionMarketAggregator,
        output_queue: asyncio.Queue,
        iv_history_store: Optional[dict[str, list[float]]] = None,
    ):
        self.settings             = settings
        self.market_provider      = market_provider
        self.prediction_aggregator = prediction_aggregator
        self.output_queue         = output_queue
        self.iv_history_store     = iv_history_store or {t: [] for t in settings.watch_tickers}

        # Per-ticker circuit breakers
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            ticker: CircuitBreaker(
                name=f"spot_{ticker}",
                max_failures=settings.risk_thresholds.max_consecutive_errors,
                reset_timeout_seconds=120.0,
            )
            for ticker in settings.watch_tickers
        }

        self._rate_limiter = AsyncRateLimiter(
            rate_per_minute=settings.market_data_provider.rate_limit_per_minute,
            provider_name=settings.market_data_provider.name,
        )
        self._sanity = DataSanityGuard(
            max_staleness_seconds=120.0,
            min_contracts_per_chain=5,
        )

        self._cycle_count = 0

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self) -> Optional[AgentAOutput]:
        """Execute one full data collection cycle. Returns None on hard failure."""
        self._cycle_count += 1
        cycle_id = f"A_{self._cycle_count:06d}_{uuid.uuid4().hex[:6]}"
        t_start = time.monotonic()

        logger.info(f"[Agent A] Cycle {cycle_id} starting — tickers: {self.settings.watch_tickers}")

        # ── 1. Fetch spot quotes (parallel) ──────────────────────────
        spot_quotes = await self._fetch_all_spots()

        # ── 2. Fetch options chains (parallel) ───────────────────────
        options_chains = await self._fetch_all_chains(spot_quotes)

        # ── 3. Enrich Greeks ─────────────────────────────────────────
        options_chains = self._enrich_greeks(options_chains)

        # ── 4. Prediction markets ────────────────────────────────────
        pred_contracts = await self._fetch_prediction_markets()

        # ── 5. Catalyst signals ──────────────────────────────────────
        catalyst_signals = self._resolve_catalyst_signals()

        # ── 6. Data quality summary ──────────────────────────────────
        quality_summary = self._assess_quality(spot_quotes, options_chains)

        latency_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            f"[Agent A] Cycle {cycle_id} complete | "
            f"latency={latency_ms:.0f}ms | "
            f"chains={sum(len(v) for v in options_chains.values())} | "
            f"pred_contracts={len(pred_contracts)}"
        )

        output = AgentAOutput(
            cycle_id=cycle_id,
            tickers_processed=list(spot_quotes.keys()),
            spot_quotes=spot_quotes,
            options_chains=options_chains,
            prediction_contracts=pred_contracts,
            catalyst_signals=catalyst_signals,
            latency_ms=latency_ms,
            data_quality_summary=quality_summary,
            timestamp_utc=datetime.utcnow(),
        )

        await self.output_queue.put(output)
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_all_spots(self) -> dict[str, SpotQuote]:
        """Fetch spot quotes for all watch_tickers in parallel."""
        tasks = {
            ticker: self._fetch_spot_safe(ticker)
            for ticker in self.settings.watch_tickers
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        quotes: dict[str, SpotQuote] = {}
        for ticker, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"[Agent A] Spot fetch failed for {ticker}: {result}")
            else:
                quotes[ticker] = result
        return quotes

    async def _fetch_spot_safe(self, ticker: str) -> SpotQuote:
        await self._rate_limiter.acquire()
        cb = self._circuit_breakers.get(ticker)
        if cb:
            return await cb.call(self.market_provider.fetch_spot_quote(ticker))
        return await self.market_provider.fetch_spot_quote(ticker)

    async def _fetch_all_chains(
        self, spot_quotes: dict[str, SpotQuote]
    ) -> dict[str, list[OptionsChain]]:
        """Fetch options chains for all tickers in parallel."""
        tasks = {
            ticker: self._fetch_chain_safe(ticker)
            for ticker in self.settings.watch_tickers
            if ticker in spot_quotes  # only if spot succeeded
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        chains: dict[str, list[OptionsChain]] = {}
        for ticker, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"[Agent A] Chain fetch failed for {ticker}: {result}")
                chains[ticker] = []
            else:
                chains[ticker] = result
        return chains

    async def _fetch_chain_safe(self, ticker: str) -> list[OptionsChain]:
        await self._rate_limiter.acquire(tokens=3.0)  # chains cost more quota
        return await self.market_provider.fetch_options_chain(
            ticker=ticker,
            moneyness_range_pct=self.settings.option_moneyness_range_pct,
        )

    def _enrich_greeks(
        self, chains: dict[str, list[OptionsChain]]
    ) -> dict[str, list[OptionsChain]]:
        """Run BS Greek enrichment on all chains (synchronous but fast)."""
        r = self.settings.risk_free_rate
        enriched: dict[str, list[OptionsChain]] = {}
        for ticker, chain_list in chains.items():
            enriched[ticker] = []
            for chain in chain_list:
                # Only enrich chains where Greeks are missing
                needs_enrichment = any(
                    c.gamma is None or c.implied_volatility is None
                    for c in chain.contracts[:5]  # sample first 5
                )
                if needs_enrichment:
                    try:
                        chain = enrich_chain_greeks(chain, r)
                    except Exception as e:
                        logger.warning(f"Greek enrichment failed for {ticker}/{chain.expiration}: {e}")
                enriched[ticker].append(chain)
        return enriched

    async def _fetch_prediction_markets(self):
        try:
            return await self.prediction_aggregator.fetch_all(
                watch_tickers=self.settings.watch_tickers
            )
        except Exception as e:
            logger.warning(f"[Agent A] Prediction market fetch failed: {e}")
            return []

    def _resolve_catalyst_signals(self) -> list[CatalystSignal]:
        """Map catalyst calendar entries to CatalystSignal objects with hours_until_event."""
        signals = []
        now = datetime.utcnow()
        for event in self.settings.catalyst_calendar:
            try:
                event_dt = datetime.fromisoformat(
                    event.event_time_utc.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                hours_until = (event_dt - now).total_seconds() / 3600.0
                is_imminent = 0 <= hours_until <= self.settings.risk_thresholds.iv_crush_hours_before_event
                signals.append(
                    CatalystSignal(
                        ticker=event.ticker,
                        event_type=event.event_type,
                        event_time_utc=event_dt,
                        description=event.description,
                        hours_until_event=max(hours_until, 0),
                        is_imminent=is_imminent,
                    )
                )
            except Exception as e:
                logger.debug(f"Catalyst parse error: {e}")
        return signals

    def _assess_quality(
        self,
        spot_quotes: dict[str, SpotQuote],
        options_chains: dict[str, list[OptionsChain]],
    ) -> dict[str, DataQuality]:
        quality: dict[str, DataQuality] = {}
        for ticker in self.settings.watch_tickers:
            if ticker not in spot_quotes:
                quality[ticker] = DataQuality.INVALID
                continue

            quote = spot_quotes[ticker]
            fresh, _ = self._sanity.check_quote_freshness(quote.timestamp_utc)
            if not fresh:
                quality[ticker] = DataQuality.STALE
                continue

            chains = options_chains.get(ticker, [])
            total_contracts = sum(len(c.contracts) for c in chains)
            ok, _ = self._sanity.check_chain_completeness(total_contracts, required=10)
            quality[ticker] = DataQuality.GOOD if ok else DataQuality.SPARSE

        return quality
