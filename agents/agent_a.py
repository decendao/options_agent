"""
agents/agent_a.py — Data Provider & Context Assembler (Batched)
=============================================================
Key change:
  - Uses batched fetch methods (fetch_spot_quotes_batch / fetch_options_chains_batch)
  - Respects Alpaca's 15 req/s rate limit across batches
  - All tickers from watchlist.yaml or WATCH_TICKERS env var
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

    Fetches all tickers in BATCHES from the market provider.
    Alpaca Free Tier (15 req/s):
      - 20 tickers → 4 batches × 5 tickers
      - Each batch: 5 spot + 5 options ≈ 10 req
      - With 1.25s batch interval, well within limit
    """

    def __init__(
        self,
        settings: Settings,
        market_provider: BaseMarketDataProvider,
        prediction_aggregator: PredictionMarketAggregator,
        output_queue: asyncio.Queue,
        iv_history_store: Optional[dict[str, list[float]]] = None,
    ):
        self.settings = settings
        self.market_provider = market_provider
        self.prediction_aggregator = prediction_aggregator
        self.output_queue = output_queue
        self._all_tickers = settings.get_enabled_tickers()
        self.iv_history_store = iv_history_store or {t: [] for t in self._all_tickers}

        # Per-ticker circuit breakers
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            ticker: CircuitBreaker(
                name=f"spot_{ticker}",
                max_failures=settings.risk_thresholds.max_consecutive_errors,
                reset_timeout_seconds=120.0,
            )
            for ticker in self._all_tickers
        }

        self._rate_limiter = AsyncRateLimiter(
            rate_per_minute=settings.spot_data_provider.rate_limit_per_minute,
            provider_name="alpaca_spot",
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
        """Execute one full data collection cycle across all tickers in batches."""
        self._cycle_count += 1
        cycle_id = f"A_{self._cycle_count:06d}_{uuid.uuid4().hex[:6]}"
        t_start = time.monotonic()

        batches = self.settings.get_batches()
        logger.info(
            f"[Agent A] Cycle {cycle_id} starting | "
            f"{len(self._all_tickers)} tickers in {len(batches)} batches × "
            f"{self.settings.batch_size} = {self.settings.poll_interval_seconds}s cycle"
        )

        # ── 1. Fetch spot quotes (batched) ──────────────────────────
        spot_quotes = await self._fetch_all_spots_batched()

        # ── 2. Fetch options chains (batched) ───────────────────────
        options_chains = await self._fetch_all_chains_batched(spot_quotes)

        # ── 3. Enrich Greeks ─────────────────────────────────────────
        options_chains = self._enrich_greeks(options_chains)

        # ── 4. Prediction markets ───────────────────────────────────
        pred_contracts = await self._fetch_prediction_markets()

        # ── 5. Catalyst signals ─────────────────────────────────────
        catalyst_signals = self._resolve_catalyst_signals()

        # ── 6. Data quality summary ─────────────────────────────────
        quality_summary = self._assess_quality(spot_quotes, options_chains)

        latency_ms = (time.monotonic() - t_start) * 1000
        total_contracts = sum(len(v) for v in options_chains.values())
        logger.info(
            f"[Agent A] Cycle {cycle_id} complete | "
            f"latency={latency_ms:.0f}ms | "
            f"tickers={len(spot_quotes)}/{len(self._all_tickers)} | "
            f"chains={total_contracts} | "
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
    # Batched spot fetching
    # ------------------------------------------------------------------

    async def _fetch_all_spots_batched(self) -> dict[str, SpotQuote]:
        """Fetch spot quotes for all tickers using batched API call."""
        batches = self.settings.get_batches()
        all_quotes: dict[str, SpotQuote] = {}

        for batch_idx, batch in enumerate(batches):
            logger.debug(f"[Agent A] Spot batch {batch_idx + 1}/{len(batches)}: {batch}")

            if hasattr(self.market_provider, 'fetch_spot_quotes_batch'):
                batch_quotes = await self.market_provider.fetch_spot_quotes_batch(
                    batch,
                    batch_interval=self.settings.batch_interval_seconds,
                )
            else:
                # Fallback: fetch one by one
                batch_quotes = {}
                for ticker in batch:
                    try:
                        q = await self.market_provider.fetch_spot_quote(ticker)
                        batch_quotes[ticker] = q
                    except Exception as e:
                        logger.warning(f"[Agent A] Spot failed for {ticker}: {e}")

            for ticker, quote in batch_quotes.items():
                all_quotes[ticker] = quote

        return all_quotes

    # ------------------------------------------------------------------
    # Batched options chain fetching
    # ------------------------------------------------------------------

    async def _fetch_all_chains_batched(
        self,
        spot_quotes: dict[str, SpotQuote],
    ) -> dict[str, list[OptionsChain]]:
        """Fetch options chains for all tickers using batched API call."""
        batches = self.settings.get_batches()
        all_chains: dict[str, list[OptionsChain]] = {t: [] for t in self._all_tickers}

        for batch_idx, batch in enumerate(batches):
            valid_batch = [t for t in batch if t in spot_quotes]
            if not valid_batch:
                continue

            logger.debug(f"[Agent A] Options batch {batch_idx + 1}/{len(batches)}: {valid_batch}")

            if hasattr(self.market_provider, 'fetch_options_chains_batch'):
                batch_chains = await self.market_provider.fetch_options_chains_batch(
                    valid_batch,
                    moneyness_range_pct=self.settings.option_moneyness_range_pct,
                    batch_interval=self.settings.batch_interval_seconds,
                )
                for ticker, chains in batch_chains.items():
                    if chains:
                        all_chains[ticker] = chains
            else:
                # Fallback: fetch one by one
                for ticker in valid_batch:
                    try:
                        chains = await self.market_provider.fetch_options_chain(
                            ticker=ticker,
                            moneyness_range_pct=self.settings.option_moneyness_range_pct,
                        )
                        if chains:
                            all_chains[ticker] = chains
                    except Exception as e:
                        logger.warning(f"[Agent A] Chain failed for {ticker}: {e}")

        return all_chains

    # ------------------------------------------------------------------
    # Greeks enrichment
    # ------------------------------------------------------------------

    def _enrich_greeks(
        self, chains: dict[str, list[OptionsChain]]
    ) -> dict[str, list[OptionsChain]]:
        """Run BS Greek enrichment on all chains."""
        r = self.settings.risk_free_rate
        enriched: dict[str, list[OptionsChain]] = {}
        for ticker, chain_list in chains.items():
            enriched[ticker] = []
            for chain in chain_list:
                needs_enrichment = any(
                    c.gamma is None or c.implied_volatility is None
                    for c in chain.contracts[:5]
                )
                if needs_enrichment:
                    try:
                        chain = enrich_chain_greeks(chain, r)
                    except Exception as e:
                        logger.warning(f"Greek enrichment failed for {ticker}/{chain.expiration}: {e}")
                enriched[ticker].append(chain)
        return enriched

    # ------------------------------------------------------------------
    # Prediction markets
    # ------------------------------------------------------------------

    async def _fetch_prediction_markets(self):
        try:
            return await self.prediction_aggregator.fetch_all(
                watch_tickers=self._all_tickers
            )
        except Exception as e:
            logger.warning(f"[Agent A] Prediction market fetch failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Catalyst signals
    # ------------------------------------------------------------------

    def _resolve_catalyst_signals(self) -> list[CatalystSignal]:
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

    # ------------------------------------------------------------------
    # Data quality assessment
    # ------------------------------------------------------------------

    def _assess_quality(
        self,
        spot_quotes: dict[str, SpotQuote],
        options_chains: dict[str, list[OptionsChain]],
    ) -> dict[str, DataQuality]:
        quality: dict[str, DataQuality] = {}
        for ticker in self._all_tickers:
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
