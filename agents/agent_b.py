"""
agents/agent_b.py — Quantitative Reasoning & Greeks Engine
===========================================================
Responsibilities:
  - Consume AgentAOutput from the A→B queue
  - Run Gamma Profile / Max Pain / Zero Gamma computations per expiry
  - Run IV Rank / IV Percentile / IV Crush analysis per ticker
  - Detect cross-market EV discrepancies vs prediction markets
  - If anomalies detected, push AgentBOutput onto the B→C queue
  - Otherwise push a no-op heartbeat for Agent C monitoring
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
    AgentBOutput,
    DataQuality,
    GammaProfile,
    IVAnalysis,
    RiskMatrix,
)
from options_agent.core.greeks_engine import (
    compute_gamma_profile,
    find_aggregate_gamma_wall,
)
from options_agent.core.iv_analytics import build_iv_analysis, extract_atm_iv
from options_agent.core.ev_engine import sweep_ev_across_expirations
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)


class AgentB:
    """
    Quantitative Reasoning & Greeks Engine.

    Consumes clean data from Agent A and produces a RiskMatrix per ticker.
    Heavy numerical work — all synchronous computation wrapped in
    asyncio.to_thread for non-blocking execution.
    """

    def __init__(
        self,
        settings: Settings,
        input_queue: asyncio.Queue,   # A→B queue
        output_queue: asyncio.Queue,  # B→C queue
        iv_history_store: Optional[dict[str, list[float]]] = None,
    ):
        self.settings        = settings
        self.input_queue     = input_queue
        self.output_queue    = output_queue
        self.iv_history_store = iv_history_store or {t: [] for t in settings.watch_tickers}

    # ------------------------------------------------------------------
    # Main cycle — await on the A→B queue
    # ------------------------------------------------------------------

    async def run_cycle(self, a_output: AgentAOutput) -> AgentBOutput:
        """Process one AgentAOutput and emit AgentBOutput."""
        t_start = time.monotonic()
        logger.info(f"[Agent B] Processing cycle {a_output.cycle_id}")

        # Run heavy computation in thread pool to keep event loop free
        risk_matrices = await asyncio.to_thread(
            self._compute_all_risk_matrices, a_output
        )

        anomalies = any(
            m.gamma_wall_breach or m.zero_gamma_breach
            or m.iv_crush_imminent or m.ev_arb_detected
            for m in risk_matrices.values()
        )

        anomaly_summary = ""
        if anomalies:
            parts = []
            for ticker, matrix in risk_matrices.items():
                flags = []
                if matrix.gamma_wall_breach:  flags.append("GammaWallBreach")
                if matrix.zero_gamma_breach:  flags.append("ZeroGammaBreach")
                if matrix.iv_crush_imminent:  flags.append("IVCrushRisk")
                if matrix.ev_arb_detected:    flags.append("EVArbitrage")
                if flags:
                    parts.append(f"{ticker}:[{','.join(flags)}]")
            anomaly_summary = " | ".join(parts)

        compute_ms = (time.monotonic() - t_start) * 1000

        b_output = AgentBOutput(
            cycle_id=a_output.cycle_id,
            risk_matrices=risk_matrices,
            anomalies_detected=anomalies,
            anomaly_summary=anomaly_summary,
            compute_time_ms=compute_ms,
            timestamp_utc=datetime.utcnow(),
        )

        await self.output_queue.put(b_output)

        logger.info(
            f"[Agent B] Cycle {a_output.cycle_id} done | "
            f"compute={compute_ms:.0f}ms | "
            f"anomalies={'YES → ' + anomaly_summary if anomalies else 'none'}"
        )
        return b_output

    # ------------------------------------------------------------------
    # Synchronous computation (runs in thread pool)
    # ------------------------------------------------------------------

    def _compute_all_risk_matrices(
        self, a_output: AgentAOutput
    ) -> dict[str, RiskMatrix]:
        matrices: dict[str, RiskMatrix] = {}

        for ticker in a_output.tickers_processed:
            quality = a_output.data_quality_summary.get(ticker, DataQuality.INVALID)
            if quality == DataQuality.INVALID:
                logger.warning(f"[Agent B] Skipping {ticker} — INVALID data quality")
                continue

            spot_quote = a_output.spot_quotes.get(ticker)
            if not spot_quote:
                continue

            chains = a_output.options_chains.get(ticker, [])

            try:
                matrix = self._compute_ticker_risk_matrix(
                    ticker=ticker,
                    spot=spot_quote.spot_price,
                    chains=chains,
                    a_output=a_output,
                )
                matrices[ticker] = matrix
            except Exception as e:
                logger.error(f"[Agent B] Risk matrix computation failed for {ticker}: {e}", exc_info=True)

        return matrices

    def _compute_ticker_risk_matrix(
        self,
        ticker: str,
        spot: float,
        chains: list,
        a_output: AgentAOutput,
    ) -> RiskMatrix:
        cfg = self.settings.risk_thresholds
        r   = self.settings.risk_free_rate

        # ── Gamma Profiles (per expiry) ───────────────────────────────
        gamma_profiles: list[GammaProfile] = []
        for chain in chains:
            try:
                profile = compute_gamma_profile(chain, r)
                gamma_profiles.append(profile)
            except Exception as e:
                logger.debug(f"Gamma profile failed for {ticker}/{chain.expiration}: {e}")

        # ── Gamma Wall / Zero Gamma breach detection ──────────────────
        gamma_wall_breach = False
        zero_gamma_breach = False

        if gamma_profiles:
            agg_wall = find_aggregate_gamma_wall(gamma_profiles)
            if agg_wall:
                proximity = abs(spot - agg_wall) / spot
                gamma_wall_breach = proximity <= cfg.gamma_wall_proximity_pct
                if gamma_wall_breach:
                    logger.debug(f"[Agent B] {ticker}: Gamma wall breach! spot={spot} wall={agg_wall} prox={proximity:.2%}")

            for profile in gamma_profiles:
                if profile.zero_gamma_strike is not None:
                    zg = profile.zero_gamma_strike
                    buf = zg * cfg.zero_gamma_breach_buffer_pct
                    if abs(spot - zg) <= buf:
                        zero_gamma_breach = True
                        logger.debug(f"[Agent B] {ticker}: Zero gamma breach! spot={spot} ZGL={zg}")

        # ── IV Analysis ───────────────────────────────────────────────
        iv_analysis: Optional[IVAnalysis] = None
        iv_crush_imminent = False

        # Select front-month chain (nearest expiry)
        if chains:
            front_chain = min(chains, key=lambda c: c.expiration)
            iv_hist = self.iv_history_store.get(ticker, [])

            # Update IV history with today's reading
            current_iv = extract_atm_iv(front_chain)
            if current_iv and current_iv > 0:
                iv_hist.append(current_iv)
                # Keep rolling window
                max_days = self.settings.iv_history_days
                self.iv_history_store[ticker] = iv_hist[-max_days:]

            iv_analysis = build_iv_analysis(
                ticker=ticker,
                front_month_chain=front_chain,
                iv_history=self.iv_history_store.get(ticker, []),
                catalyst_signals=a_output.catalyst_signals,
                iv_rank_threshold=cfg.iv_crush_iv_rank_threshold,
                iv_crush_hours=cfg.iv_crush_hours_before_event,
            )
            iv_crush_imminent = iv_analysis.iv_crush_risk

        # ── EV Discrepancy ────────────────────────────────────────────
        ev_discrepancies = sweep_ev_across_expirations(
            ticker=ticker,
            chains=chains,
            prediction_contracts=a_output.prediction_contracts,
            r=r,
            min_spread_pct=cfg.ev_min_spread_pct,
        )
        ev_arb_detected = len(ev_discrepancies) > 0

        # ── Catalyst signals for this ticker ─────────────────────────
        relevant_catalysts = [
            sig for sig in a_output.catalyst_signals
            if sig.ticker == ticker or sig.ticker == "MACRO"
        ]

        return RiskMatrix(
            ticker=ticker,
            spot_price=spot,
            timestamp_utc=datetime.utcnow(),
            gamma_profiles=gamma_profiles,
            iv_analysis=iv_analysis,
            ev_discrepancies=ev_discrepancies,
            catalyst_signals=relevant_catalysts,
            gamma_wall_breach=gamma_wall_breach,
            zero_gamma_breach=zero_gamma_breach,
            iv_crush_imminent=iv_crush_imminent,
            ev_arb_detected=ev_arb_detected,
        )
