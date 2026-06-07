"""
agents/agent_c.py — Risk Circuit Breaker & Watchdog
=====================================================
Responsibilities:
  - Evaluate AgentBOutput against pre-configured risk thresholds
  - Assign risk level (LOW / MEDIUM / HIGH / CRITICAL) per anomaly
  - Compile actionable trading narratives
  - Dispatch RiskAlerts via AlertDispatcher (webhooks, Slack, stdout)
  - Trigger simulated or live stop-loss/hedge orders (dry-run safe)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from options_agent.config import Settings
from options_agent.core.schemas import (
    AgentBOutput,
    AgentCOutput,
    RiskAlert,
    RiskLevel,
    RiskMatrix,
    GammaRegime,
)
from options_agent.core.ev_engine import build_ev_narrative
from options_agent.webhooks.dispatcher import AlertDispatcher
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)


class AgentC:
    """
    Risk Circuit Breaker & Watchdog.

    Evaluates each RiskMatrix and fires RiskAlerts when thresholds are crossed.
    Alert deduplication: won't re-fire the same rule for the same ticker
    within `dedup_window_seconds`.
    """

    def __init__(
        self,
        settings: Settings,
        dispatcher: AlertDispatcher,
        input_queue: asyncio.Queue,   # B→C queue
        dedup_window_seconds: float = 300.0,  # 5-min dedup window
    ):
        self.settings     = settings
        self.dispatcher   = dispatcher
        self.input_queue  = input_queue
        self.dedup_window = dedup_window_seconds

        # Deduplication: {(ticker, rule): last_fired_timestamp}
        self._last_fired: dict[tuple[str, str], datetime] = {}

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    async def run_cycle(self, b_output: AgentBOutput) -> AgentCOutput:
        """Evaluate risk matrices and dispatch alerts."""
        logger.info(
            f"[Agent C] Evaluating cycle {b_output.cycle_id} | "
            f"anomalies={b_output.anomalies_detected}"
        )

        alerts_fired: list[RiskAlert] = []
        simulated_orders: list[dict] = []
        webhooks_dispatched = 0

        for ticker, matrix in b_output.risk_matrices.items():
            ticker_alerts = self._evaluate_matrix(ticker, matrix)

            for alert in ticker_alerts:
                if self._is_duplicate(ticker, alert.triggered_rules):
                    logger.debug(f"[Agent C] Dedup skip: {ticker} rules={alert.triggered_rules}")
                    continue

                # Update dedup tracker
                for rule in alert.triggered_rules:
                    self._last_fired[(ticker, rule)] = alert.timestamp_utc

                # Dispatch
                ok = await self.dispatcher.dispatch(alert)
                if ok:
                    webhooks_dispatched += 1
                alerts_fired.append(alert)

                # Simulated orders (dry-run or real based on config)
                orders = self._generate_orders(alert, matrix)
                simulated_orders.extend(orders)

        c_output = AgentCOutput(
            cycle_id=b_output.cycle_id,
            alerts_fired=alerts_fired,
            webhooks_dispatched=webhooks_dispatched,
            simulated_orders=simulated_orders,
            timestamp_utc=datetime.utcnow(),
        )

        if alerts_fired:
            logger.warning(
                f"[Agent C] Cycle {b_output.cycle_id}: {len(alerts_fired)} alert(s) fired"
            )
        else:
            logger.info(f"[Agent C] Cycle {b_output.cycle_id}: No alerts")

        return c_output

    # ------------------------------------------------------------------
    # Risk evaluation rules
    # ------------------------------------------------------------------

    def _evaluate_matrix(
        self, ticker: str, matrix: RiskMatrix
    ) -> list[RiskAlert]:
        alerts: list[RiskAlert] = []
        cfg = self.settings.risk_thresholds

        # ── Rule 1: Gamma Wall Proximity ─────────────────────────────
        if matrix.gamma_wall_breach and matrix.gamma_profiles:
            profile = matrix.gamma_profiles[0]
            gw = profile.gamma_wall_strike
            proximity_pct = abs(matrix.spot_price - gw) / matrix.spot_price if gw else 0

            risk_level = (
                RiskLevel.HIGH if proximity_pct <= cfg.gamma_wall_proximity_pct / 2
                else RiskLevel.MEDIUM
            )
            alerts.append(self._build_gamma_wall_alert(ticker, matrix, risk_level))

        # ── Rule 2: Zero Gamma Line Breach (Regime Flip) ─────────────
        if matrix.zero_gamma_breach:
            # Zero gamma breach is HIGH by default — regime flips are dangerous
            alerts.append(self._build_zero_gamma_alert(ticker, matrix, RiskLevel.HIGH))

        # ── Rule 3: IV Crush Risk ─────────────────────────────────────
        if matrix.iv_crush_imminent and matrix.iv_analysis:
            iv = matrix.iv_analysis
            risk_level = (
                RiskLevel.HIGH if iv.iv_rank >= 0.85
                else RiskLevel.MEDIUM
            )
            alerts.append(self._build_iv_crush_alert(ticker, matrix, risk_level))

        # ── Rule 4: EV Discrepancy / Cross-Market Arbitrage ──────────
        if matrix.ev_arb_detected and matrix.ev_discrepancies:
            best_disc = max(matrix.ev_discrepancies, key=lambda d: d.ev_spread_pct)
            risk_level = (
                RiskLevel.HIGH if best_disc.ev_spread_pct >= 0.10
                else RiskLevel.MEDIUM
            )
            alerts.append(self._build_ev_alert(ticker, matrix, risk_level))

        # ── Rule 5: Critical Composite (multiple simultaneous flags) ──
        active_flags = sum([
            matrix.gamma_wall_breach,
            matrix.zero_gamma_breach,
            matrix.iv_crush_imminent,
            matrix.ev_arb_detected,
        ])
        if active_flags >= 3:
            alerts.append(self._build_composite_critical_alert(ticker, matrix))

        return alerts

    # ------------------------------------------------------------------
    # Alert constructors
    # ------------------------------------------------------------------

    def _build_gamma_wall_alert(
        self, ticker: str, matrix: RiskMatrix, risk_level: RiskLevel
    ) -> RiskAlert:
        profile = matrix.gamma_profiles[0] if matrix.gamma_profiles else None
        gw = profile.gamma_wall_strike if profile else None
        max_pain = profile.max_pain_strike if profile else None
        regime = profile.gamma_regime if profile else None

        regime_str = "LONG gamma (stabilizing)" if regime == GammaRegime.LONG_GAMMA \
                     else "SHORT gamma (amplifying)"

        narrative = (
            f"{ticker} spot (${matrix.spot_price:.2f}) is approaching the Gamma Wall "
            f"at ${gw:.2f}. Market makers are currently in {regime_str} regime. "
            f"Max pain is at ${max_pain:.2f}. "
            f"Expect mean-reversion pressure near the gamma wall, but a breach could "
            f"trigger accelerated moves as dealer delta-hedging amplifies direction."
        )
        action = (
            f"Monitor for confirmed break of ${gw:.2f}. "
            f"If long: consider tightening stop to ${gw * 0.985:.2f}. "
            f"If short: gamma wall acts as resistance — potential for mean-reversion entry."
        )

        return RiskAlert(
            alert_id=f"GW_{ticker}_{uuid.uuid4().hex[:8]}",
            ticker=ticker,
            risk_level=risk_level,
            headline=f"Spot approaching Gamma Wall at ${gw:.2f}" if gw else "Gamma Wall Proximity Alert",
            narrative=narrative,
            recommended_action=action,
            gamma_wall_strike=gw,
            max_pain_strike=max_pain,
            spot_price=matrix.spot_price,
            triggered_rules=["GAMMA_WALL_PROXIMITY"],
            raw_matrix=matrix,
        )

    def _build_zero_gamma_alert(
        self, ticker: str, matrix: RiskMatrix, risk_level: RiskLevel
    ) -> RiskAlert:
        zg_profiles = [p for p in matrix.gamma_profiles if p.zero_gamma_strike is not None]
        zg = zg_profiles[0].zero_gamma_strike if zg_profiles else None

        narrative = (
            f"{ticker} spot (${matrix.spot_price:.2f}) has crossed the Zero Gamma Line "
            f"at ${zg:.2f}. Market maker regime has FLIPPED from Long Gamma to Short Gamma. "
            f"In short-gamma, dealer hedging AMPLIFIES price moves (sell more when price drops, "
            f"buy more when price rises). This creates a feedback loop and significantly "
            f"increases realized volatility. Tail risk is elevated."
        )
        action = (
            "REDUCE long delta exposure immediately. "
            "Consider protective puts or VIX hedges. "
            "Widen stop-losses to accommodate higher realized vol. "
            "Do NOT add to long positions until regime resolves."
        )

        return RiskAlert(
            alert_id=f"ZG_{ticker}_{uuid.uuid4().hex[:8]}",
            ticker=ticker,
            risk_level=risk_level,
            headline=f"REGIME FLIP: Zero Gamma Line breached at ${zg:.2f}" if zg else "Zero Gamma Breach",
            narrative=narrative,
            recommended_action=action,
            zero_gamma_strike=zg,
            spot_price=matrix.spot_price,
            triggered_rules=["ZERO_GAMMA_BREACH"],
            raw_matrix=matrix,
        )

    def _build_iv_crush_alert(
        self, ticker: str, matrix: RiskMatrix, risk_level: RiskLevel
    ) -> RiskAlert:
        iv = matrix.iv_analysis
        catalysts = [c for c in matrix.catalyst_signals if c.is_imminent]
        cat_str = catalysts[0].description if catalysts else "upcoming catalyst"
        hours = catalysts[0].hours_until_event if catalysts else 0

        narrative = (
            f"{ticker} IV Rank is {iv.iv_rank:.1%} (ATM IV: {iv.current_iv:.1%}). "
            f"This is near the {iv.iv_rank:.0%} percentile of the past year. "
            f"A '{cat_str}' event is in ~{hours:.0f}h. "
            f"Post-event IV Crush is a high-probability outcome — option premiums will "
            f"likely collapse 30–60% immediately after the catalyst resolves. "
            f"Long premium holders must have a plan."
        )
        action = (
            "IF LONG PREMIUM: Exit or roll to next expiry before the event. "
            "IF SHORT PREMIUM: Current elevated IV is favorable for credit strategies. "
            "Consider iron condors or credit spreads with strikes outside expected move. "
            f"Expected 1-stdev move ≈ ${matrix.spot_price * iv.current_iv * (hours / 8760) ** 0.5:.2f}"
        )

        return RiskAlert(
            alert_id=f"IVC_{ticker}_{uuid.uuid4().hex[:8]}",
            ticker=ticker,
            risk_level=risk_level,
            headline=f"IV Crush Risk: IV Rank {iv.iv_rank:.1%} with {cat_str} in {hours:.0f}h",
            narrative=narrative,
            recommended_action=action,
            spot_price=matrix.spot_price,
            iv_rank=iv.iv_rank,
            hours_to_catalyst=hours,
            triggered_rules=["IV_CRUSH_RISK"],
            raw_matrix=matrix,
        )

    def _build_ev_alert(
        self, ticker: str, matrix: RiskMatrix, risk_level: RiskLevel
    ) -> RiskAlert:
        disc = max(matrix.ev_discrepancies, key=lambda d: d.ev_spread_pct)
        narrative = build_ev_narrative(disc)

        action = (
            f"Kelly-optimal position size: {disc.kelly_fraction:.1%} of capital. "
            f"Verify contract liquidity before execution. "
            f"Confirm spread is not explained by different reference dates or outcomes."
        ) if disc.kelly_fraction else (
            "Review the discrepancy manually — Kelly fraction is undefined (no statistical edge)."
        )

        return RiskAlert(
            alert_id=f"EV_{ticker}_{uuid.uuid4().hex[:8]}",
            ticker=ticker,
            risk_level=risk_level,
            headline=f"EV Discrepancy: {disc.ev_spread_pct:.1%} spread vs prediction market",
            narrative=narrative,
            recommended_action=action,
            spot_price=matrix.spot_price,
            ev_spread_pct=disc.ev_spread_pct,
            triggered_rules=["EV_DISCREPANCY"],
            raw_matrix=matrix,
        )

    def _build_composite_critical_alert(
        self, ticker: str, matrix: RiskMatrix
    ) -> RiskAlert:
        flags = []
        if matrix.gamma_wall_breach:  flags.append("Gamma Wall Proximity")
        if matrix.zero_gamma_breach:  flags.append("Zero Gamma Regime Flip")
        if matrix.iv_crush_imminent:  flags.append("IV Crush Risk")
        if matrix.ev_arb_detected:    flags.append("EV Discrepancy")

        narrative = (
            f"CRITICAL COMPOSITE ALERT on {ticker} (${matrix.spot_price:.2f}). "
            f"Multiple simultaneous risk signals: {', '.join(flags)}. "
            f"Confluence of gamma, volatility, and cross-market signals indicates "
            f"structurally elevated tail risk. This is a rare multi-dimensional alert."
        )

        return RiskAlert(
            alert_id=f"CRIT_{ticker}_{uuid.uuid4().hex[:8]}",
            ticker=ticker,
            risk_level=RiskLevel.CRITICAL,
            headline=f"CRITICAL: {len(flags)} simultaneous risk signals on {ticker}",
            narrative=narrative,
            recommended_action=(
                "IMMEDIATE REVIEW REQUIRED. "
                "Consider reducing all {ticker} derivatives exposure. "
                "Activate portfolio-level hedges (SPX puts, VIX calls if macro). "
                "Do not add risk until all signals resolve."
            ).format(ticker=ticker),
            spot_price=matrix.spot_price,
            triggered_rules=["COMPOSITE_CRITICAL"],
            raw_matrix=matrix,
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _is_duplicate(
        self, ticker: str, rules: list[str]
    ) -> bool:
        now = datetime.utcnow()
        for rule in rules:
            last = self._last_fired.get((ticker, rule))
            if last:
                age = (now - last).total_seconds()
                if age < self.dedup_window:
                    return True
        return False

    # ------------------------------------------------------------------
    # Simulated order generation
    # ------------------------------------------------------------------

    def _generate_orders(
        self, alert: RiskAlert, matrix: RiskMatrix
    ) -> list[dict]:
        """
        Generate simulated (paper) hedge orders for HIGH/CRITICAL alerts.
        In dry_run mode these are never executed — printed for review only.
        """
        if alert.risk_level not in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return []

        orders = []

        if "ZERO_GAMMA_BREACH" in alert.triggered_rules:
            # Protective put on the underlying (1 contract = 100 shares)
            orders.append({
                "type": "SIMULATED_PROTECTIVE_PUT",
                "ticker": alert.ticker,
                "action": "BUY",
                "quantity": 1,
                "strike_approx": round(matrix.spot_price * 0.97, 2),  # 3% OTM
                "dte_approx": 30,
                "dry_run": self.settings.dry_run,
                "alert_id": alert.alert_id,
                "timestamp_utc": datetime.utcnow().isoformat(),
            })

        if "IV_CRUSH_RISK" in alert.triggered_rules:
            # Short straddle / calendar roll signal (no live order in dry-run)
            orders.append({
                "type": "SIMULATED_IV_CRUSH_PLAY",
                "ticker": alert.ticker,
                "action": "SELL_STRADDLE",
                "note": "Sell ATM straddle post-event for IV crush capture",
                "dry_run": self.settings.dry_run,
                "alert_id": alert.alert_id,
                "timestamp_utc": datetime.utcnow().isoformat(),
            })

        return orders
