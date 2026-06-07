"""
tests/test_iv_analytics.py — Unit tests for IV analytics
"""

from datetime import date, datetime, timedelta

import pytest

from options_agent.core.iv_analytics import (
    build_iv_analysis,
    compute_iv_percentile,
    compute_iv_rank,
    detect_iv_crush_risk,
    extract_atm_iv,
    extract_skew,
)
from options_agent.core.schemas import CatalystSignal, OptionContract, OptionsChain, OptionType
from options_agent.core.greeks_engine import enrich_chain_greeks


def make_chain(spot: float = 100.0, atm_iv: float = 0.25, dte: int = 30) -> OptionsChain:
    expiration = date.today() + timedelta(days=dte)
    contracts = []
    r = 0.05
    for k in [90, 95, 100, 105, 110]:
        for opt_type in [OptionType.CALL, OptionType.PUT]:
            contracts.append(
                OptionContract(
                    ticker="SPY",
                    expiration=expiration,
                    strike=float(k),
                    option_type=opt_type,
                    bid=5.0,
                    ask=5.20,
                    last=5.10,
                    volume=1000,
                    open_interest=5000,
                    implied_volatility=atm_iv * (1 + abs(k - spot) / spot * 0.3),
                    timestamp_utc=datetime.utcnow(),
                )
            )
    return OptionsChain(
        ticker="SPY",
        expiration=expiration,
        spot_price=spot,
        contracts=contracts,
        timestamp_utc=datetime.utcnow(),
    )


class TestExtractATMIV:
    def test_returns_iv_near_atm(self):
        chain = make_chain(spot=100.0, atm_iv=0.25)
        iv = extract_atm_iv(chain)
        assert iv is not None
        assert 0.20 < iv < 0.35  # roughly ATM level

    def test_returns_none_for_empty_chain(self):
        empty = OptionsChain(
            ticker="SPY",
            expiration=date.today() + timedelta(days=30),
            spot_price=100.0,
            contracts=[],
            timestamp_utc=datetime.utcnow(),
        )
        assert extract_atm_iv(empty) is None


class TestComputeIVRank:
    def test_at_52w_high_returns_one(self):
        history = [0.10, 0.15, 0.20, 0.25, 0.30]
        rank = compute_iv_rank(0.30, history)
        assert abs(rank - 1.0) < 1e-9

    def test_at_52w_low_returns_zero(self):
        history = [0.10, 0.15, 0.20, 0.25, 0.30]
        rank = compute_iv_rank(0.10, history)
        assert abs(rank) < 1e-9

    def test_midpoint_returns_half(self):
        history = [0.10, 0.30]
        rank = compute_iv_rank(0.20, history)
        assert abs(rank - 0.5) < 1e-9

    def test_empty_history_returns_neutral(self):
        rank = compute_iv_rank(0.25, [])
        assert rank == 0.5

    def test_flat_history_returns_neutral(self):
        rank = compute_iv_rank(0.25, [0.25] * 50)
        assert rank == 0.5


class TestComputeIVPercentile:
    def test_all_below_returns_one(self):
        history = [0.10, 0.12, 0.14, 0.16, 0.18]
        pct = compute_iv_percentile(0.25, history)
        assert abs(pct - 1.0) < 1e-9

    def test_all_above_returns_zero(self):
        history = [0.30, 0.35, 0.40, 0.45, 0.50]  # all above 0.05
        pct = compute_iv_percentile(0.05, history)
        assert abs(pct) < 1e-9

    def test_half_above_half_below(self):
        history = [0.10, 0.20, 0.30, 0.40]
        pct = compute_iv_percentile(0.25, history)
        assert abs(pct - 0.5) < 1e-9  # 2 of 4 are below 0.25


class TestDetectIVCrushRisk:
    def _make_catalyst(self, hours: float) -> CatalystSignal:
        return CatalystSignal(
            ticker="MACRO",
            event_type="FED",
            event_time_utc=datetime.utcnow() + timedelta(hours=hours),
            description="FOMC Rate Decision",
            hours_until_event=hours,
            is_imminent=hours <= 48,
        )

    def test_high_iv_rank_with_imminent_catalyst(self):
        catalyst = self._make_catalyst(hours=24)
        is_risk, reason = detect_iv_crush_risk(
            {"iv_rank": 0.85, "ticker": "SPY"},
            [catalyst],
            iv_rank_threshold=0.70,
            hours_before_event=48,
        )
        assert is_risk is True
        assert "48" in reason or "crush" in reason.lower() or "iv rank" in reason.lower()

    def test_low_iv_rank_no_risk(self):
        catalyst = self._make_catalyst(hours=24)
        is_risk, reason = detect_iv_crush_risk(
            {"iv_rank": 0.40, "ticker": "SPY"},
            [catalyst],
        )
        assert is_risk is False

    def test_high_iv_rank_no_imminent_catalyst(self):
        catalyst = self._make_catalyst(hours=100)  # 100h → not imminent (>48h)
        is_risk, reason = detect_iv_crush_risk(
            {"iv_rank": 0.90, "ticker": "SPY"},
            [catalyst],
        )
        assert is_risk is False


class TestBuildIVAnalysis:
    def test_full_analysis_with_history(self):
        chain = make_chain(spot=100.0, atm_iv=0.25)
        history = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
        analysis = build_iv_analysis(
            ticker="SPY",
            front_month_chain=chain,
            iv_history=history,
            catalyst_signals=[],
        )
        assert analysis.current_iv > 0
        assert 0 <= analysis.iv_rank <= 1
        assert 0 <= analysis.iv_percentile <= 1
        assert analysis.iv_52w_high >= analysis.iv_52w_low

    def test_no_iv_chain_returns_shell(self):
        """Chain with no IV should return a safe shell with iv_rank=0.5."""
        empty_chain = OptionsChain(
            ticker="SPY",
            expiration=date.today() + timedelta(days=30),
            spot_price=100.0,
            contracts=[],
            timestamp_utc=datetime.utcnow(),
        )
        analysis = build_iv_analysis("SPY", empty_chain, [], [])
        assert analysis.current_iv == 0.0
        assert analysis.iv_crush_risk is False
