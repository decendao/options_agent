"""
tests/test_ev_engine.py — Unit tests for Cross-Market EV Engine
"""

from datetime import date, datetime, timedelta

import pytest

from options_agent.core.ev_engine import (
    _kelly_fraction,
    bs_risk_neutral_prob,
    compute_ev_discrepancy,
    extract_otm_implied_prob,
    find_matching_prediction_contract,
    sweep_ev_across_expirations,
)
from options_agent.core.schemas import (
    OptionContract,
    OptionsChain,
    OptionType,
    PredictionMarketContract,
)
from options_agent.core.greeks_engine import enrich_chain_greeks


def make_chain_with_iv(
    spot: float = 100.0,
    atm_iv: float = 0.25,
    strikes: list = None,
    dte: int = 30,
    ticker: str = "SPY",
) -> OptionsChain:
    if strikes is None:
        strikes = [90, 95, 100, 105, 110]
    expiration = date.today() + timedelta(days=dte)
    contracts = []
    for k in strikes:
        for opt_type in [OptionType.CALL, OptionType.PUT]:
            dist = abs(k - spot) / spot
            iv = atm_iv * (1 + dist * 0.4)
            contracts.append(
                OptionContract(
                    ticker=ticker,
                    expiration=expiration,
                    strike=float(k),
                    option_type=opt_type,
                    bid=max(0.01, 5.0 - dist * 20),
                    ask=max(0.02, 5.20 - dist * 20),
                    last=max(0.01, 5.10 - dist * 20),
                    volume=1000,
                    open_interest=5000,
                    implied_volatility=iv,
                    timestamp_utc=datetime.utcnow(),
                )
            )
    return OptionsChain(
        ticker=ticker,
        expiration=expiration,
        spot_price=spot,
        contracts=contracts,
        timestamp_utc=datetime.utcnow(),
    )


def make_pred_contract(
    ticker: str = "SPY",
    yes_price: float = 0.30,
    volume: float = 1_000_000,
) -> PredictionMarketContract:
    return PredictionMarketContract(
        market_name="polymarket",
        contract_id=f"mock_{ticker}_above_105",
        question=f"Will {ticker} close above $105 by month end?",
        ticker_ref=ticker,
        yes_price=yes_price,
        no_price=1.0 - yes_price,
        volume_24h=volume,
        timestamp_utc=datetime.utcnow(),
    )


class TestBSRiskNeutralProb:
    def test_atm_call_prob_near_half(self):
        prob = bs_risk_neutral_prob(100, 100, 1.0, 0.0, 0.20, OptionType.CALL)
        assert 0.40 < prob < 0.60

    def test_deep_itm_call_prob_near_one(self):
        prob = bs_risk_neutral_prob(100, 50, 1.0, 0.0, 0.20, OptionType.CALL)
        assert prob > 0.90

    def test_deep_otm_call_prob_near_zero(self):
        prob = bs_risk_neutral_prob(100, 200, 1.0, 0.0, 0.20, OptionType.CALL)
        assert prob < 0.10

    def test_put_prob_is_one_minus_call(self):
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.20
        call_prob = bs_risk_neutral_prob(S, K, T, r, sigma, OptionType.CALL)
        put_prob  = bs_risk_neutral_prob(S, K, T, r, sigma, OptionType.PUT)
        assert abs(call_prob + put_prob - 1.0) < 1e-9

    def test_prob_bounded_0_1(self):
        for K in [50, 75, 100, 125, 150]:
            p = bs_risk_neutral_prob(100, K, 0.5, 0.05, 0.25, OptionType.CALL)
            assert 0 <= p <= 1


class TestExtractOTMImpliedProb:
    def test_returns_probability_and_contract(self):
        chain = make_chain_with_iv(spot=100.0, atm_iv=0.25)
        prob, contract = extract_otm_implied_prob(chain, r=0.05, target_moneyness=1.05)
        assert prob is not None
        assert 0 < prob < 1
        assert contract is not None

    def test_empty_chain_returns_none(self):
        empty = OptionsChain(
            ticker="SPY",
            expiration=date.today() + timedelta(days=30),
            spot_price=100.0,
            contracts=[],
            timestamp_utc=datetime.utcnow(),
        )
        prob, contract = extract_otm_implied_prob(empty, r=0.05)
        assert prob is None
        assert contract is None


class TestFindMatchingPredictionContract:
    def test_finds_by_ticker_ref(self):
        contracts = [make_pred_contract("SPY"), make_pred_contract("QQQ")]
        match = find_matching_prediction_contract("SPY", contracts)
        assert match is not None
        assert match.ticker_ref == "SPY"

    def test_returns_none_when_no_match(self):
        contracts = [make_pred_contract("QQQ")]
        match = find_matching_prediction_contract("SPY", contracts)
        assert match is None

    def test_prefers_highest_volume(self):
        c1 = make_pred_contract("SPY", volume=100_000)
        c2 = make_pred_contract("SPY", volume=2_000_000)
        match = find_matching_prediction_contract("SPY", [c1, c2])
        assert match.volume_24h == 2_000_000


class TestKellyFraction:
    def test_positive_edge_returns_fraction(self):
        # Our estimate = 0.6, market implies 0.4 → we have edge
        kelly = _kelly_fraction(0.60, 0.40)
        assert kelly is not None
        assert 0 < kelly <= 0.25  # capped at 25%

    def test_no_edge_returns_none(self):
        # Our estimate = 0.3, market implies 0.6 → we have negative edge
        kelly = _kelly_fraction(0.30, 0.60)
        assert kelly is None

    def test_capped_at_25_pct(self):
        # Massive edge case → should be capped
        kelly = _kelly_fraction(0.99, 0.01)
        assert kelly is not None
        assert kelly <= 0.25


class TestComputeEVDiscrepancy:
    def test_significant_spread_detected(self):
        chain = make_chain_with_iv(spot=100.0, atm_iv=0.15)  # low IV → low options prob
        pred = make_pred_contract("SPY", yes_price=0.50)  # prediction says 50% likely
        disc = compute_ev_discrepancy(
            ticker="SPY",
            chain=chain,
            prediction_contracts=[pred],
            r=0.05,
            min_spread_pct=0.05,
        )
        # SPY at 100, OTM call at 105, low IV → low options prob → big spread vs 50%
        assert disc is not None
        assert disc.ev_spread_pct >= 0.05

    def test_no_prediction_contract_returns_none(self):
        chain = make_chain_with_iv(spot=100.0, atm_iv=0.25)
        disc = compute_ev_discrepancy("SPY", chain, [], r=0.05)
        assert disc is None


class TestSweepEVAcrossExpirations:
    def test_sweeps_all_expirations(self):
        chains = [
            make_chain_with_iv(spot=100.0, atm_iv=0.15, dte=30),
            make_chain_with_iv(spot=100.0, atm_iv=0.15, dte=60),
        ]
        pred_contracts = [make_pred_contract("SPY", yes_price=0.50)]
        results = sweep_ev_across_expirations("SPY", chains, pred_contracts, r=0.05, min_spread_pct=0.05)
        # With big spread we expect some results
        for disc in results:
            assert disc.is_significant
            assert disc.ev_spread_pct >= 0.05
