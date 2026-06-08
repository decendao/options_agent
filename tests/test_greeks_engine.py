"""
tests/test_greeks_engine.py — Unit tests for Black-Scholes Greeks Engine
"""

import math
from datetime import date, datetime, timedelta

import pytest

from options_agent.core.greeks_engine import (
    _find_zero_crossing,
    aggregate_gamma_profiles,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta,
    bs_vega,
    compute_gamma_profile,
    enrich_chain_greeks,
    find_aggregate_gamma_wall,
    implied_vol_newton,
    _compute_max_pain,
)
from options_agent.core.schemas import (
    GammaRegime,
    OptionContract,
    OptionsChain,
    OptionType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_contract(
    ticker: str = "SPY",
    strike: float = 580.0,
    option_type: OptionType = OptionType.CALL,
    bid: float = 5.0,
    ask: float = 5.20,
    last: float = 5.10,
    volume: int = 1000,
    open_interest: int = 5000,
    iv: float = 0.15,
    delta: float = None,
    gamma: float = None,
    expiration: date = None,
) -> OptionContract:
    if expiration is None:
        expiration = date.today() + timedelta(days=30)
    return OptionContract(
        ticker=ticker,
        expiration=expiration,
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        open_interest=open_interest,
        implied_volatility=iv,
        delta=delta,
        gamma=gamma,
        timestamp_utc=datetime.utcnow(),
    )


def make_chain(
    ticker: str = "SPY",
    spot: float = 578.0,
    strikes: list[float] = None,
    iv: float = 0.15,
    oi_scale: int = 5000,
) -> OptionsChain:
    if strikes is None:
        strikes = [560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0]
    expiration = date.today() + timedelta(days=30)
    contracts = []
    for k in strikes:
        for opt_type in [OptionType.CALL, OptionType.PUT]:
            dist = abs(k - spot) / spot
            oi = max(100, int(oi_scale * math.exp(-15 * dist)))
            contracts.append(
                make_contract(
                    ticker=ticker,
                    strike=k,
                    option_type=opt_type,
                    bid=max(0.01, 5.0 - dist * 100),
                    ask=max(0.02, 5.20 - dist * 100),
                    last=max(0.01, 5.10 - dist * 100),
                    open_interest=oi,
                    iv=iv + dist * 0.02,
                    expiration=expiration,
                )
            )
    return OptionsChain(
        ticker=ticker,
        expiration=expiration,
        spot_price=spot,
        contracts=contracts,
        timestamp_utc=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# BS Pricing
# ---------------------------------------------------------------------------

class TestBSPrice:
    def test_call_atm_positive(self):
        price = bs_price(100, 100, 0.25, 0.05, 0.20, OptionType.CALL)
        assert price > 0

    def test_put_atm_positive(self):
        price = bs_price(100, 100, 0.25, 0.05, 0.20, OptionType.PUT)
        assert price > 0

    def test_deep_itm_call_near_intrinsic(self):
        price = bs_price(100, 50, 0.01, 0.05, 0.20, OptionType.CALL)
        assert price >= 50 * 0.99  # roughly intrinsic

    def test_deep_otm_call_near_zero(self):
        price = bs_price(100, 200, 1.0, 0.05, 0.20, OptionType.CALL)
        assert price < 1.0

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.20
        call = bs_price(S, K, T, r, sigma, OptionType.CALL)
        put  = bs_price(S, K, T, r, sigma, OptionType.PUT)
        # C - P = S - K * exp(-r*T)
        lhs = call - put
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-6

    def test_zero_time_call_intrinsic(self):
        # Very short T → price ≈ intrinsic
        price = bs_price(110, 100, 0.001, 0.05, 0.20, OptionType.CALL)
        assert price > 9.5  # at least intrinsic minus small discount


class TestBSGreeks:
    def test_call_delta_between_0_and_1(self):
        for K in [90, 100, 110]:
            d = bs_delta(100, K, 0.5, 0.05, 0.20, OptionType.CALL)
            assert 0 <= d <= 1

    def test_put_delta_between_neg1_and_0(self):
        for K in [90, 100, 110]:
            d = bs_delta(100, K, 0.5, 0.05, 0.20, OptionType.PUT)
            assert -1 <= d <= 0

    def test_atm_call_delta_near_half(self):
        d = bs_delta(100, 100, 1.0, 0.0, 0.20, OptionType.CALL)
        assert 0.45 < d < 0.60

    def test_gamma_positive(self):
        g = bs_gamma(100, 100, 0.5, 0.05, 0.20)
        assert g > 0

    def test_gamma_symmetric_for_calls_puts(self):
        g_call = bs_gamma(100, 100, 0.5, 0.05, 0.20)
        g_put  = bs_gamma(100, 100, 0.5, 0.05, 0.20)
        assert abs(g_call - g_put) < 1e-12

    def test_theta_negative_for_long_options(self):
        theta_c = bs_theta(100, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        theta_p = bs_theta(100, 100, 0.5, 0.05, 0.20, OptionType.PUT)
        assert theta_c < 0
        assert theta_p < 0

    def test_vega_positive(self):
        v = bs_vega(100, 100, 0.5, 0.05, 0.20)
        assert v > 0

    def test_vega_per_one_pct_vol_change(self):
        # Vega should match: delta(price)/delta(sigma=1%) via finite diff
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20
        bump = 0.01
        p1 = bs_price(S, K, T, r, sigma + bump, OptionType.CALL)
        p0 = bs_price(S, K, T, r, sigma, OptionType.CALL)
        fd_vega = (p1 - p0) / 1.0  # already normalized to 1% in bs_vega
        v = bs_vega(S, K, T, r, sigma)
        assert abs(v - fd_vega) < 0.005


class TestIVSolver:
    def test_roundtrip_call(self):
        S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        price = bs_price(S, K, T, r, sigma, OptionType.CALL)
        solved = implied_vol_newton(price, S, K, T, r, OptionType.CALL)
        assert solved is not None
        assert abs(solved - sigma) < 1e-4

    def test_roundtrip_put(self):
        S, K, T, r, sigma = 100.0, 95.0, 0.25, 0.05, 0.30
        price = bs_price(S, K, T, r, sigma, OptionType.PUT)
        solved = implied_vol_newton(price, S, K, T, r, OptionType.PUT)
        assert solved is not None
        assert abs(solved - sigma) < 1e-4

    def test_returns_none_for_below_intrinsic(self):
        # Price below intrinsic value → no valid IV
        result = implied_vol_newton(0.001, 100, 50, 0.5, 0.05, OptionType.CALL)
        assert result is None

    def test_high_iv_roundtrip(self):
        S, K, T, r, sigma = 100.0, 110.0, 0.1, 0.05, 1.5  # 150% IV
        price = bs_price(S, K, T, r, sigma, OptionType.CALL)
        solved = implied_vol_newton(price, S, K, T, r, OptionType.CALL)
        assert solved is not None
        assert abs(solved - sigma) < 0.01


# ---------------------------------------------------------------------------
# Greeks enrichment
# ---------------------------------------------------------------------------

class TestEnrichChainGreeks:
    def test_enriches_missing_greeks(self):
        chain = make_chain()
        # Clear existing greeks to force re-computation
        no_greeks = [
            c.model_copy(update={"delta": None, "gamma": None,
                                  "theta": None, "vega": None})
            for c in chain.contracts
        ]
        bare_chain = chain.model_copy(update={"contracts": no_greeks})
        enriched = enrich_chain_greeks(bare_chain, r=0.05)
        for c in enriched.contracts:
            assert c.gamma is not None
            assert c.delta is not None

    def test_preserves_ticker_and_expiry(self):
        chain = make_chain()
        enriched = enrich_chain_greeks(chain, r=0.05)
        assert enriched.ticker == chain.ticker
        assert enriched.expiration == chain.expiration


# ---------------------------------------------------------------------------
# Gamma Profile computation
# ---------------------------------------------------------------------------

class TestGammaProfile:
    def test_profile_has_gamma_wall(self):
        chain = make_chain(spot=578.0)
        enriched = enrich_chain_greeks(chain, r=0.05)
        profile = compute_gamma_profile(enriched, r=0.05)
        assert profile.gamma_wall_strike is not None

    def test_max_pain_within_strike_range(self):
        chain = make_chain(spot=578.0)
        enriched = enrich_chain_greeks(chain, r=0.05)
        profile = compute_gamma_profile(enriched, r=0.05)
        if profile.max_pain_strike is not None:
            strikes = {c.strike for c in chain.contracts}
            assert profile.max_pain_strike in strikes

    def test_regime_is_enum_value(self):
        chain = make_chain(spot=578.0)
        enriched = enrich_chain_greeks(chain, r=0.05)
        profile = compute_gamma_profile(enriched, r=0.05)
        assert profile.gamma_regime in (
            GammaRegime.LONG_GAMMA, GammaRegime.SHORT_GAMMA, GammaRegime.NEUTRAL
        )

    def test_empty_chain_returns_neutral(self):
        empty_chain = OptionsChain(
            ticker="SPY",
            expiration=date.today() + timedelta(days=30),
            spot_price=578.0,
            contracts=[],
            timestamp_utc=datetime.utcnow(),
        )
        profile = compute_gamma_profile(empty_chain, r=0.05)
        assert profile.gamma_regime == GammaRegime.NEUTRAL
        assert profile.gamma_wall_strike is None


class TestZeroCrossing:
    def test_finds_crossing(self):
        import numpy as np
        strikes = np.array([560.0, 565.0, 570.0, 575.0, 580.0])
        gammas  = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        zg = _find_zero_crossing(strikes, gammas)
        assert zg is not None
        assert 569 < zg < 571  # Should be very close to 570

    def test_no_crossing_returns_none(self):
        import numpy as np
        strikes = np.array([560.0, 565.0, 570.0])
        gammas  = np.array([100.0, 200.0, 300.0])  # all positive, no crossing
        result = _find_zero_crossing(strikes, gammas)
        assert result is None


class TestMaxPain:
    def test_max_pain_minimizes_payout(self):
        chain = make_chain(spot=578.0)
        enriched = enrich_chain_greeks(chain, r=0.05)
        max_pain = _compute_max_pain(enriched)
        assert max_pain is not None

        strikes = sorted({c.strike for c in enriched.contracts})
        # Verify it's actually the minimum (not the maximum)
        def total_payout(candidate: float) -> float:
            pain = 0.0
            for c in enriched.contracts:
                oi = c.open_interest
                k = c.strike
                if c.option_type == OptionType.CALL:
                    pain += max(0.0, candidate - k) * oi * 100
                else:
                    pain += max(0.0, k - candidate) * oi * 100
            return pain

        mp_pain = total_payout(max_pain)
        for s in strikes:
            assert total_payout(s) >= mp_pain - 1e-6


class TestAggregateGammaWall:
    def test_aggregate_across_expirations(self):
        chain1 = make_chain(spot=578.0, strikes=[570.0, 575.0, 580.0, 585.0])
        chain2 = make_chain(spot=578.0, strikes=[570.0, 575.0, 580.0, 585.0])
        enriched1 = enrich_chain_greeks(chain1, r=0.05)
        enriched2 = enrich_chain_greeks(chain2, r=0.05)
        p1 = compute_gamma_profile(enriched1, r=0.05)
        p2 = compute_gamma_profile(enriched2, r=0.05)
        wall = find_aggregate_gamma_wall([p1, p2])
        assert wall is not None
