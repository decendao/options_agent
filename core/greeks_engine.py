"""
core/greeks_engine.py — Black-Scholes Greeks Engine
====================================================
Vectorized computation of:
  - Full option chain Greeks (Δ, Γ, Θ, V, ρ)
  - Gamma Wall (strike with max net gamma exposure)
  - Zero Gamma Line (regime flip strike)
  - Max Pain (strike minimizing aggregate OI payout)
  - Net Dealer Gamma (sign-aware DGex)

All functions are pure (no side effects, no I/O) and operate on
numpy arrays for performance. Thread-safe and async-safe.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

from options_agent.core.schemas import (
    GammaProfile,
    GammaRegime,
    OptionContract,
    OptionsChain,
    OptionType,
)


# ---------------------------------------------------------------------------
# Black-Scholes primitives
# ---------------------------------------------------------------------------

_SQRT_2PI = math.sqrt(2 * math.pi)
_MIN_T = 1 / (365 * 24)  # 1 hour minimum to prevent div-by-zero at expiry


def _d1_d2(
    S: np.ndarray | float,
    K: np.ndarray | float,
    T: np.ndarray | float,
    r: float,
    sigma: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d1 and d2 for Black-Scholes. All inputs may be arrays."""
    T = np.maximum(T, _MIN_T)
    sigma = np.maximum(sigma, 1e-6)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Black-Scholes option price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == OptionType.CALL:
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == OptionType.CALL:
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma is identical for calls and puts."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = max(T, _MIN_T)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def bs_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """Theta per calendar day (not annualized)."""
    T = max(T, _MIN_T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    first_term = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == OptionType.CALL:
        theta_annual = first_term - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        theta_annual = first_term + r * K * math.exp(-r * T) * norm.cdf(-d2)
    return float(theta_annual / 365)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega per 1% change in IV."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = max(T, _MIN_T)
    return float(S * norm.pdf(d1) * math.sqrt(T) * 0.01)


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Newton-Raphson IV solver.
    Returns None if the market_price is below intrinsic or solver diverges.
    """
    T = max(T, _MIN_T)
    intrinsic = max(0.0, (S - K) if option_type == OptionType.CALL else (K - S))
    if market_price < intrinsic - 1e-4:
        return None

    sigma = 0.30  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma) * 100  # undo the 0.01 factor
        if abs(vega) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)
    return sigma if 0.001 < sigma < 20.0 else None


# ---------------------------------------------------------------------------
# Vectorized Greeks enrichment of an OptionsChain
# ---------------------------------------------------------------------------

def enrich_chain_greeks(chain: OptionsChain, r: float) -> OptionsChain:
    """
    Return a new OptionsChain with all Greeks computed (in-place copy).
    Uses mid-price for IV solving; falls back to existing IV if solver fails.
    """
    S = chain.spot_price
    today = datetime.utcnow().date()

    enriched_contracts: list[OptionContract] = []
    for contract in chain.contracts:
        T = max((contract.expiration - today).days, 0) / 365.0
        mid = contract.mid if (contract.bid > 0 or contract.ask > 0) else contract.last

        # Solve IV
        iv = contract.implied_volatility
        if iv is None or iv <= 0:
            iv = implied_vol_newton(mid, S, contract.strike, T, r, contract.option_type)

        if iv and iv > 0:
            delta = bs_delta(S, contract.strike, T, r, iv, contract.option_type)
            gamma = bs_gamma(S, contract.strike, T, r, iv)
            theta = bs_theta(S, contract.strike, T, r, iv, contract.option_type)
            vega  = bs_vega(S, contract.strike, T, r, iv)
        else:
            delta = gamma = theta = vega = None

        enriched_contracts.append(
            contract.model_copy(
                update=dict(
                    implied_volatility=iv,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                )
            )
        )

    return chain.model_copy(update={"contracts": enriched_contracts})


# ---------------------------------------------------------------------------
# Net Dealer Gamma (DGex) profile
# ---------------------------------------------------------------------------

def compute_gamma_profile(chain: OptionsChain, r: float) -> GammaProfile:
    """
    Compute net dealer gamma exposure at each strike.

    Dealer Gamma Convention (standard):
      - Dealers are SHORT calls (bought by retail/funds) → long gamma on calls
        Net dealer gamma for calls = +OI * Gamma * 100 (per contract = 100 shares)
      - Dealers are SHORT puts → long gamma on puts
        Net dealer gamma for puts = +OI * Gamma * 100
    When spot crosses zero-gamma line, dealer goes SHORT gamma (amplifying moves).

    Returns a GammaProfile with:
      - gamma_by_strike  : {strike: net_dealer_gamma_notional}
      - gamma_wall_strike: strike of maximum positive gamma cluster
      - zero_gamma_strike: first strike below spot where cumulative gamma flips sign
      - max_pain_strike  : strike minimizing aggregate OI dollar loss
      - gamma_regime     : LONG_GAMMA or SHORT_GAMMA relative to current spot
    """
    S = chain.spot_price
    strikes = sorted({c.strike for c in chain.contracts})
    if not strikes:
        return GammaProfile(
            ticker=chain.ticker,
            expiration=chain.expiration,
            spot_price=S,
            gamma_regime=GammaRegime.NEUTRAL,
        )

    # Build OI-weighted gamma at each strike
    gamma_map: dict[float, float] = {k: 0.0 for k in strikes}

    for contract in chain.contracts:
        gamma = contract.gamma
        oi = contract.open_interest
        if gamma is None or oi == 0:
            continue
        # Notional gamma in dollar terms (100 shares per contract)
        notional_gamma = gamma * oi * 100 * S  # $-gamma per 1% spot move
        gamma_map[contract.strike] = gamma_map.get(contract.strike, 0.0) + notional_gamma

    # Gamma Wall — strike with highest absolute positive gamma
    if gamma_map:
        gamma_wall = max(gamma_map, key=lambda k: gamma_map[k])
    else:
        gamma_wall = None

    # Zero Gamma Line — interpolate the strike where cumulative net gamma = 0
    sorted_strikes = np.array(sorted(gamma_map.keys()))
    gammas = np.array([gamma_map[k] for k in sorted_strikes])
    zero_gamma = _find_zero_crossing(sorted_strikes, gammas)

    # Gamma at spot (interpolated)
    net_gamma_at_spot = 0.0
    if len(sorted_strikes) >= 2:
        try:
            interp = interp1d(sorted_strikes, gammas, kind="linear", bounds_error=False, fill_value=0.0)
            net_gamma_at_spot = float(interp(S))
        except Exception:
            pass

    # Regime
    regime = (
        GammaRegime.LONG_GAMMA if net_gamma_at_spot >= 0
        else GammaRegime.SHORT_GAMMA
    )

    # Max Pain
    max_pain = _compute_max_pain(chain)

    return GammaProfile(
        ticker=chain.ticker,
        expiration=chain.expiration,
        spot_price=S,
        gamma_by_strike=gamma_map,
        gamma_wall_strike=gamma_wall,
        zero_gamma_strike=zero_gamma,
        max_pain_strike=max_pain,
        gamma_regime=regime,
        net_gamma_at_spot=net_gamma_at_spot,
        computed_at=datetime.utcnow(),
    )


def _find_zero_crossing(strikes: np.ndarray, gammas: np.ndarray) -> Optional[float]:
    """
    Find the interpolated strike where cumulative net gamma changes sign.
    Scans from below spot to above spot searching for sign change.
    """
    if len(gammas) < 2:
        return None
    for i in range(len(gammas) - 1):
        if gammas[i] * gammas[i + 1] <= 0:
            # Linear interpolation of zero crossing
            dk = strikes[i + 1] - strikes[i]
            dg = gammas[i + 1] - gammas[i]
            if abs(dg) < 1e-12:
                return float(strikes[i])
            frac = -gammas[i] / dg
            return float(strikes[i] + frac * dk)
    return None


def _compute_max_pain(chain: OptionsChain) -> Optional[float]:
    """
    Max Pain = the strike at expiry that causes maximum aggregate option loss
    to ALL option holders (i.e., maximum gain to option writers / MM).

    Algorithm: For each candidate strike K_i, compute the total payout
    across all calls and puts at all strikes if the underlying expires at K_i.
    Max pain = argmin(total_payout).
    """
    strikes = sorted({c.strike for c in chain.contracts})
    if not strikes:
        return None

    total_pain: dict[float, float] = {}
    for candidate in strikes:
        pain = 0.0
        for contract in chain.contracts:
            oi = contract.open_interest
            k = contract.strike
            if contract.option_type == OptionType.CALL:
                pain += max(0.0, candidate - k) * oi * 100
            else:
                pain += max(0.0, k - candidate) * oi * 100
        total_pain[candidate] = pain

    return min(total_pain, key=lambda k: total_pain[k])


# ---------------------------------------------------------------------------
# Multi-expiry aggregate gamma profile (term structure)
# ---------------------------------------------------------------------------

def aggregate_gamma_profiles(profiles: list[GammaProfile]) -> dict[float, float]:
    """
    Sum net dealer gamma across all expirations at each strike.
    Useful for term-structure-aware gamma wall detection.
    """
    aggregate: dict[float, float] = {}
    for profile in profiles:
        for strike, gamma in profile.gamma_by_strike.items():
            aggregate[strike] = aggregate.get(strike, 0.0) + gamma
    return aggregate


def find_aggregate_gamma_wall(profiles: list[GammaProfile]) -> Optional[float]:
    agg = aggregate_gamma_profiles(profiles)
    if not agg:
        return None
    return max(agg, key=lambda k: agg[k])
