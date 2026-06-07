"""
core/ev_engine.py — Cross-Market Expected Value (EV) Engine
============================================================
Detects structural mispricings between:
  - Options-implied probability (derived via Black-Scholes inversion)
  - Prediction market consensus probability (Polymarket, Kalshi, etc.)

Also computes:
  - Raw EV spread (delta between the two probabilities)
  - Kelly Criterion optimal fraction for exploiting the spread
  - Confidence-adjusted significance flag

All functions are pure / side-effect-free.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from options_agent.core.schemas import (
    EVDiscrepancy,
    OptionContract,
    OptionsChain,
    OptionType,
    PredictionMarketContract,
)
from options_agent.core.greeks_engine import bs_price, _MIN_T


# ---------------------------------------------------------------------------
# Black-Scholes implied probability
# ---------------------------------------------------------------------------

def bs_risk_neutral_prob(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Extract the risk-neutral probability that the underlying finishes
    in-the-money at expiry.

    For a call:  P(S_T > K) = N(d2)
    For a put:   P(S_T < K) = N(-d2)

    This is the "digital option" probability — the true market-implied
    probability under the risk-neutral measure.
    """
    T = max(T, _MIN_T)
    sigma = max(sigma, 1e-6)
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    if option_type == OptionType.CALL:
        return float(norm.cdf(d2))
    else:
        return float(norm.cdf(-d2))


def extract_otm_implied_prob(
    chain: OptionsChain,
    r: float,
    target_moneyness: float = 1.05,   # e.g. 1.05 = 5% OTM call
    option_type: OptionType = OptionType.CALL,
) -> tuple[Optional[float], Optional[OptionContract]]:
    """
    Find the OTM contract closest to `target_moneyness * spot` and
    extract the implied probability of expiry ITM.

    Returns (probability, contract_used) or (None, None) if unavailable.
    """
    S = chain.spot_price
    target_strike = S * target_moneyness

    candidates = [
        c for c in chain.contracts
        if c.option_type == option_type
        and c.implied_volatility is not None
        and c.implied_volatility > 0
    ]

    if not candidates:
        return None, None

    nearest = min(candidates, key=lambda c: abs(c.strike - target_strike))
    T = max(nearest.days_to_expiry / 365.0, _MIN_T)
    iv = nearest.implied_volatility

    prob = bs_risk_neutral_prob(S, nearest.strike, T, r, iv, option_type)
    return prob, nearest


# ---------------------------------------------------------------------------
# Prediction market probability extraction
# ---------------------------------------------------------------------------

def find_matching_prediction_contract(
    ticker: str,
    prediction_contracts: list[PredictionMarketContract],
    keywords: Optional[list[str]] = None,
) -> Optional[PredictionMarketContract]:
    """
    Find the prediction market contract most likely to correspond to
    the same binary outcome as our options position.

    Matching heuristic:
      1. Contract's ticker_ref == ticker
      2. Optional: question contains any of `keywords`
    Scores by volume to prefer the most liquid market.
    """
    matches = [c for c in prediction_contracts if c.ticker_ref == ticker]
    if not matches:
        return None

    if keywords:
        kw_lower = [k.lower() for k in keywords]
        scored = [
            c for c in matches
            if any(kw in c.question.lower() for kw in kw_lower)
        ]
        if scored:
            matches = scored

    # Prefer highest volume (most liquid = tightest spread)
    return max(matches, key=lambda c: c.volume_24h)


# ---------------------------------------------------------------------------
# EV Discrepancy Computation
# ---------------------------------------------------------------------------

def compute_ev_discrepancy(
    ticker: str,
    chain: OptionsChain,
    prediction_contracts: list[PredictionMarketContract],
    r: float,
    min_spread_pct: float = 0.05,
    target_moneyness: float = 1.05,
    option_type: OptionType = OptionType.CALL,
) -> Optional[EVDiscrepancy]:
    """
    Core EV computation:

      EV_spread = P_options - P_prediction

      > 0 → Options market is MORE optimistic than prediction market.
            → Options are pricing in higher probability → relatively expensive.
            → Potential short premium opportunity.

      < 0 → Options are pricing in LOWER probability than prediction market.
            → Options are cheap relative to the event probability.
            → Potential long premium opportunity.

    Returns None if insufficient data.
    """
    options_prob, contract_used = extract_otm_implied_prob(
        chain, r, target_moneyness, option_type
    )
    if options_prob is None or contract_used is None:
        return None

    prediction_contract = find_matching_prediction_contract(ticker, prediction_contracts)
    if prediction_contract is None:
        return None

    pred_prob = prediction_contract.implied_probability
    ev_spread = options_prob - pred_prob
    ev_spread_pct = abs(ev_spread)

    # Kelly Criterion
    kelly = _kelly_fraction(options_prob, pred_prob)

    is_significant = ev_spread_pct >= min_spread_pct

    return EVDiscrepancy(
        ticker=ticker,
        option_implied_prob=options_prob,
        prediction_market_prob=pred_prob,
        ev_spread=ev_spread,
        ev_spread_pct=ev_spread_pct,
        is_significant=is_significant,
        option_contract_ref=f"{contract_used.ticker}_{contract_used.expiration}_{contract_used.strike}_{contract_used.option_type.value}",
        prediction_market_ref=f"{prediction_contract.market_name}:{prediction_contract.contract_id}",
        kelly_fraction=kelly,
        computed_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def _kelly_fraction(
    our_prob_estimate: float,
    market_implied_odds: float,
) -> Optional[float]:
    """
    Kelly Criterion for binary bet:

      f* = (p * b - q) / b

    where:
      p = our edge probability
      q = 1 - p
      b = (1 / market_implied_odds) - 1  (decimal odds minus 1)

    market_implied_odds is the "price" from prediction markets (0 to 1).
    Returns None if the Kelly fraction is negative (no edge).
    """
    p = float(np.clip(our_prob_estimate, 1e-6, 1 - 1e-6))
    q = 1.0 - p
    odds_price = float(np.clip(market_implied_odds, 1e-6, 1 - 1e-6))

    # Decimal odds: b = (1/price) - 1
    b = (1.0 / odds_price) - 1.0
    if b <= 0:
        return None

    kelly = (p * b - q) / b
    if kelly <= 0:
        return None

    # Cap at 25% — full Kelly is too aggressive for live trading
    return float(min(kelly, 0.25))


# ---------------------------------------------------------------------------
# Multi-expiry EV sweep
# ---------------------------------------------------------------------------

def sweep_ev_across_expirations(
    ticker: str,
    chains: list[OptionsChain],
    prediction_contracts: list[PredictionMarketContract],
    r: float,
    min_spread_pct: float = 0.05,
) -> list[EVDiscrepancy]:
    """
    Compute EV discrepancy for each available expiration.
    Returns only significant findings (above threshold).
    """
    results = []
    for chain in chains:
        for opt_type in [OptionType.CALL, OptionType.PUT]:
            moneyness = 1.05 if opt_type == OptionType.CALL else 0.95
            disc = compute_ev_discrepancy(
                ticker=ticker,
                chain=chain,
                prediction_contracts=prediction_contracts,
                r=r,
                min_spread_pct=min_spread_pct,
                target_moneyness=moneyness,
                option_type=opt_type,
            )
            if disc and disc.is_significant:
                results.append(disc)
    return results


# ---------------------------------------------------------------------------
# Arbitrage narrative builder
# ---------------------------------------------------------------------------

def build_ev_narrative(disc: EVDiscrepancy) -> str:
    """
    Generate a human-readable trading narrative for a given EV discrepancy.
    """
    direction = "overpriced" if disc.ev_spread > 0 else "underpriced"
    action = "consider selling premium (credit spread / iron condor)" if disc.ev_spread > 0 \
             else "consider buying premium (debit spread / long straddle)"

    kelly_str = f"  Kelly fraction: {disc.kelly_fraction:.1%}" if disc.kelly_fraction else ""

    return (
        f"Cross-market EV discrepancy detected on {disc.ticker}.\n"
        f"  Options-implied probability:    {disc.option_implied_prob:.1%}\n"
        f"  Prediction market probability:  {disc.prediction_market_prob:.1%}\n"
        f"  EV spread:                      {disc.ev_spread:+.1%} ({disc.ev_spread_pct:.1%} abs)\n"
        f"  Options appear {direction} relative to prediction markets.\n"
        f"  Suggested action: {action}.\n"
        f"{kelly_str}"
    ).strip()
