"""
core/iv_analytics.py — Implied Volatility Regime Analytics
===========================================================
Computes:
  - IV Rank (current IV relative to 52-week high/low)
  - IV Percentile (fraction of days IV was lower than today)
  - Pre-event premium bloat detection
  - IV Crush risk flag (T-48h before catalyst)

All computations are pure functions operating on numpy arrays.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from options_agent.core.schemas import (
    CatalystSignal,
    IVAnalysis,
    OptionsChain,
    OptionType,
)


# ---------------------------------------------------------------------------
# ATM IV extraction from a chain
# ---------------------------------------------------------------------------

def extract_atm_iv(chain: OptionsChain) -> Optional[float]:
    """
    Return the at-the-money implied volatility from front-month chain.
    Uses the average of ATM call IV and ATM put IV (put-call parity midpoint).
    Selects the strike closest to current spot.
    """
    S = chain.spot_price
    calls = [c for c in chain.contracts if c.option_type == OptionType.CALL and c.implied_volatility]
    puts  = [c for c in chain.contracts if c.option_type == OptionType.PUT  and c.implied_volatility]

    if not calls and not puts:
        return None

    def nearest_iv(contracts: list) -> Optional[float]:
        if not contracts:
            return None
        nearest = min(contracts, key=lambda c: abs(c.strike - S))
        return nearest.implied_volatility

    call_iv = nearest_iv(calls)
    put_iv  = nearest_iv(puts)

    ivs = [iv for iv in [call_iv, put_iv] if iv is not None]
    return float(np.mean(ivs)) if ivs else None


def extract_skew(chain: OptionsChain) -> dict[str, float]:
    """
    Return basic skew metrics:
      - '25d_put_call_skew': 25-delta put IV minus 25-delta call IV
      - 'atm_iv': ATM implied vol
    """
    S = chain.spot_price
    result: dict[str, float] = {}

    atm_iv = extract_atm_iv(chain)
    if atm_iv:
        result["atm_iv"] = atm_iv

    # Find contracts nearest to 25-delta
    def nearest_delta_iv(contracts: list, target_delta: float) -> Optional[float]:
        eligible = [c for c in contracts if c.delta is not None and c.implied_volatility]
        if not eligible:
            return None
        nearest = min(eligible, key=lambda c: abs(abs(c.delta) - abs(target_delta)))
        return nearest.implied_volatility

    calls = [c for c in chain.contracts if c.option_type == OptionType.CALL]
    puts  = [c for c in chain.contracts if c.option_type == OptionType.PUT]

    call_25d_iv = nearest_delta_iv(calls, 0.25)
    put_25d_iv  = nearest_delta_iv(puts, -0.25)

    if call_25d_iv and put_25d_iv:
        result["25d_put_call_skew"] = put_25d_iv - call_25d_iv

    return result


# ---------------------------------------------------------------------------
# IV Rank & Percentile
# ---------------------------------------------------------------------------

def compute_iv_rank(current_iv: float, iv_history: list[float]) -> float:
    """
    IV Rank = (current_iv - 52w_low) / (52w_high - 52w_low)
    Returns value in [0, 1]. 1 = at 52-week high.
    """
    if not iv_history or len(iv_history) < 5:
        return 0.5  # insufficient history — return neutral

    arr = np.array(iv_history, dtype=float)
    low  = float(np.min(arr))
    high = float(np.max(arr))

    if high - low < 1e-6:
        return 0.5  # flat history

    rank = (current_iv - low) / (high - low)
    return float(np.clip(rank, 0.0, 1.0))


def compute_iv_percentile(current_iv: float, iv_history: list[float]) -> float:
    """
    IV Percentile = fraction of historical days where IV was BELOW current IV.
    More robust than IV Rank as it's less sensitive to single extreme readings.
    """
    if not iv_history or len(iv_history) < 5:
        return 0.5

    arr = np.array(iv_history, dtype=float)
    pct = float(np.mean(arr < current_iv))
    return float(np.clip(pct, 0.0, 1.0))


def build_iv_history_from_chains(
    chains_history: list[OptionsChain],
) -> list[float]:
    """
    Construct a daily IV series from a historical list of chains.
    Extracts ATM IV from each chain snapshot.
    """
    ivs = []
    for chain in chains_history:
        iv = extract_atm_iv(chain)
        if iv is not None and iv > 0:
            ivs.append(iv)
    return ivs


# ---------------------------------------------------------------------------
# IV Crush Risk Detection
# ---------------------------------------------------------------------------

def detect_iv_crush_risk(
    iv_analysis_partial: dict,
    catalyst_signals: list[CatalystSignal],
    iv_rank_threshold: float = 0.70,
    hours_before_event: int = 48,
) -> tuple[bool, str]:
    """
    Returns (is_crush_risk, reason_string).

    IV Crush risk conditions (ALL must be met):
      1. IV Rank > iv_rank_threshold  (premium is bloated)
      2. A known catalyst exists within hours_before_event
      3. The catalyst is for this ticker or is macro-wide (ticker="MACRO")

    When both conditions align, selling premium (strangles/straddles) after
    the event is the textbook play. The RISK is if you're LONG premium and
    haven't exited — that's the crush.
    """
    iv_rank = iv_analysis_partial.get("iv_rank", 0.0)
    ticker  = iv_analysis_partial.get("ticker", "")

    is_elevated = iv_rank >= iv_rank_threshold
    if not is_elevated:
        return False, ""

    relevant_catalysts = [
        sig for sig in catalyst_signals
        if (sig.ticker == ticker or sig.ticker == "MACRO")
        and 0 <= sig.hours_until_event <= hours_before_event
    ]

    if relevant_catalysts:
        cat = relevant_catalysts[0]
        reason = (
            f"IV Rank {iv_rank:.1%} exceeds threshold {iv_rank_threshold:.0%}. "
            f"Catalyst '{cat.event_type}: {cat.description}' is in "
            f"{cat.hours_until_event:.1f}h. High IV crush risk post-event."
        )
        return True, reason

    return False, ""


# ---------------------------------------------------------------------------
# Full IVAnalysis builder
# ---------------------------------------------------------------------------

def build_iv_analysis(
    ticker: str,
    front_month_chain: OptionsChain,
    iv_history: list[float],
    catalyst_signals: list[CatalystSignal],
    iv_rank_threshold: float = 0.70,
    iv_crush_hours: int = 48,
) -> IVAnalysis:
    """
    Construct a complete IVAnalysis for one ticker.
    """
    current_iv = extract_atm_iv(front_month_chain)
    if current_iv is None or current_iv <= 0:
        # Can't compute without IV — return a sparse shell
        return IVAnalysis(
            ticker=ticker,
            current_iv=0.0,
            iv_rank=0.5,
            iv_percentile=0.5,
            iv_52w_high=0.0,
            iv_52w_low=0.0,
            is_iv_elevated=False,
            iv_crush_risk=False,
        )

    full_history = iv_history + [current_iv]
    arr = np.array(full_history, dtype=float)

    iv_rank = compute_iv_rank(current_iv, full_history)
    iv_pct  = compute_iv_percentile(current_iv, full_history)
    iv_high = float(np.max(arr))
    iv_low  = float(np.min(arr))

    is_elevated = iv_rank >= iv_rank_threshold

    crush_risk, crush_reason = detect_iv_crush_risk(
        {"iv_rank": iv_rank, "ticker": ticker},
        catalyst_signals,
        iv_rank_threshold=iv_rank_threshold,
        hours_before_event=iv_crush_hours,
    )

    return IVAnalysis(
        ticker=ticker,
        current_iv=current_iv,
        iv_rank=iv_rank,
        iv_percentile=iv_pct,
        iv_52w_high=iv_high,
        iv_52w_low=iv_low,
        is_iv_elevated=is_elevated,
        iv_crush_risk=crush_risk,
        iv_crush_risk_reason=crush_reason,
        computed_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Volatility term structure utilities
# ---------------------------------------------------------------------------

def compute_vix_term_structure(
    atm_ivs_by_dte: dict[int, float]
) -> dict[str, float]:
    """
    Given {DTE: ATM_IV} mapping across expirations, compute:
      - contango_ratio: IV(30d) / IV(60d)  > 1 → backwardation (vol risk)
      - term_slope: regression slope of IV vs DTE
    """
    if len(atm_ivs_by_dte) < 2:
        return {}

    dtes = np.array(sorted(atm_ivs_by_dte.keys()), dtype=float)
    ivs  = np.array([atm_ivs_by_dte[int(d)] for d in dtes], dtype=float)

    # Linear regression slope
    if len(dtes) >= 2:
        coeffs = np.polyfit(dtes, ivs, 1)
        slope = float(coeffs[0])
    else:
        slope = 0.0

    result = {"term_slope": slope}

    # 30d vs 60d contango ratio
    dtes_arr = np.array(list(atm_ivs_by_dte.keys()))
    nearest_30 = dtes_arr[np.argmin(np.abs(dtes_arr - 30))]
    nearest_60 = dtes_arr[np.argmin(np.abs(dtes_arr - 60))]
    iv_30 = atm_ivs_by_dte.get(int(nearest_30))
    iv_60 = atm_ivs_by_dte.get(int(nearest_60))

    if iv_30 and iv_60 and iv_60 > 1e-6:
        result["contango_ratio"] = iv_30 / iv_60

    return result
