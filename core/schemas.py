"""
core/schemas.py — Canonical Pydantic Data Models
=================================================
All inter-agent data contracts are defined here.
No ticker is embedded in any model — tickers are always explicit fields.
"""

from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class GammaRegime(str, Enum):
    LONG_GAMMA = "LONG_GAMMA"   # MM is long gamma → stabilizing, mean-reverting
    SHORT_GAMMA = "SHORT_GAMMA" # MM is short gamma → amplifying, trending
    NEUTRAL = "NEUTRAL"


class DataQuality(str, Enum):
    GOOD = "GOOD"
    STALE = "STALE"       # timestamp lag exceeds threshold
    SPARSE = "SPARSE"     # insufficient strikes/expirations
    INVALID = "INVALID"   # failed schema validation


# ---------------------------------------------------------------------------
# Layer 0 — Raw Market Data
# ---------------------------------------------------------------------------

class SpotQuote(BaseModel):
    """Real-time best-bid/ask + last for an underlying."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    spot_price: float = Field(gt=0)
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    last: float = Field(gt=0)
    volume: int = Field(ge=0)
    timestamp_utc: datetime
    source: str = ""  # data provider name

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class OptionContract(BaseModel):
    """A single option contract row from the chain."""
    model_config = ConfigDict(frozen=True)

    ticker: str
    expiration: date
    strike: float = Field(gt=0)
    option_type: OptionType

    # Market data
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    last: float = Field(ge=0)
    volume: int = Field(ge=0)
    open_interest: int = Field(ge=0)

    # Implied vol and Greeks (may be None if not yet computed)
    implied_volatility: Optional[float] = Field(None, ge=0)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    timestamp_utc: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def days_to_expiry(self) -> float:
        today = datetime.utcnow().date()
        return max((self.expiration - today).days, 0)


class OptionsChain(BaseModel):
    """Full option chain snapshot for one ticker + expiration."""

    ticker: str
    expiration: date
    spot_price: float = Field(gt=0)
    contracts: list[OptionContract] = Field(default_factory=list)
    timestamp_utc: datetime
    data_quality: DataQuality = DataQuality.GOOD
    quality_notes: str = ""

    @property
    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type == OptionType.CALL]

    @property
    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type == OptionType.PUT]

    @property
    def strikes(self) -> list[float]:
        return sorted({c.strike for c in self.contracts})


# ---------------------------------------------------------------------------
# Layer 1 — Alternative / Prediction Market Data
# ---------------------------------------------------------------------------

class PredictionMarketContract(BaseModel):
    """A binary outcome contract from a prediction market."""
    model_config = ConfigDict(frozen=True)

    market_name: str        # e.g. "polymarket"
    contract_id: str
    question: str           # human-readable: "Will SPY close above 580 by July 31?"
    ticker_ref: str         # underlying ticker this maps to (e.g. "SPY")
    yes_price: float = Field(ge=0.0, le=1.0)   # implied probability of YES
    no_price: float = Field(ge=0.0, le=1.0)
    volume_24h: float = Field(ge=0)
    timestamp_utc: datetime

    @property
    def implied_probability(self) -> float:
        """Consensus market probability of the YES outcome."""
        return self.yes_price


class CatalystSignal(BaseModel):
    """A known upcoming catalyst from the calendar."""

    ticker: str
    event_type: str
    event_time_utc: datetime
    description: str = ""
    hours_until_event: float = 0.0
    is_imminent: bool = False  # True if within iv_crush_hours_before_event


# ---------------------------------------------------------------------------
# Layer 2 — Greeks & Risk Matrices (output of Agent B)
# ---------------------------------------------------------------------------

class GammaProfile(BaseModel):
    """Net gamma exposure across all strikes for a single expiration."""

    ticker: str
    expiration: date
    spot_price: float

    # Strike → net gamma (calls OI * Γ_call + puts OI * Γ_put) mapping
    gamma_by_strike: dict[float, float] = Field(default_factory=dict)

    gamma_wall_strike: Optional[float] = None    # Strike with max net gamma
    zero_gamma_strike: Optional[float] = None    # Regime flip point
    max_pain_strike: Optional[float] = None

    gamma_regime: GammaRegime = GammaRegime.NEUTRAL
    net_gamma_at_spot: float = 0.0               # Interpolated gamma at current spot

    computed_at: datetime = Field(default_factory=datetime.utcnow)


class IVAnalysis(BaseModel):
    """Implied Volatility regime metrics for one ticker."""

    ticker: str
    current_iv: float = Field(ge=0)          # ATM IV of front-month
    iv_rank: float = Field(ge=0, le=1)       # 0–1 percentile over history window
    iv_percentile: float = Field(ge=0, le=1)
    iv_52w_high: float = Field(ge=0)
    iv_52w_low: float = Field(ge=0)

    is_iv_elevated: bool = False             # rank > threshold
    iv_crush_risk: bool = False              # elevated + imminent catalyst
    iv_crush_risk_reason: str = ""

    computed_at: datetime = Field(default_factory=datetime.utcnow)


class EVDiscrepancy(BaseModel):
    """Cross-market Expected Value mispricing signal."""

    ticker: str
    option_implied_prob: float = Field(ge=0, le=1)  # from BS inversion
    prediction_market_prob: float = Field(ge=0, le=1)
    ev_spread: float           # option_implied_prob - prediction_market_prob
    ev_spread_pct: float       # absolute spread as fraction
    is_significant: bool = False

    option_contract_ref: str = ""   # contract identifier used
    prediction_market_ref: str = "" # contract identifier used

    kelly_fraction: Optional[float] = None  # optimal position size via Kelly

    computed_at: datetime = Field(default_factory=datetime.utcnow)


class RiskMatrix(BaseModel):
    """Aggregated risk snapshot for one ticker — output of Agent B, input to Agent C."""

    ticker: str
    spot_price: float
    timestamp_utc: datetime

    gamma_profiles: list[GammaProfile] = Field(default_factory=list)
    iv_analysis: Optional[IVAnalysis] = None
    ev_discrepancies: list[EVDiscrepancy] = Field(default_factory=list)
    catalyst_signals: list[CatalystSignal] = Field(default_factory=list)

    # Composite flags
    gamma_wall_breach: bool = False
    zero_gamma_breach: bool = False
    iv_crush_imminent: bool = False
    ev_arb_detected: bool = False


# ---------------------------------------------------------------------------
# Layer 3 — Risk Alert (output of Agent C)
# ---------------------------------------------------------------------------

class RiskAlert(BaseModel):
    """Structured alert compiled by Agent C and dispatched via webhooks."""

    alert_id: str
    ticker: str
    risk_level: RiskLevel
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)

    headline: str
    narrative: str              # Full trading narrative for the operator
    recommended_action: str     # Actionable recommendation

    # Supporting evidence
    gamma_wall_strike: Optional[float] = None
    zero_gamma_strike: Optional[float] = None
    max_pain_strike: Optional[float] = None
    spot_price: float = 0.0
    iv_rank: Optional[float] = None
    ev_spread_pct: Optional[float] = None
    hours_to_catalyst: Optional[float] = None

    triggered_rules: list[str] = Field(default_factory=list)
    raw_matrix: Optional[RiskMatrix] = None  # attach for downstream audit

    def to_slack_blocks(self) -> list[dict]:
        """Serialize to Slack Block Kit for rich notifications."""
        emoji = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "🚨"}[self.risk_level]
        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} [{self.risk_level}] {self.ticker} — {self.headline}",
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": self.narrative},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Action:*\n{self.recommended_action}"},
                    {"type": "mrkdwn", "text": f"*Spot:* ${self.spot_price:.2f}"},
                    *([{"type": "mrkdwn", "text": f"*IV Rank:* {self.iv_rank:.1%}"}] if self.iv_rank else []),
                    *([{"type": "mrkdwn", "text": f"*EV Spread:* {self.ev_spread_pct:.1%}"}] if self.ev_spread_pct else []),
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🕐 {self.timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')} | Rules: {', '.join(self.triggered_rules)}",
                    }
                ],
            },
        ]


# ---------------------------------------------------------------------------
# Layer 4 — Agent-to-Agent message bus contracts
# ---------------------------------------------------------------------------

class AgentAOutput(BaseModel):
    """What Agent A emits onto the A→B queue each cycle."""

    cycle_id: str
    tickers_processed: list[str]
    spot_quotes: dict[str, SpotQuote]          # ticker → quote
    options_chains: dict[str, list[OptionsChain]]  # ticker → [chain per expiry]
    prediction_contracts: list[PredictionMarketContract]
    catalyst_signals: list[CatalystSignal]
    latency_ms: float
    data_quality_summary: dict[str, DataQuality]  # ticker → quality
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)


class AgentBOutput(BaseModel):
    """What Agent B emits onto the B→C queue each cycle."""

    cycle_id: str
    risk_matrices: dict[str, RiskMatrix]  # ticker → matrix
    anomalies_detected: bool
    anomaly_summary: str = ""
    compute_time_ms: float
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)


class AgentCOutput(BaseModel):
    """Final output of Agent C — alerts dispatched and actions taken."""

    cycle_id: str
    alerts_fired: list[RiskAlert]
    webhooks_dispatched: int
    simulated_orders: list[dict]  # dry-run order representations
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# System-level heartbeat
# ---------------------------------------------------------------------------

class SystemHeartbeat(BaseModel):
    cycle_id: str
    cycle_start_utc: datetime
    cycle_end_utc: datetime
    duration_ms: float
    agent_a_ok: bool
    agent_b_ok: bool
    agent_c_ok: bool
    alerts_this_cycle: int
    errors_this_cycle: int
    consecutive_error_count: int
