"""
core/api_models.py — FastAPI REST Schemas
==========================================
Lightweight Pydantic models for the HTTP API layer.
These are separate from internal schemas.py to keep the API surface clean
and avoid asyncio.Queue references leaking into FastAPI route handlers.

All models are derived from core/schemas.py and use identical field names
so the frontend gets consistent snake_case JSON.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums (mirrors core/schemas.py)
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
    LONG_GAMMA = "LONG_GAMMA"
    SHORT_GAMMA = "SHORT_GAMMA"
    NEUTRAL = "NEUTRAL"


class DataQuality(str, Enum):
    GOOD = "GOOD"
    STALE = "STALE"
    SPARSE = "SPARSE"
    INVALID = "INVALID"


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    OK = "OK"
    FAIL = "FAIL"
    STANDBY = "STANDBY"


# ---------------------------------------------------------------------------
# Layer 0 — Market Data (API shape)
# ---------------------------------------------------------------------------

class SpotQuoteAPI(BaseModel):
    ticker: str
    spot_price: float
    bid: float
    ask: float
    last: float
    volume: int
    timestamp_utc: datetime
    source: str = ""

    model_config = {"from_attributes": True}


class OptionContractAPI(BaseModel):
    ticker: str
    expiration: date
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    timestamp_utc: datetime

    model_config = {"from_attributes": True}


class OptionsChainAPI(BaseModel):
    ticker: str
    expiration: date
    spot_price: float
    contracts: list[OptionContractAPI]
    timestamp_utc: datetime
    data_quality: DataQuality = DataQuality.GOOD
    quality_notes: str = ""

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Layer 1 — Prediction Markets (API shape)
# ---------------------------------------------------------------------------

class PredictionMarketContractAPI(BaseModel):
    market_name: str
    contract_id: str
    question: str
    ticker_ref: str
    yes_price: float
    no_price: float
    volume_24h: float
    timestamp_utc: datetime

    model_config = {"from_attributes": True}


class CatalystSignalAPI(BaseModel):
    ticker: str
    event_type: str
    event_time_utc: datetime
    description: str
    hours_until_event: float
    is_imminent: bool

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Layer 2 — Greeks & Risk (API shape)
# ---------------------------------------------------------------------------

class GammaProfileAPI(BaseModel):
    ticker: str
    expiration: date
    spot_price: float
    gamma_by_strike: dict[str, float] = Field(default_factory=dict)
    gamma_wall_strike: Optional[float] = None
    zero_gamma_strike: Optional[float] = None
    max_pain_strike: Optional[float] = None
    gamma_regime: GammaRegime
    net_gamma_at_spot: float
    computed_at: datetime

    model_config = {"from_attributes": True}


class IVAnalysisAPI(BaseModel):
    ticker: str
    current_iv: float
    iv_rank: float
    iv_percentile: float
    iv_52w_high: float
    iv_52w_low: float
    is_iv_elevated: bool
    iv_crush_risk: bool
    iv_crush_risk_reason: str
    computed_at: datetime

    model_config = {"from_attributes": True}


class EVDiscrepancyAPI(BaseModel):
    ticker: str
    option_implied_prob: float
    prediction_market_prob: float
    ev_spread: float
    ev_spread_pct: float
    is_significant: bool
    option_contract_ref: str
    prediction_market_ref: str
    kelly_fraction: Optional[float] = None
    computed_at: datetime

    model_config = {"from_attributes": True}


class RiskMatrixAPI(BaseModel):
    ticker: str
    spot_price: float
    timestamp_utc: datetime
    gamma_profiles: list[GammaProfileAPI]
    iv_analysis: Optional[IVAnalysisAPI] = None
    ev_discrepancies: list[EVDiscrepancyAPI]
    catalyst_signals: list[CatalystSignalAPI]
    gamma_wall_breach: bool
    zero_gamma_breach: bool
    iv_crush_imminent: bool
    ev_arb_detected: bool

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Layer 3 — Alerts (API shape)
# ---------------------------------------------------------------------------

class RiskAlertAPI(BaseModel):
    alert_id: str
    ticker: str
    risk_level: RiskLevel
    timestamp_utc: datetime
    headline: str
    narrative: str
    recommended_action: str
    gamma_wall_strike: Optional[float] = None
    zero_gamma_strike: Optional[float] = None
    max_pain_strike: Optional[float] = None
    spot_price: float
    iv_rank: Optional[float] = None
    ev_spread_pct: Optional[float] = None
    hours_to_catalyst: Optional[float] = None
    triggered_rules: list[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Agent status (API shape)
# ---------------------------------------------------------------------------

class AgentInfoAPI(BaseModel):
    name: str
    status: AgentStatus
    last_cycle_id: Optional[str] = None
    last_cycle_ms: Optional[float] = None
    last_active_utc: Optional[datetime] = None
    consecutive_errors: int = 0
    cycle_count: int = 0


# ---------------------------------------------------------------------------
# System heartbeat (API shape)
# ---------------------------------------------------------------------------

class SystemHeartbeatAPI(BaseModel):
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

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Dashboard snapshot (聚合端点)
# ---------------------------------------------------------------------------

class DashboardSnapshotAPI(BaseModel):
    """Full state snapshot returned by GET /snapshot."""
    heartbeat: SystemHeartbeatAPI
    agents: list[AgentInfoAPI]
    tickers: list[str]
    spot_quotes: dict[str, SpotQuoteAPI]
    options_chains: dict[str, list[OptionsChainAPI]]
    risk_matrices: dict[str, RiskMatrixAPI]
    recent_alerts: list[RiskAlertAPI]
    uptime_seconds: float

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# WebSocket message types
# ---------------------------------------------------------------------------

class WSHeartbeat(BaseModel):
    """Lightweight heartbeat pushed via WebSocket every cycle."""
    type: str = "heartbeat"
    cycle_id: str
    duration_ms: float
    agent_a_ok: bool
    agent_b_ok: bool
    agent_c_ok: bool
    errors_this_cycle: int
    timestamp_utc: datetime


class WSSpotUpdate(BaseModel):
    """Single ticker spot price update pushed via WebSocket."""
    type: str = "spot_update"
    ticker: str
    spot_price: float
    bid: float
    ask: float
    timestamp_utc: datetime


class WSAlertFired(BaseModel):
    """New alert pushed via WebSocket when Agent C fires one."""
    type: str = "alert_fired"
    alert: RiskAlertAPI
    timestamp_utc: datetime
