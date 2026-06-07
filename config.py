"""
config.py — Central Parameterized Configuration
"""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskThresholds(BaseModel):
    gamma_wall_proximity_pct: float = Field(0.02)
    iv_crush_iv_rank_threshold: float = Field(0.70)
    iv_crush_hours_before_event: int = Field(48)
    ev_min_spread_pct: float = Field(0.05)
    zero_gamma_breach_buffer_pct: float = Field(0.005)
    max_pain_deviation_alert_pct: float = Field(0.05)
    max_consecutive_errors: int = Field(5)


class DataProviderConfig(BaseModel):
    name: str
    base_url: str
    ws_url: str = ""
    api_key: str = ""
    api_secret: str = ""
    rate_limit_per_minute: int = Field(200)
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_base: float = 2.0


class PredictionMarketConfig(BaseModel):
    name: str
    base_url: str
    poll_interval_seconds: float = 60.0
    timeout_seconds: float = 10.0


class WebhookConfig(BaseModel):
    enabled: bool = False
    url: str = ""
    secret_header: str = "X-Webhook-Secret"
    secret_value: str = ""
    timeout_seconds: float = 5.0


class CatalystEvent(BaseModel):
    ticker: str
    event_type: str
    event_time_utc: str
    description: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    poll_interval_seconds: float = Field(30.0)
    log_level: str = Field("INFO")
    dry_run: bool = Field(True)
    use_mock_data: bool = Field(True)

    # Declared as Any so pydantic-settings never tries to JSON-decode it.
    # The parse_tickers validator converts comma-sep strings or JSON arrays.
    watch_tickers: Any = Field(default="SPY,QQQ")

    option_expiry_lookahead_days: int = Field(45)
    option_moneyness_range_pct: float = Field(0.15)
    risk_free_rate: float = Field(0.0525)
    iv_history_days: int = Field(252)

    market_data_provider: DataProviderConfig = Field(
        default_factory=lambda: DataProviderConfig(
            name="alpaca",
            base_url="https://data.alpaca.markets/v2",
            ws_url="wss://stream.data.alpaca.markets/v2/iex",
            api_key=os.getenv("ALPACA_API_KEY", ""),
            api_secret=os.getenv("ALPACA_API_SECRET", ""),
            rate_limit_per_minute=200,
        )
    )

    polygon_config: DataProviderConfig = Field(
        default_factory=lambda: DataProviderConfig(
            name="polygon",
            base_url="https://api.polygon.io/v2",
            api_key=os.getenv("POLYGON_API_KEY", ""),
            rate_limit_per_minute=100,
        )
    )

    prediction_markets: list[PredictionMarketConfig] = Field(
        default_factory=lambda: [
            PredictionMarketConfig(name="polymarket", base_url="https://clob.polymarket.com"),
            PredictionMarketConfig(name="kalshi",     base_url="https://trading-api.kalshi.com/trade-api/v2"),
        ]
    )

    catalyst_calendar: list[CatalystEvent] = Field(
        default_factory=lambda: [
            CatalystEvent(ticker="MACRO", event_type="FED",
                          event_time_utc="2026-07-30T18:00:00Z", description="FOMC Rate Decision"),
            CatalystEvent(ticker="MACRO", event_type="CPI",
                          event_time_utc="2026-07-15T12:30:00Z", description="US CPI June Print"),
        ]
    )

    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)

    alert_webhook: WebhookConfig = Field(
        default_factory=lambda: WebhookConfig(
            enabled=bool(os.getenv("ALERT_WEBHOOK_URL")),
            url=os.getenv("ALERT_WEBHOOK_URL", ""),
            secret_value=os.getenv("ALERT_WEBHOOK_SECRET", ""),
        )
    )

    slack_webhook_url: str = Field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", ""))

    @field_validator("watch_tickers", mode="before")
    @classmethod
    def parse_tickers(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(t).strip().upper() for t in v if str(t).strip()]
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                import json as _j
                try:
                    return [str(t).strip().upper() for t in _j.loads(s) if str(t).strip()]
                except Exception:
                    pass
            return [t.strip().upper() for t in s.split(",") if t.strip()]
        return [str(t).upper() for t in v]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
