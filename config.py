"""
config.py — Central Parameterized Configuration
==============================================
Separates spot data (quotes) from options data (chains) into independent
provider configurations. Supports batching for rate-limit management.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Risk Thresholds
# ---------------------------------------------------------------------------

class RiskThresholds(BaseModel):
    gamma_wall_proximity_pct: float = Field(0.02)
    iv_crush_iv_rank_threshold: float = Field(0.70)
    iv_crush_hours_before_event: int = Field(48)
    ev_min_spread_pct: float = Field(0.05)
    zero_gamma_breach_buffer_pct: float = Field(0.005)
    max_pain_deviation_alert_pct: float = Field(0.05)
    max_consecutive_errors: int = Field(5)


# ---------------------------------------------------------------------------
# Data Provider Config
# ---------------------------------------------------------------------------

class SpotDataProviderConfig(BaseModel):
    """独立配置的现货/报价数据源"""
    name: str = "alpaca"
    base_url: str = ""
    api_key: str = ""
    api_secret: str = ""
    rate_limit_per_minute: int = Field(900)   # 15 req/s = 900 req/min
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_base: float = 2.0


class OptionsDataProviderConfig(BaseModel):
    """独立配置的期权数据源"""
    name: str = "alpaca"
    base_url: str = ""
    api_key: str = ""
    api_secret: str = ""
    rate_limit_per_minute: int = Field(900)   # 15 req/s = 900 req/min
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_base: float = 2.0


class PolygonConfig(BaseModel):
    """Polygon.io 配置（用于 IV 历史）"""
    name: str = "polygon"
    base_url: str = ""
    api_key: str = ""
    rate_limit_per_minute: int = Field(100)
    timeout_seconds: float = 10.0


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


# ---------------------------------------------------------------------------
# Watchlist Entry
# ---------------------------------------------------------------------------

class WatchlistEntry(BaseModel):
    ticker: str
    enabled: bool = True
    category: str = "general"
    priority: str = "medium"   # high, medium, low
    description: str = ""


class WatchlistConfig(BaseModel):
    watchlist: list[WatchlistEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Main Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core behavior ──────────────────────────────────────────────
    poll_interval_seconds: float = Field(30.0)   # 每轮完整轮询的间隔
    log_level: str = Field("INFO")
    dry_run: bool = Field(True)
    use_mock_data: bool = Field(True)

    # ── Watchlist ──────────────────────────────────────────────────
    # 支持两种模式：
    #   1. 旧模式：WATCH_TICKERS=SPY,QQQ,AAPL（逗号分隔字符串）
    #   2. 新模式：WATCHLIST_PATH=watchlist.yaml（文件路径）
    watch_tickers: Any = Field(default="SPY,QQQ")   # 旧兼容模式
    watchlist_path: str = Field(default="")          # 新模式，指向 watchlist.yaml

    # ── Batch Configuration ─────────────────────────────────────────
    # 用于控制 Alpaca 速率限制（15 req/s）
    # Alpaca Free Tier: 15 req/s 总限制（含 spot 和 options）
    # 推荐（20只股票，5s轮询）:
    #   BATCH_SIZE=5       → 每批5只股票
    #   BATCH_INTERVAL=1.25 → 批次间隔1.25秒
    #   计算: 5只 × 2次(spot+options) = 10 req
    #         10 req / 15 req/s = 0.67s，留足余量
    batch_size: int = Field(5)       # 每批多少只股票
    batch_interval_seconds: float = Field(1.25)   # 批次之间的间隔

    # ── Spot Data Provider（现货/报价）───────────────────────────────
    # 独立配置，用于 fetch_spot_quote
    spot_data_provider: SpotDataProviderConfig = Field(
        default_factory=lambda: SpotDataProviderConfig(
            name="alpaca",
            base_url="https://data.alpaca.markets/v2",
            api_key=os.getenv("ALPACA_API_KEY", ""),
            api_secret=os.getenv("ALPACA_API_SECRET", ""),
            rate_limit_per_minute=900,
        )
    )

    # ── Options Data Provider（期权链）───────────────────────────────
    # 独立配置，用于 fetch_options_chain
    options_data_provider: OptionsDataProviderConfig = Field(
        default_factory=lambda: OptionsDataProviderConfig(
            name="alpaca",
            base_url="https://data.alpaca.markets/v2",
            api_key=os.getenv("ALPACA_API_KEY", ""),
            api_secret=os.getenv("ALPACA_API_SECRET", ""),
            rate_limit_per_minute=900,
        )
    )

    # ── Polygon（IV 历史）───────────────────────────────────────────
    polygon_config: PolygonConfig = Field(
        default_factory=lambda: PolygonConfig(
            base_url="https://api.polygon.io/v2",
            api_key=os.getenv("POLYGON_API_KEY", ""),
            rate_limit_per_minute=100,
        )
    )

    # ── Options parameters ──────────────────────────────────────────
    option_expiry_lookahead_days: int = Field(45)
    option_moneyness_range_pct: float = Field(0.15)
    risk_free_rate: float = Field(0.0525)
    iv_history_days: int = Field(252)

    # ── Prediction Markets ─────────────────────────────────────────
    prediction_markets: list[PredictionMarketConfig] = Field(
        default_factory=lambda: [
            PredictionMarketConfig(name="polymarket", base_url="https://clob.polymarket.com"),
            PredictionMarketConfig(name="kalshi",     base_url="https://trading-api.kalshi.com/trade-api/v2"),
        ]
    )

    # ── Catalyst Calendar ──────────────────────────────────────────
    catalyst_calendar: list[CatalystEvent] = Field(
        default_factory=lambda: [
            CatalystEvent(ticker="MACRO", event_type="FED",
                          event_time_utc="2026-07-30T18:00:00Z", description="FOMC Rate Decision"),
            CatalystEvent(ticker="MACRO", event_type="CPI",
                          event_time_utc="2026-07-15T12:30:00Z", description="US CPI June Print"),
        ]
    )

    # ── Risk thresholds ────────────────────────────────────────────
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)

    # ── Alert webhooks ────────────────────────────────────────────
    alert_webhook: WebhookConfig = Field(
        default_factory=lambda: WebhookConfig(
            enabled=bool(os.getenv("ALPACA_API_KEY", "") and os.getenv("ALERT_WEBHOOK_URL", "")),
            url=os.getenv("ALERT_WEBHOOK_URL", ""),
            secret_value=os.getenv("ALERT_WEBHOOK_SECRET", ""),
        )
    )
    slack_webhook_url: str = Field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", ""))

    # ── Validators ────────────────────────────────────────────────

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

    # ── Computed properties ────────────────────────────────────────

    def get_enabled_tickers(self) -> list[str]:
        """Return enabled tickers from watchlist file or legacy env var."""
        if self.watchlist_path:
            return self._load_watchlist_tickers()
        # Legacy mode: parse from WATCH_TICKERS env var
        return self.watch_tickers if isinstance(self.watch_tickers, list) else [self.watch_tickers]

    def _load_watchlist_tickers(self) -> list[str]:
        """Load and parse watchlist.yaml, return enabled tickers sorted by priority."""
        try:
            path = Path(self.watchlist_path)
            if not path.exists():
                # Try relative to this config file
                config_dir = Path(__file__).parent
                path = config_dir / self.watchlist_path

            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                watchlist = WatchlistConfig(**data)
                enabled = [e.ticker.upper() for e in watchlist.watchlist if e.enabled]

                # Sort: high → medium → low
                priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
                def sort_key(ticker: str) -> tuple[int, int]:
                    entry = next((e for e in watchlist.watchlist if e.ticker.upper() == ticker), None)
                    priority = (entry.priority.upper() if entry else "MEDIUM")
                    return (priority_order.get(priority, 1), ticker)
                    enabled.sort(key=sort_key)
                return enabled
        except Exception:
            pass
        return self.watch_tickers if isinstance(self.watch_tickers, list) else [self.watch_tickers]

    def get_batches(self) -> list[list[str]]:
        """Split enabled tickers into batches."""
        tickers = self.get_enabled_tickers()
        batch_size = self.batch_size
        return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]


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
