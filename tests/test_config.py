"""
tests/test_config.py — Unit tests for Settings and watchlist loading
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from options_agent.config import (
    RiskThresholds,
    Settings,
    WatchlistConfig,
    WatchlistEntry,
    reload_settings,
)


class TestSettingsDefaults:
    def test_defaults_are_safe(self):
        os.environ["USE_MOCK_DATA"] = "true"
        os.environ["DRY_RUN"] = "true"
        os.environ.pop("WATCH_TICKERS", None)
        os.environ.pop("WATCHLIST_PATH", None)
        s = reload_settings()
        assert s.dry_run is True
        assert s.use_mock_data is True
        assert s.poll_interval_seconds > 0
        assert s.batch_size > 0
        assert s.risk_free_rate > 0

    def test_risk_thresholds_sane(self):
        s = reload_settings()
        rt = s.risk_thresholds
        assert 0 < rt.gamma_wall_proximity_pct < 0.10
        assert 0.5 < rt.iv_crush_iv_rank_threshold < 1.0
        assert rt.iv_crush_hours_before_event > 0
        assert rt.max_consecutive_errors > 0

    def test_tickers_parsed_from_env(self):
        os.environ["WATCH_TICKERS"] = "SPY,QQQ,AAPL"
        s = reload_settings()
        assert s.get_enabled_tickers() == ["SPY", "QQQ", "AAPL"]

    def test_ticker_parsing_case_insensitive(self):
        os.environ["WATCH_TICKERS"] = "spy,qqq"
        s = reload_settings()
        tickers = s.get_enabled_tickers()
        assert "SPY" in tickers
        assert "QQQ" in tickers

    def test_invalid_log_level_raises(self):
        with pytest.raises(Exception):
            os.environ["LOG_LEVEL"] = "NOTAVALIDLEVEL"
            reload_settings()

    def teardown_method(self):
        # Reset env
        os.environ.pop("LOG_LEVEL", None)
        os.environ.pop("WATCH_TICKERS", None)


class TestWatchlistLoading:
    def _make_watchlist_file(self, entries: list[dict]) -> str:
        """Write a temp watchlist.yaml and return its path."""
        data = {"watchlist": entries}
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        yaml.dump(data, f)
        f.flush()
        return f.name

    def test_loads_enabled_tickers(self):
        path = self._make_watchlist_file([
            {"ticker": "SPY", "enabled": True, "priority": "high"},
            {"ticker": "QQQ", "enabled": True, "priority": "medium"},
            {"ticker": "AAPL", "enabled": False, "priority": "low"},
        ])
        os.environ["WATCHLIST_PATH"] = path
        s = reload_settings()
        tickers = s.get_enabled_tickers()
        assert "SPY" in tickers
        assert "QQQ" in tickers
        assert "AAPL" not in tickers
        os.unlink(path)

    def test_priority_sorting(self):
        path = self._make_watchlist_file([
            {"ticker": "AMD", "enabled": True, "priority": "low"},
            {"ticker": "NVDA", "enabled": True, "priority": "high"},
            {"ticker": "MSFT", "enabled": True, "priority": "medium"},
        ])
        os.environ["WATCHLIST_PATH"] = path
        s = reload_settings()
        tickers = s.get_enabled_tickers()
        # High priority should come first
        assert tickers.index("NVDA") < tickers.index("MSFT")
        assert tickers.index("MSFT") < tickers.index("AMD")
        os.unlink(path)

    def test_missing_watchlist_falls_back_to_env_tickers(self):
        os.environ["WATCHLIST_PATH"] = "/nonexistent/path/watchlist.yaml"
        os.environ["WATCH_TICKERS"] = "SPY,QQQ"
        s = reload_settings()
        tickers = s.get_enabled_tickers()
        assert "SPY" in tickers
        assert "QQQ" in tickers

    def teardown_method(self):
        os.environ.pop("WATCHLIST_PATH", None)
        os.environ.pop("WATCH_TICKERS", None)


class TestGetBatches:
    def test_single_batch_when_few_tickers(self):
        os.environ["WATCH_TICKERS"] = "SPY,QQQ"
        os.environ["BATCH_SIZE"] = "5"
        s = reload_settings()
        batches = s.get_batches()
        assert len(batches) == 1
        assert set(batches[0]) == {"SPY", "QQQ"}

    def test_multiple_batches(self):
        os.environ["WATCH_TICKERS"] = "SPY,QQQ,AAPL,NVDA,MSFT,TSLA"
        os.environ["BATCH_SIZE"] = "2"
        s = reload_settings()
        batches = s.get_batches()
        assert len(batches) == 3
        all_tickers = [t for batch in batches for t in batch]
        assert len(all_tickers) == 6

    def teardown_method(self):
        os.environ.pop("WATCH_TICKERS", None)
        os.environ.pop("BATCH_SIZE", None)
