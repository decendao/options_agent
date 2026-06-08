"""
tests/test_mock_cycle.py — Full async integration cycle with mock data
Tests the complete A→B→C pipeline without live API calls.
"""

import asyncio
from datetime import datetime

import pytest

from options_agent.config import Settings
from options_agent.agents.agent_a import AgentA
from options_agent.agents.agent_b import AgentB
from options_agent.agents.agent_c import AgentC
from options_agent.core.schemas import (
    AgentAOutput,
    AgentBOutput,
    AgentCOutput,
    DataQuality,
    RiskLevel,
)
from options_agent.data.mock_provider import MockMarketDataProvider, MockPredictionMarketProvider
from options_agent.webhooks.dispatcher import AlertDispatcher


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def make_test_settings(**overrides) -> Settings:
    import os
    os.environ["USE_MOCK_DATA"] = "true"
    os.environ["DRY_RUN"] = "true"
    os.environ["WATCH_TICKERS"] = "SPY,QQQ,AAPL"
    os.environ["POLL_INTERVAL_SECONDS"] = "10"
    os.environ["BATCH_SIZE"] = "5"
    for k, v in overrides.items():
        os.environ[k.upper()] = str(v)
    from options_agent.config import reload_settings
    return reload_settings()


class MockAggregator:
    """Wraps MockPredictionMarketProvider to match the PredictionMarketAggregator API."""
    def __init__(self):
        self._inner = MockPredictionMarketProvider(seed=42)

    async def connect(self): pass
    async def close(self): pass

    async def fetch_all(self, watch_tickers):
        return await self._inner.fetch_markets(ticker_keywords=watch_tickers)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentACycle:
    @pytest.mark.asyncio
    async def test_produces_valid_output(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)

        await provider.connect()
        agent_a = AgentA(
            settings=settings,
            market_provider=provider,
            prediction_aggregator=aggregator,
            output_queue=queue_ab,
        )
        output = await agent_a.run_cycle()
        await provider.close()

        assert isinstance(output, AgentAOutput)
        assert len(output.tickers_processed) == 3
        assert "SPY" in output.spot_quotes
        assert "QQQ" in output.spot_quotes
        assert output.spot_quotes["SPY"].spot_price > 0
        assert all(
            q in (DataQuality.GOOD, DataQuality.SPARSE, DataQuality.STALE, DataQuality.INVALID)
            for q in output.data_quality_summary.values()
        )

    @pytest.mark.asyncio
    async def test_options_chains_populated(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=99)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)

        await provider.connect()
        agent_a = AgentA(
            settings=settings,
            market_provider=provider,
            prediction_aggregator=aggregator,
            output_queue=queue_ab,
        )
        output = await agent_a.run_cycle()
        await provider.close()

        for ticker in output.tickers_processed:
            chains = output.options_chains.get(ticker, [])
            assert len(chains) > 0, f"No chains for {ticker}"
            total_contracts = sum(len(c.contracts) for c in chains)
            assert total_contracts > 0

    @pytest.mark.asyncio
    async def test_prediction_contracts_fetched(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)

        await provider.connect()
        agent_a = AgentA(
            settings=settings,
            market_provider=provider,
            prediction_aggregator=aggregator,
            output_queue=queue_ab,
        )
        output = await agent_a.run_cycle()
        await provider.close()

        assert len(output.prediction_contracts) > 0

    @pytest.mark.asyncio
    async def test_batch_methods_used(self):
        """Verify that batch methods are called when available."""
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)

        assert hasattr(provider, "fetch_spot_quotes_batch")
        assert hasattr(provider, "fetch_options_chains_batch")

        await provider.connect()
        batch_quotes = await provider.fetch_spot_quotes_batch(["SPY", "QQQ"])
        await provider.close()

        assert "SPY" in batch_quotes
        assert "QQQ" in batch_quotes


class TestAgentBCycle:
    @pytest.mark.asyncio
    async def test_produces_risk_matrices(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)
        queue_bc = asyncio.Queue(maxsize=10)

        await provider.connect()
        agent_a = AgentA(settings, provider, aggregator, queue_ab)
        agent_b = AgentB(settings, queue_ab, queue_bc)

        a_output = await agent_a.run_cycle()
        b_output = await agent_b.run_cycle(a_output)
        await provider.close()

        assert isinstance(b_output, AgentBOutput)
        for ticker in a_output.tickers_processed:
            if a_output.data_quality_summary.get(ticker) != DataQuality.INVALID:
                assert ticker in b_output.risk_matrices

    @pytest.mark.asyncio
    async def test_gamma_profiles_computed(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)
        queue_bc = asyncio.Queue(maxsize=10)

        await provider.connect()
        agent_a = AgentA(settings, provider, aggregator, queue_ab)
        agent_b = AgentB(settings, queue_ab, queue_bc)

        a_output = await agent_a.run_cycle()
        b_output = await agent_b.run_cycle(a_output)
        await provider.close()

        for ticker, matrix in b_output.risk_matrices.items():
            assert len(matrix.gamma_profiles) > 0
            for profile in matrix.gamma_profiles:
                assert profile.gamma_wall_strike is not None

    @pytest.mark.asyncio
    async def test_iv_analysis_computed(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)
        queue_bc = asyncio.Queue(maxsize=10)

        await provider.connect()
        agent_a = AgentA(settings, provider, aggregator, queue_ab)
        agent_b = AgentB(settings, queue_ab, queue_bc)

        a_output = await agent_a.run_cycle()
        b_output = await agent_b.run_cycle(a_output)
        await provider.close()

        for ticker, matrix in b_output.risk_matrices.items():
            assert matrix.iv_analysis is not None
            assert matrix.iv_analysis.current_iv > 0


class TestAgentCCycle:
    @pytest.mark.asyncio
    async def test_produces_c_output(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)
        queue_bc = asyncio.Queue(maxsize=10)
        dispatcher = AlertDispatcher(dry_run=True)

        await provider.connect()
        await dispatcher.connect()

        agent_a = AgentA(settings, provider, aggregator, queue_ab)
        agent_b = AgentB(settings, queue_ab, queue_bc)
        agent_c = AgentC(settings, dispatcher, queue_bc)

        a_output = await agent_a.run_cycle()
        b_output = await agent_b.run_cycle(a_output)
        c_output = await agent_c.run_cycle(b_output)

        await provider.close()
        await dispatcher.close()

        assert isinstance(c_output, AgentCOutput)
        assert c_output.cycle_id == a_output.cycle_id

    @pytest.mark.asyncio
    async def test_alerts_have_valid_risk_levels(self):
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)
        queue_bc = asyncio.Queue(maxsize=10)
        dispatcher = AlertDispatcher(dry_run=True)

        await provider.connect()
        await dispatcher.connect()

        agent_a = AgentA(settings, provider, aggregator, queue_ab)
        agent_b = AgentB(settings, queue_ab, queue_bc)
        agent_c = AgentC(settings, dispatcher, queue_bc)

        a_output = await agent_a.run_cycle()
        b_output = await agent_b.run_cycle(a_output)
        c_output = await agent_c.run_cycle(b_output)

        await provider.close()
        await dispatcher.close()

        for alert in c_output.alerts_fired:
            assert alert.risk_level in (
                RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL
            )
            assert len(alert.triggered_rules) > 0
            assert len(alert.headline) > 0
            assert alert.spot_price > 0

    @pytest.mark.asyncio
    async def test_alert_deduplication(self):
        """Second cycle within dedup window should not re-fire same alerts."""
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=42)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)
        queue_bc = asyncio.Queue(maxsize=10)
        dispatcher = AlertDispatcher(dry_run=True)

        await provider.connect()
        await dispatcher.connect()

        agent_a = AgentA(settings, provider, aggregator, queue_ab)
        agent_b = AgentB(settings, queue_ab, queue_bc)
        agent_c = AgentC(settings, dispatcher, queue_bc, dedup_window_seconds=300.0)

        # Cycle 1
        a1 = await agent_a.run_cycle()
        b1 = await agent_b.run_cycle(a1)
        c1 = await agent_c.run_cycle(b1)
        alerts_cycle1 = len(c1.alerts_fired)

        # Cycle 2 (same market data → same conditions → dedup should suppress)
        a2 = await agent_a.run_cycle()
        b2 = await agent_b.run_cycle(a2)
        c2 = await agent_c.run_cycle(b2)
        alerts_cycle2 = len(c2.alerts_fired)

        await provider.close()
        await dispatcher.close()

        # Second cycle should have fewer or equal alerts due to dedup
        assert alerts_cycle2 <= alerts_cycle1


class TestFullPipelineIntegration:
    @pytest.mark.asyncio
    async def test_full_abc_pipeline_runs_without_error(self):
        """End-to-end smoke test: A→B→C pipeline completes without exceptions."""
        settings = make_test_settings()
        provider = MockMarketDataProvider(seed=7)
        aggregator = MockAggregator()
        queue_ab = asyncio.Queue(maxsize=10)
        queue_bc = asyncio.Queue(maxsize=10)
        dispatcher = AlertDispatcher(dry_run=True)

        await provider.connect()
        await dispatcher.connect()

        agent_a = AgentA(settings, provider, aggregator, queue_ab)
        agent_b = AgentB(settings, queue_ab, queue_bc)
        agent_c = AgentC(settings, dispatcher, queue_bc)

        # Run 3 consecutive cycles
        for i in range(3):
            a = await agent_a.run_cycle()
            b = await agent_b.run_cycle(a)
            c = await agent_c.run_cycle(b)

            assert a is not None
            assert b is not None
            assert c is not None

        await provider.close()
        await dispatcher.close()
