"""
main.py — Async Stateful Orchestrator
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import time
import uuid
from datetime import datetime
from typing import Optional

from options_agent.config import Settings, get_settings, reload_settings
from options_agent.core.schemas import SystemHeartbeat
from options_agent.harness.logger import configure_logging, get_logger
from options_agent.harness.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)


def build_market_provider(settings: Settings):
    if settings.use_mock_data:
        from options_agent.data.mock_provider import MockMarketDataProvider
        logger.info("Using MockMarketDataProvider (offline mode)")
        return MockMarketDataProvider()
    name = settings.market_data_provider.name.lower()
    if name == "alpaca":
        from options_agent.data.alpaca_provider import AlpacaProvider
        cfg = settings.market_data_provider
        if not cfg.api_key:
            logger.warning("No ALPACA_API_KEY — falling back to mock")
            from options_agent.data.mock_provider import MockMarketDataProvider
            return MockMarketDataProvider()
        return AlpacaProvider(
            api_key=cfg.api_key, api_secret=cfg.api_secret,
            base_url=cfg.base_url, ws_url=cfg.ws_url,
            lookahead_days=settings.option_expiry_lookahead_days,
            moneyness_range_pct=settings.option_moneyness_range_pct,
        )
    elif name == "polygon":
        from options_agent.data.polygon_provider import PolygonProvider
        cfg = settings.polygon_config
        if not cfg.api_key:
            logger.warning("No POLYGON_API_KEY — falling back to mock")
            from options_agent.data.mock_provider import MockMarketDataProvider
            return MockMarketDataProvider()
        return PolygonProvider(
            api_key=cfg.api_key,
            lookahead_days=settings.option_expiry_lookahead_days,
            moneyness_range_pct=settings.option_moneyness_range_pct,
        )
    else:
        logger.warning(f"Unknown provider '{name}' — falling back to mock")
        from options_agent.data.mock_provider import MockMarketDataProvider
        return MockMarketDataProvider()


def build_prediction_aggregator(settings: Settings, session=None):
    if settings.use_mock_data:
        from options_agent.data.mock_provider import MockPredictionMarketProvider

        class MockAgg:
            def __init__(self): self._m = MockPredictionMarketProvider()
            async def connect(self): pass
            async def close(self): pass
            async def fetch_all(self, watch_tickers):
                return await self._m.fetch_markets(ticker_keywords=watch_tickers)

        return MockAgg()
    from options_agent.data.prediction_market_provider import (
        PredictionMarketAggregator, PolymarketProvider, KalshiProvider,
    )
    providers = []
    for pm in settings.prediction_markets:
        if pm.name == "polymarket":
            providers.append(PolymarketProvider(base_url=pm.base_url, session=session))
        elif pm.name == "kalshi":
            providers.append(KalshiProvider(base_url=pm.base_url, session=session))
    return PredictionMarketAggregator(providers=providers or None, session=session)


class Orchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ab_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.bc_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.iv_history_store: dict[str, list[float]] = {t: [] for t in settings.watch_tickers}
        self._shutdown = False
        self._cycle_count = 0
        self._consecutive_errors = 0
        self._start_time = datetime.utcnow()
        self._market_provider = None
        self._prediction_aggregator = None
        self._dispatcher = None
        self._agent_a = None
        self._agent_b = None
        self._agent_c = None

    async def boot(self) -> None:
        logger.info("=" * 60)
        logger.info("Options Monitoring & Macro Arbitrage Agent — BOOTING")
        logger.info(f"  Tickers:   {self.settings.watch_tickers}")
        logger.info(f"  Interval:  {self.settings.poll_interval_seconds}s")
        logger.info(f"  Dry-run:   {self.settings.dry_run}")
        logger.info(f"  Mock data: {self.settings.use_mock_data}")
        logger.info("=" * 60)

        self._market_provider = build_market_provider(self.settings)
        await self._market_provider.connect()

        self._prediction_aggregator = build_prediction_aggregator(self.settings)
        await self._prediction_aggregator.connect()

        from options_agent.webhooks.dispatcher import AlertDispatcher
        self._dispatcher = AlertDispatcher(
            dry_run=self.settings.dry_run,
            webhook_url=self.settings.alert_webhook.url if self.settings.alert_webhook.enabled else "",
            webhook_secret=self.settings.alert_webhook.secret_value,
            slack_webhook_url=self.settings.slack_webhook_url,
        )
        await self._dispatcher.connect()

        from options_agent.agents.agent_a import AgentA
        from options_agent.agents.agent_b import AgentB
        from options_agent.agents.agent_c import AgentC

        self._agent_a = AgentA(
            settings=self.settings,
            market_provider=self._market_provider,
            prediction_aggregator=self._prediction_aggregator,
            output_queue=self.ab_queue,
            iv_history_store=self.iv_history_store,
        )
        self._agent_b = AgentB(
            settings=self.settings,
            input_queue=self.ab_queue,
            output_queue=self.bc_queue,
            iv_history_store=self.iv_history_store,
        )
        self._agent_c = AgentC(
            settings=self.settings,
            dispatcher=self._dispatcher,
            input_queue=self.bc_queue,
        )
        logger.info("All agents initialized. Starting main loop.")

    async def run(self) -> None:
        while not self._shutdown:
            cycle_start = time.monotonic()
            self._cycle_count += 1
            cycle_id = f"ORCH_{self._cycle_count:06d}"
            agent_a_ok = agent_b_ok = agent_c_ok = False
            alerts_this_cycle = errors_this_cycle = 0

            try:
                a_output = await asyncio.wait_for(
                    self._agent_a.run_cycle(),
                    timeout=self.settings.poll_interval_seconds * 0.8,
                )
                agent_a_ok = a_output is not None

                if a_output:
                    b_output = await asyncio.wait_for(
                        self._agent_b.run_cycle(a_output), timeout=30.0,
                    )
                    agent_b_ok = b_output is not None

                    if b_output:
                        c_output = await asyncio.wait_for(
                            self._agent_c.run_cycle(b_output), timeout=15.0,
                        )
                        agent_c_ok = c_output is not None
                        if c_output:
                            alerts_this_cycle = len(c_output.alerts_fired)

                self._consecutive_errors = 0

            except asyncio.TimeoutError as e:
                errors_this_cycle += 1
                self._consecutive_errors += 1
                logger.error(f"[Orchestrator] Cycle {cycle_id} timed out: {e}")
            except Exception as e:
                errors_this_cycle += 1
                self._consecutive_errors += 1
                logger.error(f"[Orchestrator] Cycle {cycle_id} error: {e}", exc_info=True)

            cycle_ms = (time.monotonic() - cycle_start) * 1000
            hb = SystemHeartbeat(
                cycle_id=cycle_id,
                cycle_start_utc=datetime.utcnow(),
                cycle_end_utc=datetime.utcnow(),
                duration_ms=cycle_ms,
                agent_a_ok=agent_a_ok,
                agent_b_ok=agent_b_ok,
                agent_c_ok=agent_c_ok,
                alerts_this_cycle=alerts_this_cycle,
                errors_this_cycle=errors_this_cycle,
                consecutive_error_count=self._consecutive_errors,
            )
            status = "OK" if (hb.agent_a_ok and hb.agent_b_ok and hb.agent_c_ok) else "FAIL"
            logger.info(
                f"[HB] {cycle_id} {status} | {hb.duration_ms:.0f}ms | "
                f"alerts={hb.alerts_this_cycle} | errors={hb.errors_this_cycle}"
            )

            max_errors = self.settings.risk_thresholds.max_consecutive_errors
            if self._consecutive_errors >= max_errors:
                logger.critical(f"[Orchestrator] {self._consecutive_errors} consecutive errors — pausing 120s")
                await asyncio.sleep(120.0)
                self._consecutive_errors = 0
                continue

            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0, self.settings.poll_interval_seconds - elapsed)
            if sleep_for > 0 and not self._shutdown:
                try:
                    await asyncio.sleep(sleep_for)
                except asyncio.CancelledError:
                    break

    async def shutdown(self) -> None:
        logger.info("[Orchestrator] Shutting down gracefully...")
        self._shutdown = True
        if self._market_provider:
            await self._market_provider.close()
        if self._prediction_aggregator:
            await self._prediction_aggregator.close()
        if self._dispatcher:
            await self._dispatcher.close()
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        logger.info(
            f"[Orchestrator] Done. Uptime={uptime:.0f}s | "
            f"Cycles={self._cycle_count} | "
            f"Alerts={self._dispatcher.total_dispatched if self._dispatcher else 0}"
        )


async def async_main(args: argparse.Namespace) -> None:
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.use_mock:
        os.environ["USE_MOCK_DATA"] = "true"
    if args.tickers:
        os.environ["WATCH_TICKERS"] = args.tickers
    if args.interval:
        os.environ["POLL_INTERVAL_SECONDS"] = str(args.interval)

    settings = reload_settings()
    configure_logging(
        log_level=settings.log_level,
        log_file="options_agent.log" if not settings.use_mock_data else None,
    )

    orchestrator = Orchestrator(settings)
    loop = asyncio.get_running_loop()

    def _handle_signal(sig):
        logger.warning(f"Signal {sig.name} — shutting down")
        loop.create_task(orchestrator.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))
        except (NotImplementedError, AttributeError):
            pass

    try:
        await orchestrator.boot()
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
    finally:
        if not orchestrator._shutdown:
            await orchestrator.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Options Monitoring & Macro Arbitrage Agent")
    parser.add_argument("--dry-run",  action="store_true", default=False)
    parser.add_argument("--use-mock", action="store_true", default=False)
    parser.add_argument("--tickers",  type=str, default="")
    parser.add_argument("--interval", type=float, default=None)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
