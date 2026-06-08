"""
api_server.py — FastAPI HTTP/WS Server for options_agent
========================================================
Runs alongside the Orchestrator in the same asyncio event loop.

Endpoints:
  GET  /health              — liveness probe
  GET  /snapshot            — full dashboard snapshot (latest cycle state)
  GET  /heartbeat           — current SystemHeartbeat
  GET  /agents              — agent status list
  GET  /spot                — all spot quotes
  GET  /spot/{ticker}       — single ticker spot quote
  GET  /chains              — all options chains
  GET  /chains/{ticker}     — options chains for one ticker
  GET  /risk                — all risk matrices
  GET  /risk/{ticker}       — risk matrix for one ticker
  GET  /alerts              — recent alerts (last 50)
  WS   /ws                  — real-time push stream

Usage:
  from options_agent.api_server import APIRegistry, create_app
  registry = APIRegistry()
  app = create_app(registry)
  # Run with: uvicorn options_agent.api_server:app --factory ...
  # Or use run_server() below for inline startup.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from options_agent.core.api_models import (
    AgentInfoAPI,
    AgentStatus,
    CatalystSignalAPI,
    DataQuality,
    DashboardSnapshotAPI,
    EVDiscrepancyAPI,
    GammaProfileAPI,
    IVAnalysisAPI,
    OptionContractAPI,
    OptionsChainAPI,
    PredictionMarketContractAPI,
    RiskAlertAPI,
    RiskMatrixAPI,
    ScoutStatusAPI,
    SpotQuoteAPI,
    SystemHeartbeatAPI,
    UnusualFlowSignalAPI,
    WSSpotUpdate,
    WSHeartbeat,
    WSAlertFired,
)
from options_agent.core.schemas import (
    AgentAOutput,
    AgentBOutput,
    AgentCOutput,
    DataQuality as _DQ,
    EVDiscrepancy,
    FlowSentiment,
    GammaProfile,
    GammaRegime,
    IVAnalysis,
    OptionContract,
    OptionType,
    OptionsChain,
    PredictionMarketContract,
    RiskAlert,
    RiskLevel,
    ScoutAgentStatus,
    SpotQuote,
    SystemHeartbeat,
    UnusualFlowSignal,
)
from options_agent.harness.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------

class WSConnectionManager:
    """Manages multiple WebSocket client connections."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(f"[WS] Client connected. Total: {len(self._connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info(f"[WS] Client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, payload: dict) -> None:
        """Send JSON payload to all connected clients."""
        if not self._connections:
            return
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ---------------------------------------------------------------------------
# API Registry — shared state between Orchestrator and FastAPI
# ---------------------------------------------------------------------------

class APIRegistry:
    """
    Thread-safe (actually asyncio-safe) shared state holder.

    The Orchestrator calls register_*() methods after each cycle.
    FastAPI route handlers read from these fields.

    All fields are set exclusively by the orchestrator thread
    (same event loop as FastAPI), so no explicit locking needed.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # Latest outputs from each agent
        self._a_output: Optional[AgentAOutput] = None
        self._b_output: Optional[AgentBOutput] = None
        self._c_output: Optional[AgentCOutput] = None
        self._heartbeat: Optional[SystemHeartbeat] = None

        # Agent metadata (tracked separately since outputs don't persist between cycles)
        self._agent_meta: dict[str, dict] = {
            "agent_a": {"status": AgentStatus.IDLE, "last_cycle_id": None,
                        "last_cycle_ms": None, "last_active_utc": None,
                        "consecutive_errors": 0, "cycle_count": 0},
            "agent_b": {"status": AgentStatus.IDLE, "last_cycle_id": None,
                        "last_cycle_ms": None, "last_active_utc": None,
                        "consecutive_errors": 0, "cycle_count": 0},
            "agent_c": {"status": AgentStatus.IDLE, "last_cycle_id": None,
                        "last_cycle_ms": None, "last_active_utc": None,
                        "consecutive_errors": 0, "cycle_count": 0},
        }

        # Recent alerts (keep last 50 across cycles)
        self._recent_alerts: list[RiskAlert] = []
        self._start_time: Optional[datetime] = None

        # Scout Agent state
        self._flow_signals: list[UnusualFlowSignal] = []   # last 100 qualifying signals
        self._scout_status: ScoutAgentStatus = ScoutAgentStatus()

        # WebSocket manager
        self._ws_manager = WSConnectionManager()

        # Registered flag
        self._registered = False

    def register_orchestrator(self, orchestrator) -> None:
        """Called by Orchestrator during boot to pass self reference."""
        self._orchestrator = orchestrator
        self._start_time = datetime.utcnow()
        self._registered = True
        logger.info("[API] Orchestrator registered with API registry")

    def register_cycle(
        self,
        a_output: Optional[AgentAOutput],
        b_output: Optional[AgentBOutput],
        c_output: Optional[AgentCOutput],
        heartbeat: SystemHeartbeat,
    ) -> None:
        """Called by Orchestrator after each cycle."""
        self._a_output = a_output
        self._b_output = b_output
        self._c_output = c_output
        self._heartbeat = heartbeat

        now = datetime.utcnow()
        meta = self._agent_meta

        # Agent A
        meta["agent_a"].update({
            "status": AgentStatus.OK if heartbeat.agent_a_ok else AgentStatus.FAIL,
            "last_cycle_id": a_output.cycle_id if a_output else None,
            "last_cycle_ms": a_output.latency_ms if a_output else None,
            "last_active_utc": now,
            "consecutive_errors": 0 if heartbeat.agent_a_ok else meta["agent_a"]["consecutive_errors"] + 1,
            "cycle_count": meta["agent_a"]["cycle_count"] + 1,
        })

        # Agent B
        meta["agent_b"].update({
            "status": AgentStatus.OK if heartbeat.agent_b_ok else AgentStatus.FAIL,
            "last_cycle_id": b_output.cycle_id if b_output else None,
            "last_cycle_ms": b_output.compute_time_ms if b_output else None,
            "last_active_utc": now,
            "consecutive_errors": 0 if heartbeat.agent_b_ok else meta["agent_b"]["consecutive_errors"] + 1,
            "cycle_count": meta["agent_b"]["cycle_count"] + 1,
        })

        # Agent C
        meta["agent_c"].update({
            "status": AgentStatus.OK if heartbeat.agent_c_ok else AgentStatus.FAIL,
            "last_cycle_id": c_output.cycle_id if c_output else None,
            "last_cycle_ms": None,
            "last_active_utc": now,
            "consecutive_errors": 0 if heartbeat.agent_c_ok else meta["agent_c"]["consecutive_errors"] + 1,
            "cycle_count": meta["agent_c"]["cycle_count"] + 1,
        })

        # Append new alerts
        if c_output and c_output.alerts_fired:
            for alert in c_output.alerts_fired:
                self._recent_alerts.append(alert)
            self._recent_alerts = self._recent_alerts[-50:]

    @property
    def ws_manager(self) -> WSConnectionManager:
        return self._ws_manager

    # -------------------------------------------------------------------------
    # Converter helpers — internal schema → API schema
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_spot_quote_api(q: SpotQuote) -> SpotQuoteAPI:
        return SpotQuoteAPI(
            ticker=q.ticker, spot_price=q.spot_price, bid=q.bid, ask=q.ask,
            last=q.last, volume=q.volume, timestamp_utc=q.timestamp_utc, source=q.source,
        )

    @staticmethod
    def _to_option_contract_api(c: OptionContract) -> OptionContractAPI:
        return OptionContractAPI(
            ticker=c.ticker, expiration=c.expiration, strike=c.strike,
            option_type=OptionType(c.option_type.value),
            bid=c.bid, ask=c.ask, last=c.last, volume=c.volume,
            open_interest=c.open_interest,
            implied_volatility=c.implied_volatility,
            delta=c.delta, gamma=c.gamma, theta=c.theta, vega=c.vega,
            timestamp_utc=c.timestamp_utc,
        )

    @staticmethod
    def _to_options_chain_api(chain: OptionsChain) -> OptionsChainAPI:
        return OptionsChainAPI(
            ticker=chain.ticker, expiration=chain.expiration,
            spot_price=chain.spot_price,
            contracts=[APIRegistry._to_option_contract_api(c) for c in chain.contracts],
            timestamp_utc=chain.timestamp_utc,
            data_quality=DataQuality(chain.data_quality.value),
            quality_notes=chain.quality_notes,
        )

    @staticmethod
    def _to_pred_contract_api(p: PredictionMarketContract) -> PredictionMarketContractAPI:
        return PredictionMarketContractAPI(
            market_name=p.market_name, contract_id=p.contract_id,
            question=p.question, ticker_ref=p.ticker_ref,
            yes_price=p.yes_price, no_price=p.no_price,
            volume_24h=p.volume_24h, timestamp_utc=p.timestamp_utc,
        )

    @staticmethod
    def _to_catalyst_api(c: "CatalystSignal") -> CatalystSignalAPI:
        from options_agent.core.schemas import CatalystSignal
        return CatalystSignalAPI(
            ticker=c.ticker, event_type=c.event_type,
            event_time_utc=c.event_time_utc, description=c.description,
            hours_until_event=c.hours_until_event, is_imminent=c.is_imminent,
        )

    @staticmethod
    def _to_gamma_profile_api(g: GammaProfile) -> GammaProfileAPI:
        return GammaProfileAPI(
            ticker=g.ticker, expiration=g.expiration, spot_price=g.spot_price,
            gamma_by_strike={str(k): v for k, v in g.gamma_by_strike.items()},
            gamma_wall_strike=g.gamma_wall_strike,
            zero_gamma_strike=g.zero_gamma_strike,
            max_pain_strike=g.max_pain_strike,
            gamma_regime=GammaRegime(g.gamma_regime.value),
            net_gamma_at_spot=g.net_gamma_at_spot,
            computed_at=g.computed_at,
        )

    @staticmethod
    def _to_iv_analysis_api(iv: IVAnalysis) -> IVAnalysisAPI:
        return IVAnalysisAPI(
            ticker=iv.ticker, current_iv=iv.current_iv, iv_rank=iv.iv_rank,
            iv_percentile=iv.iv_percentile, iv_52w_high=iv.iv_52w_high,
            iv_52w_low=iv.iv_52w_low, is_iv_elevated=iv.is_iv_elevated,
            iv_crush_risk=iv.iv_crush_risk, iv_crush_risk_reason=iv.iv_crush_risk_reason,
            computed_at=iv.computed_at,
        )

    @staticmethod
    def _to_ev_disc_api(e: EVDiscrepancy) -> EVDiscrepancyAPI:
        return EVDiscrepancyAPI(
            ticker=e.ticker, option_implied_prob=e.option_implied_prob,
            prediction_market_prob=e.prediction_market_prob,
            ev_spread=e.ev_spread, ev_spread_pct=e.ev_spread_pct,
            is_significant=e.is_significant,
            option_contract_ref=e.option_contract_ref,
            prediction_market_ref=e.prediction_market_ref,
            kelly_fraction=e.kelly_fraction, computed_at=e.computed_at,
        )

    @staticmethod
    def _to_risk_matrix_api(m: "RiskMatrix") -> RiskMatrixAPI:
        from options_agent.core.schemas import RiskMatrix
        return RiskMatrixAPI(
            ticker=m.ticker, spot_price=m.spot_price, timestamp_utc=m.timestamp_utc,
            gamma_profiles=[APIRegistry._to_gamma_profile_api(g) for g in m.gamma_profiles],
            iv_analysis=APIRegistry._to_iv_analysis_api(m.iv_analysis) if m.iv_analysis else None,
            ev_discrepancies=[APIRegistry._to_ev_disc_api(e) for e in m.ev_discrepancies],
            catalyst_signals=[APIRegistry._to_catalyst_api(c) for c in m.catalyst_signals],
            gamma_wall_breach=m.gamma_wall_breach,
            zero_gamma_breach=m.zero_gamma_breach,
            iv_crush_imminent=m.iv_crush_imminent,
            ev_arb_detected=m.ev_arb_detected,
        )

    @staticmethod
    def _to_risk_alert_api(a: RiskAlert) -> RiskAlertAPI:
        return RiskAlertAPI(
            alert_id=a.alert_id, ticker=a.ticker,
            risk_level=RiskLevel(a.risk_level.value),
            timestamp_utc=a.timestamp_utc,
            headline=a.headline, narrative=a.narrative,
            recommended_action=a.recommended_action,
            gamma_wall_strike=a.gamma_wall_strike,
            zero_gamma_strike=a.zero_gamma_strike,
            max_pain_strike=a.max_pain_strike,
            spot_price=a.spot_price,
            iv_rank=a.iv_rank, ev_spread_pct=a.ev_spread_pct,
            hours_to_catalyst=a.hours_to_catalyst,
            triggered_rules=a.triggered_rules,
        )

    @staticmethod
    def _to_hb_api(h: SystemHeartbeat) -> SystemHeartbeatAPI:
        return SystemHeartbeatAPI(
            cycle_id=h.cycle_id, cycle_start_utc=h.cycle_start_utc,
            cycle_end_utc=h.cycle_end_utc, duration_ms=h.duration_ms,
            agent_a_ok=h.agent_a_ok, agent_b_ok=h.agent_b_ok, agent_c_ok=h.agent_c_ok,
            alerts_this_cycle=h.alerts_this_cycle, errors_this_cycle=h.errors_this_cycle,
            consecutive_error_count=h.consecutive_error_count,
        )

    # -------------------------------------------------------------------------
    # API endpoint implementations
    # -------------------------------------------------------------------------

    def get_health(self) -> dict:
        return {
            "status": "ok",
            "registered": self._registered,
            "cycle_count": self._agent_meta["agent_a"]["cycle_count"],
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

    def get_snapshot(self) -> DashboardSnapshotAPI:
        hb = self._heartbeat
        if not hb:
            raise HTTPException(status_code=503, detail="No cycle data yet — orchestrator not running")

        a = self._a_output
        b = self._b_output

        uptime = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0

        return DashboardSnapshotAPI(
            heartbeat=self._to_hb_api(hb),
            agents=self.get_agents(),
            tickers=list(a.tickers_processed) if a else [],
            spot_quotes={
                t: self._to_spot_quote_api(q) for t, q in (a.spot_quotes.items() if a else {})
            },
            options_chains={
                t: [self._to_options_chain_api(c) for c in chains]
                for t, chains in (a.options_chains.items() if a else {})
            },
            risk_matrices={
                t: self._to_risk_matrix_api(m) for t, m in (b.risk_matrices.items() if b else {})
            },
            recent_alerts=[self._to_risk_alert_api(a_) for a_ in self._recent_alerts[-10:]],
            uptime_seconds=uptime,
        )

    def get_heartbeat(self) -> SystemHeartbeatAPI:
        if not self._heartbeat:
            raise HTTPException(status_code=503, detail="No heartbeat yet")
        return self._to_hb_api(self._heartbeat)

    def get_agents(self) -> list[AgentInfoAPI]:
        return [
            AgentInfoAPI(
                name="Agent A — Data Provider",
                status=self._agent_meta["agent_a"]["status"],
                last_cycle_id=self._agent_meta["agent_a"]["last_cycle_id"],
                last_cycle_ms=self._agent_meta["agent_a"]["last_cycle_ms"],
                last_active_utc=self._agent_meta["agent_a"]["last_active_utc"],
                consecutive_errors=self._agent_meta["agent_a"]["consecutive_errors"],
                cycle_count=self._agent_meta["agent_a"]["cycle_count"],
            ),
            AgentInfoAPI(
                name="Agent B — Risk Engine",
                status=self._agent_meta["agent_b"]["status"],
                last_cycle_id=self._agent_meta["agent_b"]["last_cycle_id"],
                last_cycle_ms=self._agent_meta["agent_b"]["last_cycle_ms"],
                last_active_utc=self._agent_meta["agent_b"]["last_active_utc"],
                consecutive_errors=self._agent_meta["agent_b"]["consecutive_errors"],
                cycle_count=self._agent_meta["agent_b"]["cycle_count"],
            ),
            AgentInfoAPI(
                name="Agent C — Alert Dispatcher",
                status=self._agent_meta["agent_c"]["status"],
                last_cycle_id=self._agent_meta["agent_c"]["last_cycle_id"],
                last_cycle_ms=self._agent_meta["agent_c"]["last_cycle_ms"],
                last_active_utc=self._agent_meta["agent_c"]["last_active_utc"],
                consecutive_errors=self._agent_meta["agent_c"]["consecutive_errors"],
                cycle_count=self._agent_meta["agent_c"]["cycle_count"],
            ),
        ]

    def get_all_spots(self) -> dict[str, SpotQuoteAPI]:
        a = self._a_output
        if not a:
            return {}
        return {t: self._to_spot_quote_api(q) for t, q in a.spot_quotes.items()}

    def get_spot(self, ticker: str) -> SpotQuoteAPI:
        a = self._a_output
        if not a or ticker not in a.spot_quotes:
            raise HTTPException(status_code=404, detail=f"No spot data for ticker: {ticker}")
        return self._to_spot_quote_api(a.spot_quotes[ticker])

    def get_all_chains(self) -> dict[str, list[OptionsChainAPI]]:
        a = self._a_output
        if not a:
            return {}
        return {
            t: [self._to_options_chain_api(c) for c in chains]
            for t, chains in a.options_chains.items()
        }

    def get_chains(self, ticker: str) -> list[OptionsChainAPI]:
        a = self._a_output
        if not a or ticker not in a.options_chains:
            raise HTTPException(status_code=404, detail=f"No options chains for ticker: {ticker}")
        return [self._to_chain(c) for c in a.options_chains[ticker]]

    def _to_chain(self, c: OptionsChain) -> OptionsChainAPI:
        return self._to_options_chain_api(c)

    def get_all_risk(self) -> dict[str, RiskMatrixAPI]:
        b = self._b_output
        if not b:
            return {}
        return {t: self._to_risk_matrix_api(m) for t, m in b.risk_matrices.items()}

    def get_risk(self, ticker: str) -> RiskMatrixAPI:
        b = self._b_output
        if not b or ticker not in b.risk_matrices:
            raise HTTPException(status_code=404, detail=f"No risk matrix for ticker: {ticker}")
        return self._to_risk_matrix_api(b.risk_matrices[ticker])

    def get_alerts(self) -> list[RiskAlertAPI]:
        return [self._to_risk_alert_api(a_) for a_ in self._recent_alerts[-50:]]

    # ── Scout Agent ───────────────────────────────────────────────────────

    def register_flow_signal(self, signal: UnusualFlowSignal) -> None:
        """Called by ScoutAgent on each qualifying flow signal."""
        self._flow_signals.append(signal)
        if len(self._flow_signals) > 100:
            self._flow_signals = self._flow_signals[-100:]

    def register_scout_status(self, status: ScoutAgentStatus) -> None:
        self._scout_status = status

    def get_flow_signals(self, ticker: Optional[str] = None, limit: int = 50) -> list[UnusualFlowSignalAPI]:
        signals = self._flow_signals
        if ticker:
            signals = [s for s in signals if s.ticker == ticker.upper()]
        return [self._to_flow_signal_api(s) for s in signals[-limit:]]

    def get_scout_status(self) -> ScoutStatusAPI:
        s = self._scout_status
        return ScoutStatusAPI(
            is_connected=s.is_connected,
            last_signal_at=s.last_signal_at,
            signals_today=s.signals_today,
            total_signals=s.total_signals,
            reconnect_count=s.reconnect_count,
            last_error=s.last_error,
        )

    @staticmethod
    def _to_flow_signal_api(s: UnusualFlowSignal) -> UnusualFlowSignalAPI:
        from options_agent.core.api_models import FlowSentiment as _FS
        return UnusualFlowSignalAPI(
            signal_id=s.signal_id,
            ticker=s.ticker,
            expiration=s.expiration,
            strike=s.strike,
            option_type=OptionType(s.option_type.value),
            premium_usd=s.premium_usd,
            volume=s.volume,
            open_interest=s.open_interest,
            spot_price_at_trade=s.spot_price_at_trade,
            implied_volatility=s.implied_volatility,
            delta=s.delta,
            sentiment=_FS(s.sentiment.value),
            is_sweep=s.is_sweep,
            is_block=s.is_block,
            exchange=s.exchange,
            timestamp_utc=s.timestamp_utc,
            source=s.source,
        )


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(registry: APIRegistry) -> FastAPI:
    """Build the FastAPI application with all routes and CORS."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("[API] Server starting up...")
        yield
        logger.info("[API] Server shutting down...")

    app = FastAPI(
        title="Options Agent API",
        description="REST + WebSocket API for the Options Monitoring & Macro Arbitrage Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow frontend origin in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten this to your frontend domain in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    r = registry  # shorthand

    # ── Health ────────────────────────────────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health():
        return r.get_health()

    # ── Snapshot ─────────────────────────────────────────────────────────
    @app.get("/snapshot", response_model=DashboardSnapshotAPI, tags=["dashboard"])
    async def snapshot():
        return r.get_snapshot()

    # ── Heartbeat ────────────────────────────────────────────────────────
    @app.get("/heartbeat", response_model=SystemHeartbeatAPI, tags=["system"])
    async def heartbeat():
        return r.get_heartbeat()

    # ── Agents ───────────────────────────────────────────────────────────
    @app.get("/agents", response_model=list[AgentInfoAPI], tags=["agents"])
    async def agents():
        return r.get_agents()

    # ── Spot quotes ──────────────────────────────────────────────────────
    @app.get("/spot", response_model=dict[str, SpotQuoteAPI], tags=["market"])
    async def all_spots():
        return r.get_all_spots()

    @app.get("/spot/{ticker}", response_model=SpotQuoteAPI, tags=["market"])
    async def spot_quote(ticker: str):
        return r.get_spot(ticker.upper())

    # ── Options chains ───────────────────────────────────────────────────
    @app.get("/chains", response_model=dict[str, list[OptionsChainAPI]], tags=["market"])
    async def all_chains():
        return r.get_all_chains()

    @app.get("/chains/{ticker}", response_model=list[OptionsChainAPI], tags=["market"])
    async def ticker_chains(ticker: str):
        return r.get_chains(ticker.upper())

    # ── Risk matrices ─────────────────────────────────────────────────────
    @app.get("/risk", response_model=dict[str, RiskMatrixAPI], tags=["risk"])
    async def all_risk():
        return r.get_all_risk()

    @app.get("/risk/{ticker}", response_model=RiskMatrixAPI, tags=["risk"])
    async def ticker_risk(ticker: str):
        return r.get_risk(ticker.upper())

    # ── Alerts ───────────────────────────────────────────────────────────
    @app.get("/alerts", response_model=list[RiskAlertAPI], tags=["alerts"])
    async def alerts():
        return r.get_alerts()

    # ── Scout / Unusual Flow ─────────────────────────────────────────────
    @app.get("/scout", response_model=ScoutStatusAPI, tags=["scout"])
    async def scout_status():
        return r.get_scout_status()

    @app.get("/flow", response_model=list[UnusualFlowSignalAPI], tags=["scout"])
    async def flow_all(limit: int = 50):
        return r.get_flow_signals(limit=min(limit, 100))

    @app.get("/flow/{ticker}", response_model=list[UnusualFlowSignalAPI], tags=["scout"])
    async def flow_ticker(ticker: str, limit: int = 50):
        return r.get_flow_signals(ticker=ticker.upper(), limit=min(limit, 100))

    # ── WebSocket ─────────────────────────────────────────────────────────
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await r.ws_manager.connect(websocket)
        try:
            while True:
                # Keep-alive: client can send anything; we just don't disconnect
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    # Echo back as pong (ping/pong for connection health)
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    # Send current heartbeat as keep-alive payload
                    if r._heartbeat:
                        hb = r._to_hb_api(r._heartbeat)
                        await websocket.send_json(hb.model_dump(mode="json"))
        except WebSocketDisconnect:
            r.ws_manager.disconnect(websocket)
        except Exception as e:
            logger.warning(f"[WS] Error: {e}")
            r.ws_manager.disconnect(websocket)

    return app


# ---------------------------------------------------------------------------
# Standalone runner (for development)
# ---------------------------------------------------------------------------

async def run_server(registry: APIRegistry, host: str = "0.0.0.0", port: int = 8000):
    """Start uvicorn with the given registry. Used for dev/testing."""
    import uvicorn
    app = create_app(registry)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
