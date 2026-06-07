"""
api_server.py — FastAPI wrapper for the Options Agent orchestrator.

Exposes a health-check endpoint and runs the async orchestrator as a
background task so the service can be started with:

    uvicorn api_server:app --host 0.0.0.0 --port $PORT

or, via the installed package:

    uvicorn options_agent.api_server:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from options_agent.config import reload_settings
from options_agent.harness.logger import configure_logging, get_logger
from main import Orchestrator

logger = get_logger(__name__)

_orchestrator: Orchestrator | None = None
_start_time: datetime = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot the orchestrator on startup; shut it down on exit."""
    global _orchestrator

    settings = reload_settings()
    configure_logging(log_level=settings.log_level)

    _orchestrator = Orchestrator(settings)
    await _orchestrator.boot()

    # Run the orchestrator loop as a background task
    task = asyncio.create_task(_orchestrator.run())

    yield  # application is running

    # Graceful shutdown
    if _orchestrator and not _orchestrator._shutdown:
        await _orchestrator.shutdown()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Options Agent",
    description="Async Options Monitoring & Macro Arbitrage Agent",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe — returns 200 when the service is running."""
    uptime_seconds = (datetime.now(timezone.utc) - _start_time).total_seconds()
    return {
        "status": "ok",
        "uptime_seconds": round(uptime_seconds, 1),
        "orchestrator_running": _orchestrator is not None and not _orchestrator._shutdown,
    }


@app.get("/", tags=["ops"])
async def root():
    return {"service": "options-agent", "docs": "/docs"}
