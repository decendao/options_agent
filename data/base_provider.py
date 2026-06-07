"""
data/base_provider.py — Abstract Data Provider Interface
=========================================================
Defines the protocol all market data adapters must implement.
Concrete providers (Alpaca, Polygon, Mock) subclass BaseMarketDataProvider.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import date
from typing import AsyncIterator, Optional

import aiohttp

from options_agent.core.schemas import OptionsChain, SpotQuote


class BaseMarketDataProvider(ABC):
    """
    Abstract base class for all market data providers.

    Lifecycle:
      1. await provider.connect()
      2. Use fetch_* methods in loop
      3. await provider.close()

    All methods are async-safe and re-entrant.
    """

    def __init__(self, name: str, session: Optional[aiohttp.ClientSession] = None):
        self.name = name
        self._session: Optional[aiohttp.ClientSession] = session
        self._owns_session = session is None

    async def connect(self) -> None:
        """Initialize the HTTP session and any WebSocket connections."""
        if self._owns_session and self._session is None:
            timeout = aiohttp.ClientTimeout(total=30.0)
            connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def close(self) -> None:
        """Gracefully close HTTP session and WebSocket connections."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_spot_quote(self, ticker: str) -> SpotQuote:
        """Fetch the latest spot quote for a single ticker."""
        ...

    @abstractmethod
    async def fetch_options_chain(
        self,
        ticker: str,
        expiration: Optional[date] = None,
        moneyness_range_pct: float = 0.15,
    ) -> list[OptionsChain]:
        """
        Fetch the full options chain for a ticker.

        Args:
            ticker: Underlying symbol.
            expiration: If None, fetch all available expirations within lookahead.
            moneyness_range_pct: Strike filter ± X% of spot.

        Returns:
            List of OptionsChain objects (one per expiration).
        """
        ...

    @abstractmethod
    async def fetch_available_expirations(self, ticker: str) -> list[date]:
        """Return list of available option expiration dates for a ticker."""
        ...

    # ------------------------------------------------------------------
    # Optional: real-time streaming (default: unsupported)
    # ------------------------------------------------------------------

    async def stream_quotes(self, tickers: list[str]) -> AsyncIterator[SpotQuote]:
        """
        Stream real-time quote updates via WebSocket.
        Default implementation raises NotImplementedError.
        Concrete providers that support streaming override this.
        """
        raise NotImplementedError(f"{self.name} does not support streaming quotes")
        # Required: `yield` to satisfy AsyncIterator protocol
        # This line is unreachable but needed for type checking
        yield  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Shared HTTP helper
    # ------------------------------------------------------------------

    async def _get(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        """
        Perform a GET request. Raises aiohttp exceptions on failure.
        Rate-limit (429) handling is done at the harness layer.
        """
        if self._session is None:
            raise RuntimeError(f"Provider '{self.name}' not connected. Call connect() first.")

        async with self._session.get(url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def health_check(self) -> bool:
        """Return True if the provider is reachable and authenticated."""
        return True
