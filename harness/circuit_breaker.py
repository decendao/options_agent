"""
harness/circuit_breaker.py — Resilience & Rate-Limit Harness
=============================================================
Provides:
  - RetryWithBackoff: exponential backoff decorator for API calls
  - RateLimiter: async token-bucket rate limiter per provider
  - CircuitBreaker: trips after N consecutive failures, auto-resets
  - DataSanityGuard: validates schema conformance and data freshness
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, TypeVar

import aiohttp

from options_agent.harness.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        retry_on: tuple = (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionResetError,
        ),
        retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.retry_on = retry_on
        self.retry_on_status = retry_on_status


def with_retry(config: Optional[RetryConfig] = None):
    """
    Async retry decorator with exponential backoff.
    Handles:
      - Network errors (aiohttp.ClientError, TimeoutError)
      - HTTP 429 (rate limit): respects Retry-After header
      - HTTP 5xx (server errors)

    Usage:
        @with_retry(RetryConfig(max_retries=3))
        async def fetch_data(...): ...
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except aiohttp.ClientResponseError as e:
                    if e.status in config.retry_on_status:
                        wait = _backoff_wait(attempt, config, e)
                        logger.warning(
                            f"HTTP {e.status} on {func.__name__} (attempt {attempt+1}/{config.max_retries+1}). "
                            f"Retrying in {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)
                        last_exception = e
                    else:
                        raise  # Non-retryable HTTP error

                except config.retry_on as e:
                    wait = _backoff_wait(attempt, config)
                    logger.warning(
                        f"{type(e).__name__} on {func.__name__} (attempt {attempt+1}/{config.max_retries+1}). "
                        f"Retrying in {wait:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait)
                    last_exception = e

            logger.error(f"{func.__name__} failed after {config.max_retries+1} attempts: {last_exception}")
            raise last_exception or RuntimeError(f"{func.__name__} exhausted retries")

        return wrapper  # type: ignore[return-value]
    return decorator


def _backoff_wait(
    attempt: int,
    config: RetryConfig,
    exc: Optional[aiohttp.ClientResponseError] = None,
) -> float:
    # Respect Retry-After header for 429s
    if exc and hasattr(exc, "headers") and exc.headers:
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), config.backoff_max)
            except ValueError:
                pass

    jitter = 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
    wait = min(config.backoff_base ** attempt + jitter, config.backoff_max)
    return wait


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

class AsyncRateLimiter:
    """
    Async token-bucket rate limiter.
    Enforces `rate_limit_per_minute` API calls per minute.
    Thread-safe via asyncio.Lock.
    """

    def __init__(self, rate_per_minute: int, provider_name: str = ""):
        self.rate_per_minute = rate_per_minute
        self.provider_name   = provider_name
        self._tokens         = float(rate_per_minute)
        self._max_tokens     = float(rate_per_minute)
        self._last_refill    = time.monotonic()
        self._lock           = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until the requested tokens are available."""
        async with self._lock:
            await self._refill()
            while self._tokens < tokens:
                wait = (tokens - self._tokens) / (self.rate_per_minute / 60.0)
                logger.debug(
                    f"RateLimiter[{self.provider_name}]: waiting {wait:.2f}s "
                    f"(tokens={self._tokens:.1f}/{self._max_tokens})"
                )
                await asyncio.sleep(wait)
                await self._refill()
            self._tokens -= tokens

    async def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill = elapsed * (self.rate_per_minute / 60.0)
        self._tokens = min(self._max_tokens, self._tokens + refill)
        self._last_refill = now

    def __repr__(self) -> str:
        return f"AsyncRateLimiter({self.provider_name}, {self.rate_per_minute}/min)"


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState:
    CLOSED  = "CLOSED"    # normal operation
    OPEN    = "OPEN"      # tripped — fast-fail all calls
    HALF_OPEN = "HALF_OPEN"  # testing if service recovered


class CircuitBreaker:
    """
    Three-state circuit breaker for external API calls.

    CLOSED  → normal. Tracks failure count.
    OPEN    → trips after max_failures. Rejects calls immediately.
    HALF_OPEN → allows one probe call after reset_timeout. If it succeeds,
                transitions back to CLOSED; if not, re-opens.
    """

    def __init__(
        self,
        name: str,
        max_failures: int = 5,
        reset_timeout_seconds: float = 60.0,
    ):
        self.name = name
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def call(self, coro):
        """
        Execute `coro` through the circuit breaker.
        Raises CircuitOpenError if the breaker is OPEN.
        """
        async with self._lock:
            await self._maybe_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN. "
                    f"Last failure: {self._last_failure_time}"
                )

        try:
            result = await coro
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _maybe_transition(self) -> None:
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info(f"Circuit '{self.name}' CLOSED (recovered)")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.max_failures:
                if self._state != CircuitState.OPEN:
                    logger.error(
                        f"Circuit '{self.name}' OPEN after {self._failure_count} failures"
                    )
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit (e.g. after config change)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None


class CircuitOpenError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data Sanity Guard
# ---------------------------------------------------------------------------

class DataSanityGuard:
    """
    Validates data freshness and shape before passing to Agent B.
    Flags stale data and prevents "garbage in, garbage out" scenarios.
    """

    def __init__(
        self,
        max_staleness_seconds: float = 120.0,
        min_contracts_per_chain: int = 5,
    ):
        self.max_staleness_seconds = max_staleness_seconds
        self.min_contracts_per_chain = min_contracts_per_chain

    def check_quote_freshness(self, timestamp_utc: datetime) -> tuple[bool, str]:
        """Returns (is_fresh, reason)."""
        age = (datetime.utcnow() - timestamp_utc).total_seconds()
        if age > self.max_staleness_seconds:
            return False, f"Quote is {age:.0f}s old (threshold: {self.max_staleness_seconds}s)"
        return True, ""

    def check_chain_completeness(
        self, chain_contracts: int, required: Optional[int] = None
    ) -> tuple[bool, str]:
        threshold = required or self.min_contracts_per_chain
        if chain_contracts < threshold:
            return False, f"Chain has only {chain_contracts} contracts (min: {threshold})"
        return True, ""

    def check_price_sanity(
        self, bid: float, ask: float, last: float
    ) -> tuple[bool, str]:
        if ask < bid:
            return False, f"Inverted market: bid={bid} > ask={ask}"
        if last <= 0 and bid <= 0 and ask <= 0:
            return False, "All prices are zero"
        spread_pct = (ask - bid) / max(ask, 1e-6)
        if spread_pct > 0.50:
            return False, f"Spread {spread_pct:.0%} is abnormally wide (>50%)"
        return True, ""
