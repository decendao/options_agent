# Options Agent

Async Stateful Options Monitoring & Macro Arbitrage Agent.

Monitors gamma walls, IV crush events, and prediction markets to generate real-time trading alerts.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py --use-mock
```

## Architecture

- **Agent A** — Data Provider & Context Assembler
- **Agent B** — Quantitative Reasoning & Greeks Engine  
- **Agent C** — Risk Circuit Breaker & Watchdog

## Data Providers

- Alpaca (market data)
- Polygon.io (IV history)
- Polymarket / Kalshi (prediction markets)
