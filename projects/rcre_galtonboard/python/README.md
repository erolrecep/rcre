# RCRE Galton Board FastAPI Service

High-performance Python 3.15 FastAPI implementation of the Galton Board (Quincunx) flow simulator and Central Limit Theorem analyzer.

## Features
- **Python 3.15 Native**: Built on Python 3.15.2 with strict Pydantic v2 schemas and modern typing.
- **REST API**: Batch Monte-Carlo simulation with exact log-gamma Binomial PMF and Gaussian PDF comparisons.
- **Real-Time WebSockets**: Live streaming of ball drop events, paths, and accumulator moments (`/ws/stream`).
- **Interactive UI**: Embedded zero-dependency web visualizer served directly from `/`.
- **Fast Execution**: Accelerated NumPy 2.0 vectorized Binomial sampling (millions of drops/sec).

## Running the Service
```bash
uv run --python /opt/homebrew/bin/python3.15 uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Running Tests
```bash
uv run --python /opt/homebrew/bin/python3.15 pytest tests/ -v
```
