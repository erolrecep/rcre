"""FastAPI application for RCRE Galton Board flow demonstration and streaming."""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models import SimulationConfig, SimulationResponse, HealthResponse, SimulationStats
from app.engine import GaltonEngine
from app.streaming import ConnectionManager, StreamController

# Global Engine and State
g_start_time = time.time()
g_engine = GaltonEngine()
g_manager = ConnectionManager()
g_controller = StreamController(g_engine, g_manager)

# Resolve path to HTML visualizer
_candidates = [
    Path(__file__).resolve().parent.parent / "web" / "index.html",
    Path(__file__).resolve().parent.parent.parent / "web" / "index.html",
]
WEB_INDEX_PATH = next((p for p in _candidates if p.exists()), _candidates[0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan."""
    yield
    g_controller.stop()


app = FastAPI(
    title="RCRE Galton Board Simulation API",
    version="1.0.0",
    description="High-performance C23 & Python 3.15 Galton Board (Quincunx) flow simulator and Central Limit Theorem analyzer.",
    lifespan=lifespan,
)

# Enable CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the interactive Web visualizer."""
    if WEB_INDEX_PATH.exists():
        return HTMLResponse(content=WEB_INDEX_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>RCRE Galton Board Visualizer (web/index.html not found)</h1>", status_code=404)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System and environment health verification."""
    return HealthResponse(
        status="healthy",
        service="rcre-galtonboard-fastapi",
        python_version=sys.version.split()[0],
        rcre_engine="ISO C23 & NumPy 2.0 accelerated",
        uptime_seconds=round(time.time() - g_start_time, 2),
    )


@app.post("/api/v1/simulate", response_model=SimulationResponse)
async def run_simulation(config: SimulationConfig):
    """Execute high-speed Monte-Carlo batch simulation and return full statistical parity metrics."""
    g_engine.reset(config)
    _, elapsed_ms = g_engine.simulate_batch(config.balls)
    stats = g_engine.calculate_statistics()
    items = g_engine.get_distribution_items()

    return SimulationResponse(
        configuration=config,
        stats=stats,
        distribution=items,
        elapsed_ms=round(elapsed_ms, 3),
    )


@app.get("/api/v1/stats", response_model=SimulationStats)
async def get_stats():
    """Retrieve current accumulated simulation moments and metrics."""
    return g_engine.calculate_statistics()


@app.post("/api/v1/reset")
async def reset_simulation(config: SimulationConfig | None = None):
    """Reset accumulated histogram counts."""
    g_controller.stop()
    g_engine.reset(config)
    return {"message": "Simulation reset successfully", "rows": g_engine.config.rows, "bias": g_engine.config.bias}


@app.get("/api/v1/export/csv", response_class=PlainTextResponse)
async def export_csv():
    """Export current bin accumulation as CSV text."""
    items = g_engine.get_distribution_items()
    lines = ["bin_index,count,empirical_probability,binomial_pmf,mixture_pmf,gaussian_pdf"]
    for it in items:
        lines.append(f"{it.bin},{it.count},{it.empirical_prob:.8f},{it.binomial_pmf:.8f},{it.mixture_pmf:.8f},{it.gaussian_pdf:.8f}")
    return PlainTextResponse(content="\n".join(lines), media_type="text/csv")


@app.get("/api/v1/export/json")
async def export_json():
    """Export current state and moments as formatted JSON."""
    stats = g_engine.calculate_statistics()
    items = g_engine.get_distribution_items()
    return JSONResponse(
        content={
            "configuration": g_engine.config.model_dump(),
            "stats": stats.model_dump(),
            "distribution": [it.model_dump() for it in items],
        }
    )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket streaming endpoint for real-time visual client synchronization."""
    await g_manager.connect(websocket)
    try:
        # Initial greeting with current stats
        stats = g_engine.calculate_statistics()
        await websocket.send_json({
            "type": "init",
            "rows": g_engine.config.rows,
            "bias": g_engine.config.bias,
            "total_dropped": g_engine.total_dropped,
            "stats": stats.model_dump(),
        })

        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "start":
                rate = data.get("rate", 30)
                g_controller.start(rate)
                await websocket.send_json({"type": "status", "running": True, "rate": g_controller.rate_per_sec})

            elif action == "stop":
                g_controller.stop()
                await websocket.send_json({"type": "status", "running": False})

            elif action == "batch":
                count = int(data.get("count", 1000))
                g_engine.simulate_batch(count)
                stats = g_engine.calculate_statistics()
                await websocket.send_json({
                    "type": "batch_completed",
                    "total_dropped": g_engine.total_dropped,
                    "stats": stats.model_dump(),
                })

            elif action == "reset":
                rows = data.get("rows", g_engine.config.rows)
                bias = data.get("bias", g_engine.config.bias)
                cfg = SimulationConfig(rows=rows, bias=bias)
                g_engine.reset(cfg)
                await websocket.send_json({"type": "reset", "rows": rows, "bias": bias})

    except WebSocketDisconnect:
        g_manager.disconnect(websocket)
    except Exception:
        g_manager.disconnect(websocket)
