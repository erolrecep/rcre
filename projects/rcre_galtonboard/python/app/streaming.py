"""WebSocket connection manager and streaming broadcaster for Galton Board flow."""

import asyncio
import json
from typing import Set, Dict, Any
from fastapi import WebSocket

from app.engine import GaltonEngine


class ConnectionManager:
    """Manages active WebSockets and broadcasts real-time simulation updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast payload to all connected clients."""
        if not self.active_connections:
            return
        payload = json.dumps(message)
        dead = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for d in dead:
            self.disconnect(d)


class StreamController:
    """Controls the background streaming loop for real-time drop animation."""

    def __init__(self, engine: GaltonEngine, manager: ConnectionManager):
        self.engine = engine
        self.manager = manager
        self.is_running = False
        self.rate_per_sec = 25
        self.task: asyncio.Task | None = None

    def start(self, rate: int = 25):
        self.rate_per_sec = max(1, min(rate, 500))
        if not self.is_running:
            self.is_running = True
            self.task = asyncio.create_task(self._loop())

    def stop(self):
        self.is_running = False
        if self.task and not self.task.done():
            self.task.cancel()
            self.task = None

    async def _loop(self):
        try:
            while self.is_running:
                # Interval between drops
                interval = 1.0 / self.rate_per_sec
                # Single ball drop with path
                landed_bin, path = self.engine.drop_single()

                # Broadcast single event
                stats = self.engine.calculate_statistics()
                msg = {
                    "type": "ball_drop",
                    "landed_bin": landed_bin,
                    "path": path,
                    "total_dropped": self.engine.total_dropped,
                    "mean": stats.mean,
                    "std_dev": stats.std_dev,
                    "chi_square": stats.chi_square_stat,
                }
                await self.manager.broadcast(msg)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
