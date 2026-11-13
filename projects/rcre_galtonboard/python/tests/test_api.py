"""Integration tests for FastAPI endpoints and WebSocket streaming."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify healthcheck endpoint returns healthy status and Python 3.15."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rcre-galtonboard-fastapi"
    assert data["python_version"].startswith("3.15")


def test_index_page():
    """Verify web visualizer is returned at root URL."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Galton Board" in response.text
    assert "canvas" in response.text


def test_simulate_endpoint():
    """Verify POST /api/v1/simulate executes batch and returns full statistics."""
    payload = {
        "rows": 12,
        "balls": 25000,
        "bias": 0.5,
        "seed": 123
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["stats"]["total_balls"] == 25000
    assert abs(data["stats"]["mean"] - 6.0) < 0.05
    assert len(data["distribution"]) == 13  # 12 rows + 1 bins
    assert data["elapsed_ms"] >= 0


def test_simulate_endpoint_start_pins():
    """Verify simulation with apex starting pins P_0 = 4 produces mixture distribution."""
    payload = {
        "rows": 12,
        "start_pins": 4,
        "balls": 30000,
        "bias": 0.5,
        "seed": 42
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["stats"]["total_balls"] == 30000
    # Expected mean = (4 - 1)/2 + 12 * 0.5 = 7.5
    assert abs(data["stats"]["mean"] - 7.5) < 0.08
    # Number of bins = 12 + 4 = 16
    assert len(data["distribution"]) == 16
    assert data["stats"]["theoretical_mean"] == 7.5
    assert abs(data["stats"]["theoretical_variance"] - 4.25) < 1e-4


def test_export_endpoints():
    """Verify CSV and JSON export routes."""
    # Ensure there's data in the board
    client.post("/api/v1/simulate", json={"rows": 8, "balls": 500, "bias": 0.5})

    # Test CSV
    csv_resp = client.get("/api/v1/export/csv")
    assert csv_resp.status_code == 200
    assert "bin_index,count,empirical_probability" in csv_resp.text

    # Test JSON
    json_resp = client.get("/api/v1/export/json")
    assert json_resp.status_code == 200
    j_data = json_resp.json()
    assert "configuration" in j_data
    assert "stats" in j_data
    assert "distribution" in j_data


def test_websocket_streaming():
    """Verify WebSocket handshake and bidirectional message passing."""
    with client.websocket_connect("/ws/stream") as websocket:
        init_data = websocket.receive_json()
        assert init_data["type"] == "init"
        assert "rows" in init_data

        # Trigger batch drop via WebSocket
        websocket.send_json({"action": "batch", "count": 200})
        batch_resp = websocket.receive_json()
        assert batch_resp["type"] == "batch_completed"
        assert batch_resp["total_dropped"] >= 200
