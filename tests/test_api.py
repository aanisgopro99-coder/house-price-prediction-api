"""Automated tests for the prediction API, using the ASGI transport."""

import asyncio

import httpx
from src.main import app

transport = httpx.ASGITransport(app=app)


async def _send_request(method: str, path: str, **kwargs):
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def request(method: str, path: str, **kwargs):
    return asyncio.run(_send_request(method, path, **kwargs))


def test_health_endpoint():
    response = request("GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_endpoint():
    response = request("GET", "/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == "LinearRegression"
    assert body["features"] == ["surface", "chambres"]
    assert "r2" in body["metrics"]


def test_prediction_endpoint():
    response = request(
        "POST",
        "/predict",
        json={"surface": 100, "chambres": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_price"], float)
    assert body["predicted_price"] > 0


def test_prediction_rejects_invalid_surface():
    response = request(
        "POST",
        "/predict",
        json={"surface": -10, "chambres": 3},
    )
    assert response.status_code == 422


def test_prediction_rejects_invalid_bedroom_count():
    response = request(
        "POST",
        "/predict",
        json={"surface": 100, "chambres": 0},
    )
    assert response.status_code == 422