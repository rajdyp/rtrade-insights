import asyncio
import json

import pytest
from fastapi import HTTPException

from stock_calculator.api import app, rank
from stock_calculator.config import AppConfig
from stock_calculator.ranking import rank_candidates


def test_health_route_is_registered():
    route_paths = {route.path for route in app.routes}

    assert "/health" in route_paths


def test_rank_endpoint_returns_table():
    response = _call_rank("EP\nTEST 100 95 5\n", "table")

    assert response.media_type == "text/plain"
    text = response.body.decode("utf-8")
    assert "Strategy" in text
    assert "TEST" in text
    assert text.endswith("\n")


def test_rank_endpoint_returns_csv():
    response = _call_rank("EP\nTEST 100 95 5\n", "csv")

    assert response.media_type == "text/csv"
    text = response.body.decode("utf-8")
    assert text.startswith("symbol,market_regime,mode,strategy")
    assert "TEST" in text


def test_rank_endpoint_returns_json():
    response = _call_rank("EP\nTEST 100 95 5\n", "json")

    assert response.media_type == "application/json"
    payload = json.loads(response.body.decode("utf-8"))
    assert "rows" not in payload
    assert payload["groups"]["EP"][0]["symbol"] == "TEST"
    assert payload["groups"]["4% BO"] == []
    assert payload["groups"]["BO"] == []


def test_rank_endpoint_rejects_invalid_format():
    with pytest.raises(HTTPException):
        _call_rank("EP\nTEST 100 95 5\n", "xml")


class FakeRequest:
    def __init__(self, text: str):
        self.text = text

    async def body(self) -> bytes:
        return self.text.encode("utf-8")


def _call_rank(text: str, output_format: str):
    original_rank_candidates = rank.__globals__["rank_candidates"]

    def fake_rank_candidates(body: str):
        return rank_candidates(
            body,
            config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
            load_strategy_metrics=False,
        )

    try:
        rank.__globals__["rank_candidates"] = fake_rank_candidates
        return asyncio.run(rank(FakeRequest(text), output_format))
    finally:
        rank.__globals__["rank_candidates"] = original_rank_candidates
