from __future__ import annotations

import os
from typing import Any

import requests


DATA_BASE_URL = "https://api.massive.com"
SNAPSHOT_PATH = "/v2/snapshot/locale/us/markets/stocks/tickers"


class MassiveMarketDataClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DATA_BASE_URL,
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Massive API key is required. Set MASSIVE_API_KEY.")

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def from_env(cls) -> MassiveMarketDataClient:
        return cls(os.environ.get("MASSIVE_API_KEY", "").strip())

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        normalized_symbols = _normalize_symbols(symbols)
        if not normalized_symbols:
            return {}

        payload = self._request(
            SNAPSHOT_PATH,
            {
                "tickers": ",".join(normalized_symbols),
            },
        )
        tickers = payload.get("tickers")
        if not isinstance(tickers, list):
            raise ValueError("Massive snapshot response was missing a tickers list.")

        requested = set(normalized_symbols)
        prices: dict[str, float] = {}
        for item in tickers:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("ticker") or "").upper().strip()
            if not symbol or symbol not in requested:
                continue
            price = _snapshot_price(item)
            if price is not None:
                prices[symbol] = price
        return prices

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.get(url, headers=self._headers, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            detail = ""
            if response is not None:
                detail = f" Detail: {response.text[:300]}"
            raise ValueError(f"Massive market data request failed for {path}.{detail}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(f"Massive market data response for {path} was not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Massive market data response for {path} was not an object.")
        return payload


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        candidate = str(symbol or "").upper().strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)
    return normalized


def _snapshot_price(snapshot: dict[str, Any]) -> float | None:
    latest_trade = _mapping(snapshot.get("lastTrade"))
    price = _positive_float(latest_trade.get("p"))
    if price is not None:
        return price

    latest_quote = _mapping(snapshot.get("lastQuote"))
    bid = _positive_float(latest_quote.get("p"))
    ask = _positive_float(latest_quote.get("P"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2

    minute_bar = _mapping(snapshot.get("min"))
    price = _positive_float(minute_bar.get("c"))
    if price is not None:
        return price

    day_bar = _mapping(snapshot.get("day"))
    price = _positive_float(day_bar.get("c"))
    if price is not None:
        return price

    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number
