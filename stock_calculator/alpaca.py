from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import requests


DATA_BASE_URL = "https://data.alpaca.markets/v2"
SUPPORTED_FEEDS = ("iex", "delayed_sip", "sip")


@dataclass(frozen=True)
class AlpacaSymbolData:
    symbol: str
    price: float | None
    today_low: float | None
    atr_percent: float | None
    price_timestamp: str
    low_timestamp: str
    atr_timestamp: str


class AlpacaMarketDataClient:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = DATA_BASE_URL,
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("Alpaca credentials are required. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

    @classmethod
    def from_env(cls) -> AlpacaMarketDataClient:
        return cls(
            os.environ.get("APCA_API_KEY_ID", "").strip(),
            os.environ.get("APCA_API_SECRET_KEY", "").strip(),
        )

    def get_market_data(
        self,
        symbols: list[str],
        *,
        feed: str = "iex",
        today: date | None = None,
    ) -> dict[str, AlpacaSymbolData]:
        normalized_symbols = _normalize_symbols(symbols)
        if not normalized_symbols:
            return {}
        if feed not in SUPPORTED_FEEDS:
            raise ValueError(f"Unsupported Alpaca feed: {feed}. Use iex, delayed_sip, or sip.")

        today = today or date.today()
        snapshots = self._get_snapshots(normalized_symbols, feed)
        bars = self._get_daily_bars(normalized_symbols, _historical_bars_feed(feed), today)

        result: dict[str, AlpacaSymbolData] = {}
        for symbol in normalized_symbols:
            snapshot = snapshots.get(symbol, {})
            price, price_timestamp = _snapshot_price(snapshot)
            today_low, low_timestamp = _snapshot_low(snapshot)
            atr_percent, atr_timestamp = _atr_percent(bars.get(symbol, []), today)
            result[symbol] = AlpacaSymbolData(
                symbol=symbol,
                price=price,
                today_low=today_low,
                atr_percent=atr_percent,
                price_timestamp=price_timestamp,
                low_timestamp=low_timestamp,
                atr_timestamp=atr_timestamp,
            )
        return result

    def _get_snapshots(self, symbols: list[str], feed: str) -> dict[str, dict[str, Any]]:
        payload = self._request(
            "/stocks/snapshots",
            {
                "symbols": ",".join(symbols),
                "feed": feed,
            },
        )
        snapshots = payload.get("snapshots", payload)
        if not isinstance(snapshots, dict):
            return {}
        return {str(symbol).upper(): value for symbol, value in snapshots.items() if isinstance(value, dict)}

    def _get_daily_bars(self, symbols: list[str], feed: str, today: date) -> dict[str, list[dict[str, Any]]]:
        payload = self._request(
            "/stocks/bars",
            {
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": (today - timedelta(days=70)).isoformat(),
                "end": today.isoformat(),
                "limit": 10000,
                "adjustment": "raw",
                "feed": feed,
            },
        )
        bars = payload.get("bars", {})
        if not isinstance(bars, dict):
            return {}
        return {
            str(symbol).upper(): [bar for bar in values if isinstance(bar, dict)]
            for symbol, values in bars.items()
            if isinstance(values, list)
        }

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
            raise ValueError(f"Alpaca market data request failed for {path}.{detail}") from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Alpaca market data response for {path} was not an object.")
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


def _historical_bars_feed(feed: str) -> str:
    return "sip" if feed == "delayed_sip" else feed


def _snapshot_price(snapshot: dict[str, Any]) -> tuple[float | None, str]:
    latest_trade = _mapping(snapshot.get("latestTrade"))
    price = _positive_float(latest_trade.get("p"))
    if price is not None:
        return price, str(latest_trade.get("t") or "")

    latest_quote = _mapping(snapshot.get("latestQuote"))
    bid = _positive_float(latest_quote.get("bp"))
    ask = _positive_float(latest_quote.get("ap"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2, str(latest_quote.get("t") or "")

    minute_bar = _mapping(snapshot.get("minuteBar"))
    price = _positive_float(minute_bar.get("c"))
    if price is not None:
        return price, str(minute_bar.get("t") or "")

    daily_bar = _mapping(snapshot.get("dailyBar"))
    price = _positive_float(daily_bar.get("c"))
    if price is not None:
        return price, str(daily_bar.get("t") or "")

    return None, ""


def _snapshot_low(snapshot: dict[str, Any]) -> tuple[float | None, str]:
    daily_bar = _mapping(snapshot.get("dailyBar"))
    low = _positive_float(daily_bar.get("l"))
    if low is not None:
        return low, str(daily_bar.get("t") or "")

    minute_bar = _mapping(snapshot.get("minuteBar"))
    low = _positive_float(minute_bar.get("l"))
    if low is not None:
        return low, str(minute_bar.get("t") or "")

    return None, ""


def _atr_percent(bars: list[dict[str, Any]], today: date) -> tuple[float | None, str]:
    completed_bars = [bar for bar in bars if _bar_date(bar) is not None and _bar_date(bar) < today]
    completed_bars.sort(key=lambda bar: str(bar.get("t") or ""))
    if len(completed_bars) < 22:
        return None, ""

    daily_ranges: list[float] = []
    previous_close: float | None = None
    for bar in completed_bars:
        high = _positive_float(bar.get("h"))
        low = _positive_float(bar.get("l"))
        close = _positive_float(bar.get("c"))
        if high is None or low is None or close is None:
            previous_close = close
            continue
        if previous_close is not None:
            effective_range = max(high, previous_close) - min(low, previous_close)
            daily_ranges.append((effective_range / previous_close) * 100)
        previous_close = close

    if len(daily_ranges) < 21:
        return None, ""

    return round(sum(daily_ranges[-21:]) / 21, 2), str(completed_bars[-1].get("t") or "")


def _bar_date(bar: dict[str, Any]) -> date | None:
    raw_timestamp = str(bar.get("t") or "")
    if not raw_timestamp:
        return None
    try:
        return date.fromisoformat(raw_timestamp[:10])
    except ValueError:
        return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
