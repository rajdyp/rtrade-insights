from __future__ import annotations

from typing import Protocol

from stock_calculator.alpaca import AlpacaMarketDataClient
from stock_calculator.massive import MassiveMarketDataClient


PRICE_SOURCE_MASSIVE = "Massive"
PRICE_SOURCE_ALPACA = "Alpaca"
CAMPAIGN_PRICE_SOURCES = (PRICE_SOURCE_MASSIVE, PRICE_SOURCE_ALPACA)
DEFAULT_CAMPAIGN_PRICE_SOURCE = PRICE_SOURCE_MASSIVE
ALPACA_CAMPAIGN_PRICE_FEED = "iex"


class LatestPriceProvider(Protocol):
    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]: ...


class AlpacaLatestPriceProvider:
    def __init__(self, client: AlpacaMarketDataClient, *, feed: str = ALPACA_CAMPAIGN_PRICE_FEED) -> None:
        self._client = client
        self._feed = feed

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        return self._client.get_latest_prices(symbols, feed=self._feed)


def get_campaign_live_prices(
    symbols: list[str],
    source: str,
    *,
    massive_provider: LatestPriceProvider | None = None,
    alpaca_provider: LatestPriceProvider | None = None,
) -> dict[str, float]:
    normalized_source = _normalize_source(source)
    if normalized_source == PRICE_SOURCE_MASSIVE:
        provider = massive_provider or MassiveMarketDataClient.from_env()
        return provider.get_latest_prices(symbols)
    if normalized_source == PRICE_SOURCE_ALPACA:
        provider = alpaca_provider or AlpacaLatestPriceProvider(AlpacaMarketDataClient.from_env())
        return provider.get_latest_prices(symbols)
    raise ValueError(f"Unsupported campaign price source: {source}. Use Massive or Alpaca.")


def _normalize_source(source: str) -> str:
    candidate = str(source or "").strip().lower()
    if candidate == "massive":
        return PRICE_SOURCE_MASSIVE
    if candidate == "alpaca":
        return PRICE_SOURCE_ALPACA
    return str(source or "").strip()
