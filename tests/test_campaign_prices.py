import pytest

from stock_calculator.campaign_prices import get_campaign_live_prices


def test_get_campaign_live_prices_uses_massive_provider():
    provider = FakeProvider({"AAPL": 198.4})

    prices = get_campaign_live_prices(["AAPL"], "Massive", massive_provider=provider)

    assert prices == {"AAPL": 198.4}
    assert provider.symbols == ["AAPL"]


def test_get_campaign_live_prices_uses_alpaca_provider():
    provider = FakeProvider({"AAPL": 198.25})

    prices = get_campaign_live_prices(["AAPL"], "Alpaca", alpaca_provider=provider)

    assert prices == {"AAPL": 198.25}
    assert provider.symbols == ["AAPL"]


def test_get_campaign_live_prices_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unsupported campaign price source"):
        get_campaign_live_prices(["AAPL"], "Other")


class FakeProvider:
    def __init__(self, prices):
        self._prices = prices
        self.symbols = None

    def get_latest_prices(self, symbols):
        self.symbols = symbols
        return self._prices
