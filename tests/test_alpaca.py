from datetime import date, timedelta

import pytest

from stock_calculator.alpaca import AlpacaMarketDataClient, _atr_percent


def test_alpaca_client_fetches_snapshots_and_daily_bars_with_feed():
    session = FakeSession()
    client = AlpacaMarketDataClient("key", "secret", session=session)

    data = client.get_market_data(["test"], feed="iex", today=date(2026, 5, 7))

    assert [call["params"]["feed"] for call in session.calls] == ["iex", "iex"]
    assert session.calls[0]["headers"]["APCA-API-KEY-ID"] == "key"
    assert session.calls[0]["headers"]["APCA-API-SECRET-KEY"] == "secret"
    assert data["TEST"].price == 100.0
    assert data["TEST"].today_low == 99.8
    assert data["TEST"].atr_percent == 18.18
    assert data["TEST"].price_timestamp == "2026-05-07T15:59:00Z"
    assert data["TEST"].low_timestamp == "2026-05-07T00:00:00Z"
    assert data["TEST"].atr_timestamp == "2026-05-06T00:00:00Z"


def test_alpaca_client_rejects_missing_credentials():
    with pytest.raises(ValueError, match="Alpaca credentials are required"):
        AlpacaMarketDataClient("", "")


def test_alpaca_client_maps_delayed_sip_to_sip_for_historical_daily_bars():
    session = FakeSession()
    client = AlpacaMarketDataClient("key", "secret", session=session)

    client.get_market_data(["test"], feed="delayed_sip", today=date(2026, 5, 7))

    assert [call["params"]["feed"] for call in session.calls] == ["delayed_sip", "sip"]


def test_alpaca_client_latest_prices_fetches_snapshots_only():
    session = FakeSession()
    client = AlpacaMarketDataClient("key", "secret", session=session)

    prices = client.get_latest_prices(["test"], feed="iex")

    assert prices == {"TEST": 100.0}
    assert len(session.calls) == 1
    assert session.calls[0]["url"].endswith("/stocks/snapshots")
    assert session.calls[0]["params"]["feed"] == "iex"


def test_atr_percent_uses_marketsurge_style_daily_range_percent_not_true_range_over_price():
    bars = [{"h": 100.0, "l": 100.0, "c": 100.0, "t": "2026-04-01T00:00:00Z"}]
    for index in range(21):
        bars.append(
            {
                "h": 110.0,
                "l": 100.0,
                "c": 105.0,
                "t": (date(2026, 4, 2) + timedelta(days=index)).isoformat() + "T00:00:00Z",
            }
        )

    atr_percent, timestamp = _atr_percent(bars, date(2026, 5, 7))

    assert atr_percent == 9.55
    assert atr_percent != 10.0
    assert timestamp == "2026-04-22T00:00:00Z"


def test_atr_percent_includes_gap_up_and_gap_down_range_from_previous_close():
    bars = [{"h": 100.0, "l": 100.0, "c": 100.0, "t": "2026-04-01T00:00:00Z"}]
    for index in range(21):
        if index % 2 == 0:
            high, low = 112.0, 108.0
        else:
            high, low = 95.0, 88.0
        bars.append(
            {
                "h": high,
                "l": low,
                "c": 100.0,
                "t": (date(2026, 4, 2) + timedelta(days=index)).isoformat() + "T00:00:00Z",
            }
        )

    atr_percent, timestamp = _atr_percent(bars, date(2026, 5, 7))

    assert atr_percent == 12.0
    assert timestamp == "2026-04-22T00:00:00Z"


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, *, headers, params, timeout):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        if url.endswith("/stocks/snapshots"):
            return FakeResponse(
                {
                    "snapshots": {
                        "TEST": {
                            "latestTrade": {"p": 100.0, "t": "2026-05-07T15:59:00Z"},
                            "dailyBar": {"l": 99.8, "c": 100.0, "t": "2026-05-07T00:00:00Z"},
                        }
                    }
                }
            )
        return FakeResponse({"bars": {"TEST": _daily_bars()}})


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _daily_bars():
    start = date(2026, 4, 15)
    return [
        {"h": 12.0, "l": 10.0, "c": 11.0, "t": (start + timedelta(days=index)).isoformat() + "T00:00:00Z"}
        for index in range(22)
    ]
