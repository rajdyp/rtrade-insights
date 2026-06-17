import pytest

from stock_calculator.massive import MassiveMarketDataClient


def test_massive_client_fetches_snapshots_with_bearer_auth():
    session = FakeSession(
        {
            "tickers": [
                {
                    "ticker": "AAPL",
                    "lastTrade": {"p": 198.25},
                    "min": {"c": 198.0},
                    "day": {"c": 197.5},
                }
            ]
        }
    )
    client = MassiveMarketDataClient("key", session=session)

    prices = client.get_latest_prices([" aapl "])

    assert prices == {"AAPL": 198.25}
    assert session.calls[0]["url"] == "https://api.massive.com/v2/snapshot/locale/us/markets/stocks/tickers"
    assert session.calls[0]["headers"] == {"Authorization": "Bearer key"}
    assert session.calls[0]["params"] == {"tickers": "AAPL"}


def test_massive_client_falls_back_to_quote_midpoint_min_then_day_close():
    session = FakeSession(
        {
            "tickers": [
                {"ticker": "QUOTE", "lastQuote": {"p": 10.0, "P": 10.2}, "min": {"c": 9.9}},
                {"ticker": "MIN", "min": {"c": 22.5}, "day": {"c": 22.0}},
                {"ticker": "DAY", "day": {"c": 31.75}},
                {"ticker": "BAD", "lastTrade": {"p": 0}, "min": {"c": None}, "day": {"c": -1}},
                {"ticker": "EXTRA", "lastTrade": {"p": 99.0}},
            ]
        }
    )
    client = MassiveMarketDataClient("key", session=session)

    prices = client.get_latest_prices(["quote", "min", "day", "bad"])

    assert prices == {
        "QUOTE": 10.1,
        "MIN": 22.5,
        "DAY": 31.75,
    }


def test_massive_client_handles_verified_starter_snapshot_shape():
    session = FakeSession(
        {
            "tickers": [
                {
                    "ticker": "AAPL",
                    "todaysChangePerc": 0.1,
                    "todaysChange": 0.2,
                    "updated": 123,
                    "day": {"dv": 1, "o": 196.0, "h": 199.0, "l": 195.0, "c": 197.0, "v": 10, "vw": 197.2},
                    "min": {
                        "dv": 1,
                        "dav": 2,
                        "av": 3,
                        "t": 123,
                        "n": 4,
                        "o": 198.0,
                        "h": 198.5,
                        "l": 197.8,
                        "c": 198.4,
                        "v": 5,
                        "vw": 198.2,
                    },
                    "prevDay": {"c": 196.8},
                }
            ]
        }
    )
    client = MassiveMarketDataClient("key", session=session)

    prices = client.get_latest_prices(["AAPL"])

    assert prices == {"AAPL": 198.4}


def test_massive_client_rejects_missing_env_credentials(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
        MassiveMarketDataClient.from_env()


def test_massive_client_rejects_invalid_snapshot_shape():
    client = MassiveMarketDataClient("key", session=FakeSession({"tickers": {"AAPL": {}}}))

    with pytest.raises(ValueError, match="tickers list"):
        client.get_latest_prices(["AAPL"])


class FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, *, headers, params, timeout):
        self.calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return FakeResponse(self._payload)


class FakeResponse:
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload
