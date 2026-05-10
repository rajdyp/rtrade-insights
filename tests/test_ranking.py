import json
from datetime import date
from types import SimpleNamespace

import pandas as pd

from stock_calculator.config import AppConfig
from stock_calculator.ranking import (
    fallback_stop_from_low,
    parse_rank_text,
    rank_candidates,
    render_csv,
    render_rank_result,
    render_table,
)


def test_parse_rank_text_reads_grouped_compact_rows():
    candidates, errors = parse_rank_text(
        """
        # Candidates
        5% BO
        pins 21.16 20.69 5.2

        EP
        NVDA 921.40 890.00 3.6
        """
    )

    assert errors == []
    assert [(candidate.strategy, candidate.symbol, candidate.price) for candidate in candidates] == [
        ("5% BO", "PINS", 21.16),
        ("EP", "NVDA", 921.40),
    ]


def test_parse_rank_text_reports_malformed_rows_with_line_numbers():
    candidates, errors = parse_rank_text(
        """
        PINS 21.16 20.69 5.2
        EP
        BAD 10 nope 2
        SHORT 10 9
        """
    )

    assert candidates == []
    assert [error.line for error in errors] == [2, 4, 5]
    assert [error.message for error in errors] == [
        "Row appears before a strategy header.",
        "Price, stop, and ATR must be numeric.",
        "Expected row format: SYMBOL PRICE STOP ATR%.",
    ]


def test_parse_rank_text_reads_enriched_symbol_and_manual_stop_rows():
    candidates, errors = parse_rank_text(
        """
        EP
        pins
        APP 100.25
        NVDA 921.40 890.00 3.6
        """,
        enrich=True,
    )

    assert errors == []
    assert [(candidate.symbol, candidate.price, candidate.stop, candidate.atr) for candidate in candidates] == [
        ("PINS", None, None, None),
        ("APP", None, 100.25, None),
        ("NVDA", 921.40, 890.00, 3.6),
    ]
    assert candidates[1].stop_source == "manual"


def test_parse_rank_text_rejects_partial_rows_without_enrichment():
    candidates, errors = parse_rank_text(
        """
        EP
        PINS
        APP 100.25
        """
    )

    assert candidates == []
    assert [error.message for error in errors] == [
        "Expected row format: SYMBOL PRICE STOP ATR%.",
        "Expected row format: SYMBOL PRICE STOP ATR%.",
    ]


def test_fallback_stop_uses_minimum_percentage_cap_and_rounding():
    assert fallback_stop_from_low(20.00, 20.00) == 20.10
    assert fallback_stop_from_low(100.00, 100.00) == 100.20
    assert fallback_stop_from_low(100.00, 1000.00) == 101.00
    assert fallback_stop_from_low(50.00, 66.665) == 50.13


def test_rank_candidates_uses_risk_matrix_strategy_modes_and_sorts_within_strategy_groups():
    result = rank_candidates(
        """
        5% BO
        PINS 21.16 20.69 5.2

        EP
        NVDA 100.00 95.00 5.0
        AAPL 100.00 99.00 5.0
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame(
            [
                {"strategy": "EP", "mode": "Working"},
                {"strategy": "5% BO", "mode": "Weak"},
            ]
        ),
        today=date(2026, 5, 7),
    )

    assert result.warnings == []
    assert [row["symbol"] for row in result.rows] == ["AAPL", "NVDA", "PINS"]
    assert list(result.groups) == ["EP", "5% BO", "BO"]
    assert [row["symbol"] for row in result.groups["EP"]] == ["AAPL", "NVDA"]
    assert result.groups["5% BO"][0]["symbol"] == "PINS"
    assert result.groups["EP"][0]["mode"] == "Working"
    assert result.groups["EP"][0]["risk_percent"] == 0.5
    assert result.groups["EP"][0]["risk_in_atr"] == 0.2
    assert result.groups["5% BO"][0]["mode"] == "Weak"
    assert result.groups["5% BO"][0]["risk_percent"] == 0.12
    assert result.groups["5% BO"][0]["risk_in_atr"] == 0.43


def test_rank_candidates_uses_unknown_mode_when_strategy_metrics_are_missing():
    result = rank_candidates(
        """
        BO
        PLTR 24.80 23.60 4.1
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "EP", "mode": "Working"}]),
        today=date(2026, 5, 7),
    )

    assert result.rows[0]["mode"] == "Unknown"
    assert result.rows[0]["risk_percent"] == 0.06


def test_rank_candidates_enriches_symbol_only_rows_and_records_sources():
    provider = FakeMarketDataProvider(
        {
            "TEST": SimpleNamespace(
                price=100.0,
                today_low=99.8,
                atr_percent=5.0,
                price_timestamp="2026-05-07T15:59:00Z",
                low_timestamp="2026-05-07T00:00:00Z",
                atr_timestamp="2026-05-06T00:00:00Z",
            )
        }
    )

    result = rank_candidates(
        """
        EP
        TEST
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "EP", "mode": "Working"}]),
        today=date(2026, 5, 7),
        enrich=True,
        feed="delayed_sip",
        market_data_provider=provider,
    )

    row = result.rows[0]
    assert provider.calls == [(["TEST"], "delayed_sip", date(2026, 5, 7))]
    assert row["price"] == 100.0
    assert row["stop"] == 100.0
    assert row["atr"] == 5.0
    assert row["price_source"] == "alpaca"
    assert row["stop_source"] == "alpaca_low_buffer"
    assert row["atr_source"] == "alpaca_marketsurge_21d"
    assert row["market_data_feed"] == "delayed_sip"
    assert row["price_timestamp"] == "2026-05-07T15:59:00Z"
    assert row["stop_timestamp"] == "2026-05-07T00:00:00Z"
    assert row["atr_timestamp"] == "2026-05-06T00:00:00Z"


def test_rank_candidates_preserves_manual_stop_during_enrichment():
    provider = FakeMarketDataProvider(
        {
            "TEST": SimpleNamespace(
                price=100.0,
                today_low=98.0,
                atr_percent=4.5,
                price_timestamp="price-time",
                low_timestamp="low-time",
                atr_timestamp="atr-time",
            )
        }
    )

    result = rank_candidates(
        """
        EP
        TEST 99.25
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "EP", "mode": "Working"}]),
        today=date(2026, 5, 7),
        enrich=True,
        market_data_provider=provider,
    )

    row = result.rows[0]
    assert row["price"] == 100.0
    assert row["stop"] == 99.25
    assert row["atr"] == 4.5
    assert row["stop_source"] == "manual"
    assert row["stop_timestamp"] == ""


def test_rank_candidates_reports_row_level_enrichment_failures_and_continues():
    provider = FakeMarketDataProvider(
        {
            "GOOD": SimpleNamespace(
                price=100.0,
                today_low=99.8,
                atr_percent=5.0,
                price_timestamp="",
                low_timestamp="",
                atr_timestamp="",
            )
        }
    )

    result = rank_candidates(
        """
        EP
        GOOD
        BAD
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "EP", "mode": "Working"}]),
        today=date(2026, 5, 7),
        enrich=True,
        market_data_provider=provider,
    )

    assert [row["symbol"] for row in result.rows] == ["GOOD"]
    assert [error.message for error in result.errors] == ["Could not enrich BAD; no Alpaca data returned."]


def test_table_csv_and_json_render_same_ranked_rows():
    result = rank_candidates(
        """
        EP
        TEST 100 95 5
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "EP", "mode": "Working"}]),
        today=date(2026, 5, 7),
    )

    table = render_table(result)
    csv_output = render_csv(result)
    json_output = render_rank_result(result, "json")
    json_payload = json.loads(json_output)

    assert "Strategy" in table
    assert "TEST" in table
    assert table.endswith("\n")
    assert "strategy,mode,market_regime,symbol" in csv_output
    assert "TEST" in csv_output
    assert "rows" not in json_payload
    assert list(json_payload["groups"]) == ["EP", "5% BO", "BO"]
    assert json_payload["groups"]["EP"][0]["symbol"] == "TEST"
    assert json_payload["groups"]["EP"][0]["risk_percent"] == 0.5
    assert "price_source" in json_payload["groups"]["EP"][0]
    assert json_payload["groups"]["5% BO"] == []
    assert json_payload["groups"]["BO"] == []


def test_table_and_csv_render_rows_by_strategy_group_order():
    result = rank_candidates(
        """
        BO
        BOA 100 99 5

        5% BO
        FIVE 100 99 5

        EP
        EPZ 100 95 5
        EPA 100 99 5
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame(
            [
                {"strategy": "EP", "mode": "Working"},
                {"strategy": "5% BO", "mode": "Working"},
                {"strategy": "BO", "mode": "Working"},
            ]
        ),
        today=date(2026, 5, 7),
    )

    table = render_table(result)
    csv_output = render_csv(result)

    table_lines = table.splitlines()
    assert table_lines.index("EP") < table_lines.index("5% BO") < table_lines.index("BO")
    assert table.index("EPA") < table.index("EPZ") < table.index("FIVE") < table.index("BOA")
    csv_lines = csv_output.splitlines()
    assert [line.split(",")[3] for line in csv_lines[1:]] == ["EPA", "EPZ", "FIVE", "BOA"]


def test_storage_load_failure_returns_warning_and_unknown_mode(monkeypatch):
    def fail_load_robinhood_transactions():
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr("stock_calculator.ranking.load_robinhood_transactions", fail_load_robinhood_transactions)

    result = rank_candidates(
        """
        EP
        TEST 100 95 5
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        today=date(2026, 5, 7),
    )

    assert result.warnings == [
        "Could not load strategy history; using Unknown mode for risk sizing. Detail: storage unavailable"
    ]
    assert result.rows[0]["mode"] == "Unknown"
    assert result.rows[0]["risk_percent"] == 0.06


class FakeMarketDataProvider:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def get_market_data(self, symbols, *, feed, today):
        self.calls.append((symbols, feed, today))
        return self.data
