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
        4% BO
        pins 21.16 20.69 5.2

        EP
        NVDA 921.40 890.00 3.6
        """
    )

    assert errors == []
    assert [(candidate.strategy, candidate.symbol, candidate.price) for candidate in candidates] == [
        ("4% BO", "PINS", 21.16),
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
        "Numeric fields must be valid numbers.",
        "Expected row format: SYMBOL PRICE LOD ATR%, optionally with SL:<value>.",
    ]


def test_parse_rank_text_reads_enriched_symbol_and_strategy_based_value_rows():
    candidates, errors = parse_rank_text(
        """
        EP
        pins
        APP 100.25
        NVDA 921.40 890.00 3.6
        BO
        RIGL 27.83 4.54
        """,
        enrich=True,
    )

    assert errors == []
    assert [(candidate.symbol, candidate.price, candidate.stop, candidate.atr) for candidate in candidates] == [
        ("PINS", None, None, None),
        ("APP", None, 100.25, None),
        ("NVDA", 921.40, 889.00, 3.6),
        ("RIGL", None, 27.83, 4.54),
    ]
    assert candidates[1].stop_source == "manual_low_buffer"
    assert candidates[3].stop_source == "manual_low_buffer"
    assert candidates[3].atr_source == "manual"


def test_parse_rank_text_accepts_manual_sl_token_anywhere_after_symbol():
    candidates, errors = parse_rank_text(
        """
        EP
        FIRST SL:29 3.21
        MIDDLE 29.10 SL:29 3.21
        LAST 31.55 29.10 3.21 SL:29
        """,
        enrich=True,
    )

    assert errors == []
    assert [(candidate.symbol, candidate.price, candidate.stop, candidate.atr) for candidate in candidates] == [
        ("FIRST", None, 29.0, 3.21),
        ("MIDDLE", None, 29.0, 3.21),
        ("LAST", 31.55, 29.0, 3.21),
    ]
    assert [candidate.stop_source for candidate in candidates] == ["manual_sl", "manual_sl", "manual_sl"]


def test_parse_rank_text_rejects_malformed_manual_sl_tokens():
    candidates, errors = parse_rank_text(
        """
        EP
        BLANK SL:
        TEXT SL:nope
        DUP SL:29 SL:28
        JOINED SL:29:28
        MANY 31.55 29.10 3.21 1.2 SL:29
        """,
        enrich=True,
    )

    assert candidates == []
    assert [error.message for error in errors] == [
        "SL:<value> must include a numeric stop loss.",
        "SL:<value> must include a numeric stop loss.",
        "Only one SL:<value> token is allowed.",
        "SL:<value> must include a numeric stop loss.",
        "Expected row format: SYMBOL, SYMBOL LOD, SYMBOL LOD ATR%, SYMBOL PRICE LOD ATR%, or add SL:<value>.",
    ]


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
        "Expected row format: SYMBOL PRICE LOD ATR%, optionally with SL:<value>.",
        "Expected row format: SYMBOL PRICE LOD ATR%, optionally with SL:<value>.",
    ]


def test_fallback_stop_uses_minimum_percentage_cap_and_rounding():
    assert fallback_stop_from_low(20.00, 20.00) == 19.90
    assert fallback_stop_from_low(100.00, 100.00) == 99.80
    assert fallback_stop_from_low(100.00, 1000.00) == 99.00
    assert fallback_stop_from_low(50.00, 66.665) == 49.87


def test_rank_candidates_uses_risk_matrix_strategy_modes_and_sorts_within_strategy_groups():
    result = rank_candidates(
        """
        4% BO
        PINS 21.16 20.69 5.2

        EP
        NVDA 100.00 95.00 5.0
        AAPL 100.00 99.00 5.0
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame(
            [
                {"strategy": "EP", "mode": "Working"},
                {"strategy": "4% BO", "mode": "Weak"},
            ]
        ),
        today=date(2026, 5, 7),
    )

    assert result.warnings == []
    assert [row["symbol"] for row in result.rows] == ["AAPL", "NVDA", "PINS"]
    assert list(result.groups) == ["EP", "4% BO", "BO"]
    assert [row["symbol"] for row in result.groups["EP"]] == ["AAPL", "NVDA"]
    assert result.groups["4% BO"][0]["symbol"] == "PINS"
    assert result.groups["EP"][0]["mode"] == "Working"
    assert result.groups["EP"][0]["risk_percent"] == 0.5
    assert result.groups["EP"][0]["risk_in_atr"] == 0.24
    assert result.groups["4% BO"][0]["mode"] == "Weak"
    assert result.groups["4% BO"][0]["risk_percent"] == 0.12
    assert result.groups["4% BO"][0]["risk_in_atr"] == 0.52


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


def test_rank_candidates_reports_gate_closed_for_zero_matrix_risk():
    result = rank_candidates(
        """
        EP
        TEST 100 95 5
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="NO-GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "EP", "mode": "Failing"}]),
        today=date(2026, 5, 7),
    )

    expected_error = "Gate closed: NO-GO / Failing; no new sizing."
    assert result.rows[0]["risk_percent"] == 0.0
    assert result.rows[0]["validation_error"] == expected_error
    assert expected_error in render_table(result)
    assert expected_error in render_csv(result)

    json_payload = json.loads(render_rank_result(result, "json"))
    assert json_payload["groups"]["EP"][0]["validation_error"] == expected_error


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
    assert row["raw_price"] is None
    assert row["sizing_price_buffer"] is None
    assert row["stop"] == 99.6
    assert row["atr"] == 5.0
    assert row["price_source"] == "alpaca"
    assert row["stop_source"] == "alpaca_low_buffer"
    assert row["atr_source"] == "alpaca_marketsurge_21d"
    assert row["market_data_feed"] == "delayed_sip"
    assert row["price_timestamp"] == "2026-05-07T15:59:00Z"
    assert row["stop_timestamp"] == "2026-05-07T00:00:00Z"
    assert row["atr_timestamp"] == "2026-05-06T00:00:00Z"


def test_rank_candidates_interprets_enriched_value_rows_as_lod_for_all_strategies():
    provider = FakeMarketDataProvider(
        {
            symbol: SimpleNamespace(
                price=30.0,
                today_low=20.0,
                atr_percent=5.5,
                price_timestamp=f"{symbol}-price-time",
                low_timestamp=f"{symbol}-low-time",
                atr_timestamp=f"{symbol}-atr-time",
            )
            for symbol in ["EPONE", "EPATR", "FIVE", "BOONE", "BOATR"]
        }
    )

    result = rank_candidates(
        """
        EP
        EPONE 27.83
        EPATR 27.83 4.54

        4% BO
        FIVE 27.83 4.54

        BO
        BOONE 27.83
        BOATR 27.83 4.54
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame(
            [
                {"strategy": "EP", "mode": "Working"},
                {"strategy": "4% BO", "mode": "Working"},
                {"strategy": "BO", "mode": "Working"},
            ]
        ),
        today=date(2026, 5, 7),
        enrich=True,
        feed="delayed_sip",
        market_data_provider=provider,
    )

    rows = {row["symbol"]: row for row in result.rows}

    assert rows["EPONE"]["price"] == 30.0
    assert rows["EPONE"]["stop"] == 27.73
    assert rows["EPONE"]["atr"] == 5.5
    assert rows["EPONE"]["stop_source"] == "manual_low_buffer"
    assert rows["EPONE"]["atr_source"] == "alpaca_marketsurge_21d"
    assert rows["EPONE"]["stop_timestamp"] == ""

    assert rows["EPATR"]["price"] == 30.0
    assert rows["EPATR"]["stop"] == 27.73
    assert rows["EPATR"]["atr"] == 4.54
    assert rows["EPATR"]["stop_source"] == "manual_low_buffer"
    assert rows["EPATR"]["atr_source"] == "manual"

    assert rows["FIVE"]["price"] == 30.0
    assert rows["FIVE"]["stop"] == 27.73
    assert rows["FIVE"]["atr"] == 4.54
    assert rows["FIVE"]["stop_source"] == "manual_low_buffer"
    assert rows["FIVE"]["atr_source"] == "manual"

    assert rows["BOONE"]["price"] == 30.0
    assert rows["BOONE"]["stop"] == 27.73
    assert rows["BOONE"]["atr"] == 5.5
    assert rows["BOONE"]["stop_source"] == "manual_low_buffer"
    assert rows["BOONE"]["atr_source"] == "alpaca_marketsurge_21d"
    assert rows["BOONE"]["stop_timestamp"] == ""

    assert rows["BOATR"]["price"] == 30.0
    assert rows["BOATR"]["stop"] == 27.73
    assert rows["BOATR"]["atr"] == 4.54
    assert rows["BOATR"]["stop_source"] == "manual_low_buffer"
    assert rows["BOATR"]["atr_source"] == "manual"


def test_rank_candidates_applies_iex_sizing_cushion_to_alpaca_price_after_manual_sl_stop():
    provider = FakeMarketDataProvider(
        {
            "TEST": SimpleNamespace(
                price=31.43,
                today_low=29.50,
                atr_percent=5.7,
                price_timestamp="2026-05-07T15:59:00Z",
                low_timestamp="2026-05-07T00:00:00Z",
                atr_timestamp="2026-05-06T00:00:00Z",
            )
        }
    )

    result = rank_candidates(
        """
        BO
        TEST SL:29.76 5.7
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "BO", "mode": "Working"}]),
        today=date(2026, 5, 7),
        enrich=True,
        feed="iex",
        market_data_provider=provider,
    )

    row = result.rows[0]
    assert row["price"] == 31.51
    assert row["raw_price"] == 31.43
    assert row["sizing_price_buffer"] == 0.08
    assert row["stop"] == 29.76
    assert row["stop_source"] == "manual_sl"
    assert row["shares"] == 110
    assert row["position_size"] == 3466.10


def test_rank_candidates_keeps_strategy_lod_stop_parsing_before_iex_sizing_cushion():
    provider = FakeMarketDataProvider(
        {
            "EPTEST": SimpleNamespace(
                price=31.43,
                today_low=29.50,
                atr_percent=5.7,
                price_timestamp="EPTEST-price-time",
                low_timestamp="EPTEST-low-time",
                atr_timestamp="EPTEST-atr-time",
            ),
            "FIVETEST": SimpleNamespace(
                price=31.43,
                today_low=29.50,
                atr_percent=5.7,
                price_timestamp="FIVETEST-price-time",
                low_timestamp="FIVETEST-low-time",
                atr_timestamp="FIVETEST-atr-time",
            ),
        }
    )

    result = rank_candidates(
        """
        EP
        EPTEST 29.76 5.7

        4% BO
        FIVETEST 29.76 5.7
        """,
        config=AppConfig(sizing_portfolio_amount=4_620, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame(
            [
                {"strategy": "EP", "mode": "Working"},
                {"strategy": "4% BO", "mode": "Working"},
            ]
        ),
        today=date(2026, 5, 7),
        enrich=True,
        feed="iex",
        market_data_provider=provider,
    )

    rows = {row["symbol"]: row for row in result.rows}

    assert rows["EPTEST"]["price"] == 31.51
    assert rows["EPTEST"]["raw_price"] == 31.43
    assert rows["EPTEST"]["sizing_price_buffer"] == 0.08
    assert rows["EPTEST"]["stop"] == 29.66
    assert rows["EPTEST"]["stop_source"] == "manual_low_buffer"

    assert rows["FIVETEST"]["price"] == 31.51
    assert rows["FIVETEST"]["raw_price"] == 31.43
    assert rows["FIVETEST"]["sizing_price_buffer"] == 0.08
    assert rows["FIVETEST"]["stop"] == 29.66
    assert rows["FIVETEST"]["stop_source"] == "manual_low_buffer"


def test_rank_candidates_manual_sl_overrides_lod_computed_stop():
    result = rank_candidates(
        """
        EP
        TEST 31.55 29.10 3.21 SL:29
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "EP", "mode": "Working"}]),
        today=date(2026, 5, 7),
    )

    row = result.rows[0]
    assert row["price"] == 31.55
    assert row["stop"] == 29.0
    assert row["atr"] == 3.21
    assert row["stop_source"] == "manual_sl"


def test_rank_candidates_does_not_apply_sizing_cushion_to_sip_feeds_or_manual_price_rows():
    provider = FakeMarketDataProvider(
        {
            "SIPTEST": SimpleNamespace(
                price=31.43,
                today_low=29.50,
                atr_percent=5.7,
                price_timestamp="SIPTEST-price-time",
                low_timestamp="SIPTEST-low-time",
                atr_timestamp="SIPTEST-atr-time",
            )
        }
    )

    sip_result = rank_candidates(
        """
        BO
        SIPTEST SL:29.76 5.7
        """,
        config=AppConfig(sizing_portfolio_amount=4_620, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "BO", "mode": "Working"}]),
        today=date(2026, 5, 7),
        enrich=True,
        feed="sip",
        market_data_provider=provider,
    )

    sip_row = sip_result.rows[0]
    assert sip_row["price"] == 31.43
    assert sip_row["raw_price"] is None
    assert sip_row["sizing_price_buffer"] is None
    assert sip_row["shares"] == 13

    manual_provider = FakeMarketDataProvider({})
    manual_result = rank_candidates(
        """
        BO
        MANUAL 31.43 29.76 5.7 SL:29.76
        """,
        config=AppConfig(sizing_portfolio_amount=4_620, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame([{"strategy": "BO", "mode": "Working"}]),
        today=date(2026, 5, 7),
        enrich=True,
        feed="iex",
        market_data_provider=manual_provider,
    )

    manual_row = manual_result.rows[0]
    assert manual_provider.calls == []
    assert manual_row["price"] == 31.43
    assert manual_row["raw_price"] is None
    assert manual_row["sizing_price_buffer"] is None
    assert manual_row["shares"] == 13


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
    table_header = table.splitlines()[1]
    csv_header = csv_output.splitlines()[0].split(",")

    assert table_header.startswith("Symbol  Regime")
    assert table_header.split()[:4] == ["Symbol", "Regime", "Mode", "Strategy"]
    assert "TEST" in table
    assert table.endswith("\n")
    assert csv_header[:14] == [
        "symbol",
        "market_regime",
        "mode",
        "strategy",
        "price",
        "stop",
        "atr",
        "stop_loss_percent",
        "position_size",
        "risk_percent",
        "total_risk",
        "shares",
        "risk_in_atr",
        "validation_error",
    ]
    assert csv_header[14:] == [
        "price_source",
        "raw_price",
        "sizing_price_buffer",
        "stop_source",
        "atr_source",
        "market_data_feed",
        "price_timestamp",
        "stop_timestamp",
        "atr_timestamp",
    ]
    assert "TEST" in csv_output
    assert "rows" not in json_payload
    assert list(json_payload["groups"]) == ["EP", "4% BO", "BO"]
    assert json_payload["groups"]["EP"][0]["symbol"] == "TEST"
    assert json_payload["groups"]["EP"][0]["risk_percent"] == 0.5
    assert "price_source" in json_payload["groups"]["EP"][0]
    assert json_payload["groups"]["4% BO"] == []
    assert json_payload["groups"]["BO"] == []


def test_table_and_csv_render_rows_by_strategy_group_order():
    result = rank_candidates(
        """
        BO
        BOA 100 99 5

        4% BO
        FIVE 100 99 5

        EP
        EPZ 100 95 5
        EPA 100 99 5
        """,
        config=AppConfig(sizing_portfolio_amount=19_250, risk_percent=0.25, market_regime="SELECTIVE GO"),
        strategy_metrics=pd.DataFrame(
            [
                {"strategy": "EP", "mode": "Working"},
                {"strategy": "4% BO", "mode": "Working"},
                {"strategy": "BO", "mode": "Working"},
            ]
        ),
        today=date(2026, 5, 7),
    )

    table = render_table(result)
    csv_output = render_csv(result)

    table_lines = table.splitlines()
    assert table_lines.index("EP") < table_lines.index("4% BO") < table_lines.index("BO")
    assert table.index("EPA") < table.index("EPZ") < table.index("FIVE") < table.index("BOA")
    csv_lines = csv_output.splitlines()
    assert [line.split(",")[0] for line in csv_lines[1:]] == ["EPA", "EPZ", "FIVE", "BOA"]


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
