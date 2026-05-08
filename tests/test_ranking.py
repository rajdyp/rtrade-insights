import json
from datetime import date

import pandas as pd

from stock_calculator.config import AppConfig
from stock_calculator.ranking import parse_rank_text, rank_candidates, render_csv, render_rank_result, render_table


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
