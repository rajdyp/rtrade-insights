import json
from pathlib import Path

from stock_calculator import cli
from stock_calculator.research import normalize_tickers, read_ticker_file


def test_read_ticker_file_normalizes_comments_blanks_lowercase_and_duplicates(tmp_path):
    ticker_file = tmp_path / "research_tickers.txt"
    ticker_file.write_text(
        """
        pins
        APP NVDA
        # skip this
        pins

        hood # comment
        """,
        encoding="utf-8",
    )

    assert read_ticker_file(ticker_file) == ["PINS", "APP", "NVDA", "HOOD"]


def test_normalize_tickers_preserves_first_seen_order():
    assert normalize_tickers(["pins", "APP", "pins", "", "nvda"]) == ["PINS", "APP", "NVDA"]


def test_rank_file_prints_table_output(tmp_path, capsys, monkeypatch):
    rank_file = tmp_path / "rank_candidates.txt"
    rank_file.write_text("EP\nTEST 100 95 5\n", encoding="utf-8")
    monkeypatch.setattr("stock_calculator.ranking.load_robinhood_transactions", lambda: _empty_transactions())
    monkeypatch.setattr("stock_calculator.ranking.load_planned_stops", lambda: _empty_planned_stops())

    exit_code = cli.main(["rank", "--file", rank_file.as_posix()])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Strategy" in output
    assert "TEST" in output
    assert output.endswith("\n")


def test_rank_file_supports_json_output(tmp_path, capsys, monkeypatch):
    rank_file = tmp_path / "rank_candidates.txt"
    rank_file.write_text("EP\nTEST 100 95 5\n", encoding="utf-8")
    monkeypatch.setattr("stock_calculator.ranking.load_robinhood_transactions", lambda: _empty_transactions())
    monkeypatch.setattr("stock_calculator.ranking.load_planned_stops", lambda: _empty_planned_stops())

    exit_code = cli.main(["rank", "--file", rank_file.as_posix(), "--format", "json"])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert "rows" not in payload
    assert payload["groups"]["EP"][0]["symbol"] == "TEST"
    assert payload["groups"]["5% BO"] == []
    assert payload["groups"]["BO"] == []


def test_rank_file_passes_enrichment_flags(tmp_path, capsys, monkeypatch):
    rank_file = tmp_path / "rank_candidates.txt"
    rank_file.write_text("EP\nTEST\n", encoding="utf-8")
    captured_call = {}

    def fake_rank_candidates(text, **kwargs):
        captured_call["text"] = text
        captured_call["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli, "rank_candidates", fake_rank_candidates)
    monkeypatch.setattr(cli, "render_rank_result", lambda result, output_format: f"{output_format}: ok\n")

    exit_code = cli.main(["rank", "--file", rank_file.as_posix(), "--enrich", "--feed", "delayed_sip", "--format", "json"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output == "json: ok\n"
    assert captured_call["text"] == "EP\nTEST\n"
    assert captured_call["kwargs"] == {"enrich": True, "feed": "delayed_sip"}


def test_missing_rank_file_returns_validation_error(capsys):
    exit_code = cli.main(["rank", "--file", "/missing/rank_candidates.txt"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Ranking file does not exist" in captured.err


def test_research_rejects_template_without_ticker(monkeypatch, capsys):
    called = False

    def fake_open_research_tabs(*args, **kwargs):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cli, "open_research_tabs", fake_open_research_tabs)

    exit_code = cli.main(["research", "PINS", "--template", "stock today"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--template must contain {ticker}" in captured.err
    assert called is False


def test_research_uses_file_and_tickers_without_opening_browser(tmp_path, monkeypatch):
    ticker_file = tmp_path / "research_tickers.txt"
    ticker_file.write_text("pins\nAPP\n", encoding="utf-8")
    captured_call = {}

    def fake_open_research_tabs(tickers, **kwargs):
        captured_call["tickers"] = tickers
        captured_call["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(cli, "open_research_tabs", fake_open_research_tabs)

    exit_code = cli.main(["research", "NVDA", "--file", ticker_file.as_posix(), "--no-keep-open"])

    assert exit_code == 0
    assert captured_call["tickers"] == ["NVDA", "PINS", "APP"]
    assert captured_call["kwargs"]["keep_open"] is False


def _empty_transactions():
    import pandas as pd

    from stock_calculator.robinhood import TRANSACTION_COLUMNS

    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def _empty_planned_stops():
    import pandas as pd

    from stock_calculator.robinhood import PLANNED_STOP_COLUMNS

    return pd.DataFrame(columns=PLANNED_STOP_COLUMNS)
