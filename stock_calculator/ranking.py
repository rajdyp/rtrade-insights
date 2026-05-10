from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from stock_calculator.alpaca import AlpacaMarketDataClient, SUPPORTED_FEEDS
from stock_calculator.calculations import calculate_positions
from stock_calculator.config import AppConfig, load_config
from stock_calculator.risk import normalize_market_regime, strategy_mode_for_selection, suggested_risk_percent
from stock_calculator.robinhood import STRATEGY_OPTIONS, calculate_strategy_metrics, derive_fifo_trades
from stock_calculator.storage import load_planned_stops, load_robinhood_transactions


TABLE_COLUMNS = [
    "strategy",
    "mode",
    "market_regime",
    "symbol",
    "price",
    "stop",
    "atr",
    "stop_loss_percent",
    "risk_in_atr",
    "shares",
    "position_size",
    "risk_percent",
    "total_risk",
    "validation_error",
]

METADATA_COLUMNS = [
    "price_source",
    "stop_source",
    "atr_source",
    "market_data_feed",
    "price_timestamp",
    "stop_timestamp",
    "atr_timestamp",
]

RANK_COLUMNS = [*TABLE_COLUMNS, *METADATA_COLUMNS]

TABLE_HEADERS = {
    "strategy": "Strategy",
    "mode": "Mode",
    "market_regime": "Regime",
    "symbol": "Symbol",
    "price": "Price",
    "stop": "Stop",
    "atr": "ATR%",
    "stop_loss_percent": "Stop%",
    "risk_in_atr": "R/ATR",
    "shares": "Shares",
    "position_size": "Pos Size",
    "risk_percent": "Risk%",
    "total_risk": "Risk $",
    "validation_error": "Error",
}


@dataclass(frozen=True)
class ParseError:
    line: int
    message: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {"line": self.line, "message": self.message, "text": self.text}


@dataclass(frozen=True)
class ParsedCandidate:
    line: int
    strategy: str
    symbol: str
    price: float | None
    stop: float | None
    atr: float | None
    price_source: str = "manual"
    stop_source: str = "manual"
    atr_source: str = "manual"
    market_data_feed: str = ""
    price_timestamp: str = ""
    stop_timestamp: str = ""
    atr_timestamp: str = ""


@dataclass(frozen=True)
class RankResult:
    groups: dict[str, list[dict[str, object]]]
    errors: list[ParseError]
    warnings: list[str]
    generated_at: str
    market_regime: str

    @property
    def rows(self) -> list[dict[str, object]]:
        return [row for strategy in STRATEGY_OPTIONS for row in self.groups.get(strategy, [])]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "market_regime": self.market_regime,
            "groups": self.groups,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": self.warnings,
        }


def parse_rank_text(text: str, *, enrich: bool = False) -> tuple[list[ParsedCandidate], list[ParseError]]:
    candidates: list[ParsedCandidate] = []
    errors: list[ParseError] = []
    current_strategy = ""

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped in STRATEGY_OPTIONS:
            current_strategy = stripped
            continue

        if not current_strategy:
            errors.append(ParseError(line_number, "Row appears before a strategy header.", raw_line))
            continue

        parts = stripped.split()
        if len(parts) not in _allowed_part_counts(enrich):
            errors.append(ParseError(line_number, _expected_row_message(enrich), raw_line))
            continue

        symbol = parts[0].upper().strip()
        if not symbol:
            errors.append(ParseError(line_number, "Symbol is required.", raw_line))
            continue

        try:
            price, stop, atr = _parse_rank_values(parts, enrich=enrich)
        except ValueError:
            errors.append(ParseError(line_number, _numeric_error_message(parts, enrich=enrich), raw_line))
            continue

        candidates.append(
            ParsedCandidate(
                line=line_number,
                strategy=current_strategy,
                symbol=symbol,
                price=price,
                stop=stop,
                atr=atr,
                price_source="manual" if price is not None else "",
                stop_source="manual" if stop is not None else "",
                atr_source="manual" if atr is not None else "",
            )
        )

    return candidates, errors


def rank_candidates(
    text: str,
    *,
    config: AppConfig | None = None,
    strategy_metrics: pd.DataFrame | None = None,
    today: date | None = None,
    load_strategy_metrics: bool = True,
    enrich: bool = False,
    feed: str = "iex",
    market_data_provider: Any | None = None,
) -> RankResult:
    config = config or load_config()
    today = today or date.today()
    market_regime = normalize_market_regime(config.market_regime)
    candidates, errors = parse_rank_text(text, enrich=enrich)
    warnings: list[str] = []

    if enrich:
        candidates, enrichment_errors = enrich_rank_candidates(
            candidates,
            feed=feed,
            today=today,
            market_data_provider=market_data_provider,
        )
        errors.extend(enrichment_errors)

    if strategy_metrics is None and load_strategy_metrics:
        try:
            derivation = derive_fifo_trades(load_robinhood_transactions(), load_planned_stops())
            strategy_metrics = calculate_strategy_metrics(derivation.closed_trades)
        except Exception as exc:
            strategy_metrics = pd.DataFrame()
            warnings.append(f"Could not load strategy history; using Unknown mode for risk sizing. Detail: {exc}")
    elif strategy_metrics is None:
        strategy_metrics = pd.DataFrame()

    rows = []
    for candidate in candidates:
        mode = strategy_mode_for_selection(strategy_metrics, candidate.strategy)
        risk_percent = suggested_risk_percent(market_regime, mode, config.risk_percent)
        calculated = calculate_positions(
            pd.DataFrame(
                [
                    {
                        "symbol": candidate.symbol,
                        "buy_date": today.isoformat(),
                        "share_price": candidate.price,
                        "stop_price": candidate.stop,
                        "atr": candidate.atr,
                        "portfolio_amount": config.sizing_portfolio_amount,
                        "risk_percent": risk_percent,
                    }
                ]
            ),
            as_of=today,
        ).iloc[0]
        rows.append(_rank_row(candidate, calculated, mode, market_regime))

    return RankResult(
        groups=_group_rank_rows(rows),
        errors=errors,
        warnings=warnings,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        market_regime=market_regime,
    )


def enrich_rank_candidates(
    candidates: list[ParsedCandidate],
    *,
    feed: str = "iex",
    today: date | None = None,
    market_data_provider: Any | None = None,
) -> tuple[list[ParsedCandidate], list[ParseError]]:
    if feed not in SUPPORTED_FEEDS:
        raise ValueError("Unsupported Alpaca feed. Use iex, delayed_sip, or sip.")

    provider = market_data_provider or AlpacaMarketDataClient.from_env()
    symbols = [
        candidate.symbol
        for candidate in candidates
        if candidate.price is None or candidate.stop is None or candidate.atr is None
    ]
    market_data = provider.get_market_data(symbols, feed=feed, today=today) if symbols else {}

    enriched: list[ParsedCandidate] = []
    errors: list[ParseError] = []
    for candidate in candidates:
        if candidate.price is not None and candidate.stop is not None and candidate.atr is not None:
            enriched.append(candidate)
            continue

        data = market_data.get(candidate.symbol)
        if data is None:
            errors.append(ParseError(candidate.line, f"Could not enrich {candidate.symbol}; no Alpaca data returned.", candidate.symbol))
            continue

        price = candidate.price if candidate.price is not None else data.price
        if price is None:
            errors.append(ParseError(candidate.line, f"Could not enrich {candidate.symbol}; missing latest price.", candidate.symbol))
            continue

        stop = candidate.stop
        stop_timestamp = candidate.stop_timestamp
        if stop is None:
            if data.today_low is None:
                errors.append(ParseError(candidate.line, f"Could not enrich {candidate.symbol}; missing today's low.", candidate.symbol))
                continue
            stop = fallback_stop_from_low(data.today_low, price)
            stop_timestamp = data.low_timestamp

        atr = candidate.atr if candidate.atr is not None else data.atr_percent
        if atr is None:
            errors.append(ParseError(candidate.line, f"Could not enrich {candidate.symbol}; missing 21-day ATR%.", candidate.symbol))
            continue

        enriched.append(
            ParsedCandidate(
                line=candidate.line,
                strategy=candidate.strategy,
                symbol=candidate.symbol,
                price=price,
                stop=stop,
                atr=atr,
                price_source=candidate.price_source or "alpaca",
                stop_source=candidate.stop_source or "alpaca_low_buffer",
                atr_source=candidate.atr_source or "alpaca_marketsurge_21d",
                market_data_feed=feed,
                price_timestamp=candidate.price_timestamp or (data.price_timestamp if candidate.price is None else ""),
                stop_timestamp=stop_timestamp,
                atr_timestamp=candidate.atr_timestamp or (data.atr_timestamp if candidate.atr is None else ""),
            )
        )

    return enriched, errors


def fallback_stop_from_low(today_low: float, price: float) -> float:
    buffer = min(max(0.10, price * 0.002), 1.00)
    return round(today_low + buffer, 2)


def render_rank_result(result: RankResult, output_format: str = "table") -> str:
    normalized_format = str(output_format or "table").lower().strip()
    if normalized_format == "json":
        return json.dumps(result.to_dict(), indent=2)
    if normalized_format == "csv":
        return render_csv(result)
    if normalized_format == "table":
        return render_table(result)
    raise ValueError("Unsupported format. Use table, csv, or json.")


def render_csv(result: RankResult) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RANK_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for rows in result.groups.values():
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in RANK_COLUMNS})
    if result.errors or result.warnings:
        writer.writerow({})
        writer.writerow({"strategy": "Warnings/Errors"})
        for warning in result.warnings:
            writer.writerow({"strategy": "warning", "mode": warning})
        for error in result.errors:
            writer.writerow({"strategy": "error", "mode": f"Line {error.line}: {error.message}", "symbol": error.text})
    return buffer.getvalue()


def render_table(result: RankResult) -> str:
    table_rows = [_format_table_row(row) for row in result.rows]
    headers = [TABLE_HEADERS[column] for column in TABLE_COLUMNS]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows)) if table_rows else len(headers[index])
        for index in range(len(headers))
    ]

    lines = []
    for strategy, rows in result.groups.items():
        if not rows:
            continue
        if lines:
            lines.append("")
        lines.append(strategy)
        lines.append(_join_table_line(headers, widths))
        lines.append(_join_table_line(["-" * width for width in widths], widths))
        lines.extend(_join_table_line(_format_table_row(row), widths) for row in rows)
    if not table_rows:
        lines.append("No valid candidate rows parsed.")

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)

    if result.errors:
        lines.append("")
        lines.append("Errors:")
        lines.extend(f"- Line {error.line}: {error.message} ({error.text.strip()})" for error in result.errors)

    return "\n".join(lines) + "\n"


def _rank_row(
    candidate: ParsedCandidate,
    calculated: pd.Series,
    mode: str,
    market_regime: str,
) -> dict[str, object]:
    return {
        "strategy": candidate.strategy,
        "mode": mode,
        "market_regime": market_regime,
        "symbol": calculated["symbol"],
        "price": _optional_float(calculated["share_price"]),
        "stop": _optional_float(calculated["stop_price"]),
        "atr": _optional_float(calculated["atr"]),
        "stop_loss_percent": _optional_float(calculated["stop_loss_percent"]),
        "risk_in_atr": _optional_float(calculated["risk_in_atr"]),
        "shares": _optional_int(calculated["number_of_shares"]),
        "position_size": _optional_float(calculated["position_size"]),
        "risk_percent": _optional_float(calculated["risk_percent"]),
        "total_risk": _optional_float(calculated["risk_amount"]),
        "validation_error": str(calculated["validation_error"] or ""),
        "price_source": candidate.price_source,
        "stop_source": candidate.stop_source,
        "atr_source": candidate.atr_source,
        "market_data_feed": candidate.market_data_feed,
        "price_timestamp": candidate.price_timestamp,
        "stop_timestamp": candidate.stop_timestamp,
        "atr_timestamp": candidate.atr_timestamp,
    }


def _group_rank_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups = {strategy: [] for strategy in STRATEGY_OPTIONS}
    for row in sorted(rows, key=_rank_row_sort_key):
        strategy = str(row.get("strategy") or "")
        if strategy in groups:
            groups[strategy].append(row)
    return groups


def _rank_row_sort_key(row: dict[str, object]) -> tuple[int, bool, float, str]:
    strategy = str(row.get("strategy") or "")
    strategy_rank = STRATEGY_OPTIONS.index(strategy) if strategy in STRATEGY_OPTIONS else 99
    risk_in_atr = row.get("risk_in_atr")
    has_blank_risk = risk_in_atr is None or pd.isna(risk_in_atr)
    return (strategy_rank, has_blank_risk, float(risk_in_atr) if not has_blank_risk else float("inf"), str(row.get("symbol")))


def _format_table_row(row: dict[str, object]) -> list[str]:
    return [
        str(row.get("strategy") or ""),
        str(row.get("mode") or ""),
        str(row.get("market_regime") or ""),
        str(row.get("symbol") or ""),
        _format_number(row.get("price")),
        _format_number(row.get("stop")),
        _format_number(row.get("atr")),
        _format_number(row.get("stop_loss_percent")),
        _format_number(row.get("risk_in_atr")),
        _format_integer(row.get("shares")),
        _format_number(row.get("position_size")),
        _format_number(row.get("risk_percent")),
        _format_number(row.get("total_risk")),
        str(row.get("validation_error") or ""),
    ]


def _allowed_part_counts(enrich: bool) -> set[int]:
    return {1, 2, 4} if enrich else {4}


def _expected_row_message(enrich: bool) -> str:
    if enrich:
        return "Expected row format: SYMBOL, SYMBOL STOP, or SYMBOL PRICE STOP ATR%."
    return "Expected row format: SYMBOL PRICE STOP ATR%."


def _parse_rank_values(parts: list[str], *, enrich: bool) -> tuple[float | None, float | None, float | None]:
    if len(parts) == 4:
        return float(parts[1]), float(parts[2]), float(parts[3])
    if enrich and len(parts) == 2:
        return None, float(parts[1]), None
    if enrich and len(parts) == 1:
        return None, None, None
    raise ValueError


def _numeric_error_message(parts: list[str], *, enrich: bool) -> str:
    if enrich and len(parts) == 2:
        return "Stop must be numeric."
    return "Price, stop, and ATR must be numeric."


def _join_table_line(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values)).rstrip()


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _format_integer(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(int(value))


def _csv_value(value: object) -> object:
    return "" if value is None or pd.isna(value) else value
