from __future__ import annotations

import csv
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from stock_calculator.calculations import weekday_hold_count


ROBINHOOD_COLUMNS = [
    "Activity Date",
    "Process Date",
    "Settle Date",
    "Instrument",
    "Description",
    "Trans Code",
    "Quantity",
    "Price",
    "Amount",
]

TRADE_CODES = {"Buy", "Sell"}
STRATEGY_OPTIONS = ["EP", "5% BO", "BO"]
UNCLASSIFIED_STRATEGY = "Unclassified"

TRANSACTION_COLUMNS = [
    "activity_date",
    "process_date",
    "settle_date",
    "symbol",
    "description",
    "trans_code",
    "quantity",
    "price",
    "amount",
]

PLANNED_STOP_COLUMNS = [
    "symbol",
    "buy_date",
    "quantity",
    "planned_stop",
    "strategy",
    "atr",
    "market_regime",
]

CLOSED_TRADE_COLUMNS = [
    "symbol",
    "buy_date",
    "sell_date",
    "quantity",
    "planned_stop",
    "strategy",
    "atr",
    "market_regime",
    "buy_price",
    "buy_amount",
    "sell_price",
    "sell_amount",
    "realized_pnl",
    "realized_pnl_percent",
    "hold_days",
]

OPEN_LOT_COLUMNS = [
    "symbol",
    "buy_date",
    "quantity",
    "planned_stop",
    "strategy",
    "atr",
    "market_regime",
    "buy_price",
    "cost_basis",
    "hold_days",
]

UNMATCHED_SELL_COLUMNS = ["symbol", "sell_date", "quantity", "sell_price"]
IMPORT_ISSUE_COLUMNS = ["row_number", "issue", "raw_row"]
PLANNED_STOP_ISSUE_COLUMNS = ["symbol", "buy_date", "quantity", "issue", "detail"]
MISSING_PLANNED_STOP_COLUMNS = ["status", "symbol", "buy_date", "sell_date", "quantity", "buy_price", "detail"]
LOT_GROUPING_AUDIT_COLUMNS = [
    "symbol",
    "buy_date",
    "planned_quantity",
    "split_quantities",
    "buy_price",
    "planned_stop",
    "strategy",
    "atr",
    "market_regime",
    "reason",
]
STRATEGY_MODE_WINDOW = 15
STRATEGY_MODE_LOOKBACK = 20
STRATEGY_MODE_ADJUSTMENT_K = 0.5
STRATEGY_MODE_WORKING_THRESHOLD = 0.30
STRATEGY_MODE_FAILING_THRESHOLD = -0.10
ATTRIBUTION_TREND_THRESHOLD = 0.25
ATTRIBUTION_HOLD_TIME_RATIO_THRESHOLD = 1.50
ATTRIBUTION_HOLD_TIME_RATIO_DELTA_THRESHOLD = 0.30
STRATEGY_METRIC_COLUMNS = [
    "strategy",
    "trade_count",
    "total_realized_pnl",
    "win_rate",
    "expectancy",
    "profit_factor",
    "average_win",
    "average_loss",
    "average_win_r",
    "average_loss_r",
    "r_ratio",
    "average_win_hold",
    "average_loss_hold",
    "rolling_mode_exp",
    "mode_adjusted_score",
    "mode",
    "action",
]
STRATEGY_ATTRIBUTION_COLUMNS = [
    "strategy",
    "mode",
    "mode_basis",
    "trend",
    "trend_driver",
    "evidence",
    "playbook",
]


@dataclass(frozen=True)
class ImportResult:
    transactions: pd.DataFrame
    ignored_rows: pd.DataFrame
    malformed_rows: pd.DataFrame


@dataclass(frozen=True)
class TradeDerivation:
    closed_trades: pd.DataFrame
    exit_matches: pd.DataFrame
    open_lots: pd.DataFrame
    unmatched_sells: pd.DataFrame
    missing_planned_stops: int
    planned_stop_issues: pd.DataFrame
    missing_planned_stop_rows: pd.DataFrame
    lot_grouping_audit: pd.DataFrame


@dataclass(frozen=True)
class RollingModeScores:
    rolling_exp: float | None
    adjusted_score: float | None
    valid_count: int
    examined_count: int
    skipped_count: int
    lookback_count: int


def calculate_trade_metrics(closed_trades: pd.DataFrame) -> dict[str, float | int | None]:
    if closed_trades.empty:
        return _empty_trade_metrics()

    trades = closed_trades.copy()
    trades["realized_pnl"] = pd.to_numeric(trades["realized_pnl"], errors="coerce")
    trades["realized_pnl_percent"] = pd.to_numeric(trades["realized_pnl_percent"], errors="coerce")
    trades["hold_days"] = pd.to_numeric(trades["hold_days"], errors="coerce")
    trades = trades.dropna(subset=["realized_pnl"]).reset_index(drop=True)
    if trades.empty:
        return _empty_trade_metrics()

    r_trades = trades.copy()
    for column in ["planned_stop", "buy_price", "quantity", "atr"]:
        r_trades[column] = pd.to_numeric(r_trades.get(column), errors="coerce")
    r_trades["initial_risk"] = (r_trades["buy_price"] - r_trades["planned_stop"]) * r_trades["quantity"]
    r_trades = r_trades[r_trades["initial_risk"] > 0].copy()
    r_trades["r_multiple"] = r_trades["realized_pnl"] / r_trades["initial_risk"]
    r_trades["risk_in_atr"] = _risk_in_atr_series(r_trades)
    r_wins = r_trades[r_trades["realized_pnl"] > 0]
    r_losses = r_trades[r_trades["realized_pnl"] < 0]
    average_win_r = _mean(r_wins["r_multiple"])
    average_loss_r = _mean(r_losses["r_multiple"])
    expectancy_r = _mean(r_trades["r_multiple"])

    wins = trades[trades["realized_pnl"] > 0]
    losses = trades[trades["realized_pnl"] < 0]
    breakevens = trades[trades["realized_pnl"] == 0]
    gross_win = float(wins["realized_pnl"].sum())
    gross_loss = abs(float(losses["realized_pnl"].sum()))
    average_win = _mean(wins["realized_pnl"])
    average_loss = _mean(losses["realized_pnl"])
    average_win_hold = _mean(wins["hold_days"])
    average_loss_hold = _mean(losses["hold_days"])
    hold_time_ratio = (
        average_loss_hold / average_win_hold
        if average_win_hold is not None and average_win_hold > 0 and average_loss_hold is not None
        else None
    )

    return {
        "trade_count": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "breakeven_count": len(breakevens),
        "win_rate": round((len(wins) / len(trades)) * 100, 1),
        "win_loss_ratio": _safe_round(average_win / abs(average_loss)) if average_win is not None and average_loss else None,
        "average_win_r": _safe_round(average_win_r),
        "average_loss_r": _safe_round(average_loss_r),
        "r_ratio": _safe_round(average_win_r / abs(average_loss_r))
        if average_win_r is not None and average_loss_r
        else None,
        "average_risk_in_atr": _safe_round(_mean(r_trades["risk_in_atr"])),
        "expectancy_r": _safe_round(expectancy_r),
        "expectancy": _safe_round(_mean(trades["realized_pnl"])),
        "profit_factor": _safe_round(gross_win / gross_loss) if gross_loss else None,
        "average_win": _safe_round(average_win),
        "average_win_percent": _safe_round(_mean(wins["realized_pnl_percent"])),
        "average_win_hold": _safe_round(average_win_hold, digits=1),
        "win_streak": _longest_streak(trades, winning=True),
        "top_win": _safe_round(wins["realized_pnl"].max()) if not wins.empty else None,
        "average_loss": _safe_round(average_loss),
        "average_loss_percent": _safe_round(_mean(losses["realized_pnl_percent"])),
        "average_loss_hold": _safe_round(average_loss_hold, digits=1),
        "hold_time_ratio": _safe_round(hold_time_ratio),
        "loss_streak": _longest_streak(trades, winning=False),
        "top_loss": _safe_round(losses["realized_pnl"].min()) if not losses.empty else None,
    }


def calculate_total_realized_pnl(closed_trades: pd.DataFrame) -> float:
    if closed_trades.empty or "realized_pnl" not in closed_trades.columns:
        return 0.0

    realized_pnl = pd.to_numeric(closed_trades["realized_pnl"], errors="coerce")
    return round(float(realized_pnl.dropna().sum()), 2)


def calculate_rolling_mode(closed_trades: pd.DataFrame) -> dict[str, str]:
    rolling_scores = _calculate_rolling_mode_scores(closed_trades)
    if rolling_scores is None or rolling_scores.rolling_exp is None or rolling_scores.adjusted_score is None:
        found = rolling_scores.valid_count if rolling_scores is not None else 0
        lookback = rolling_scores.lookback_count if rolling_scores is not None else min(len(closed_trades), STRATEGY_MODE_LOOKBACK)
        return {
            "rolling_mode_exp": "N/A",
            "mode_adjusted_score": "N/A",
            "mode": "Unknown",
            "action": "Tiny size only",
            "mode_basis": f"Need {STRATEGY_MODE_WINDOW} valid R trades; found {found} in latest {lookback} closed trades",
        }
    rolling_exp = rolling_scores.rolling_exp
    adjusted_score = rolling_scores.adjusted_score
    rolling_exp = round(rolling_exp, 2)
    adjusted_score = round(adjusted_score, 2)

    if rolling_exp > STRATEGY_MODE_WORKING_THRESHOLD:
        mode = "Working"
        action = "Normal size"
    elif rolling_exp >= 0:
        mode = "Caution"
        action = "Half size"
    elif rolling_exp >= STRATEGY_MODE_FAILING_THRESHOLD:
        mode = "Weak"
        action = "Quarter size"
    else:
        mode = "Failing"
        action = "Probe only / pause"

    mode_basis = f"{STRATEGY_MODE_WINDOW}R {rolling_exp:+.2f}R | Adj {adjusted_score:+.2f}R"
    if rolling_scores.skipped_count:
        plural = "" if rolling_scores.skipped_count == 1 else "s"
        mode_basis = (
            f"{STRATEGY_MODE_WINDOW} valid R trades from latest {rolling_scores.examined_count} closed trades; "
            f"skipped {rolling_scores.skipped_count} missing/invalid stop{plural} | {mode_basis}"
        )

    return {
        "rolling_mode_exp": f"{rolling_exp:+.2f}R",
        "mode_adjusted_score": f"{adjusted_score:+.2f}R",
        "mode": mode,
        "action": action,
        "mode_basis": mode_basis,
    }


def calculate_rolling_10r_mode(closed_trades: pd.DataFrame) -> dict[str, str]:
    rolling_mode = calculate_rolling_mode(closed_trades)
    return {**rolling_mode, "rolling_10r_exp": rolling_mode["rolling_mode_exp"]}


def calculate_strategy_metrics(closed_trades: pd.DataFrame) -> pd.DataFrame:
    if closed_trades.empty:
        return pd.DataFrame(columns=STRATEGY_METRIC_COLUMNS)

    trades = closed_trades.copy()
    strategy_values = trades["strategy"] if "strategy" in trades.columns else pd.Series([""] * len(trades), index=trades.index)
    trades["strategy_group"] = strategy_values.apply(display_strategy)
    rows = []
    grouped_by_strategy = dict(tuple(trades.groupby("strategy_group", sort=False, dropna=False)))
    for strategy in [*STRATEGY_OPTIONS, UNCLASSIFIED_STRATEGY]:
        grouped_trades = grouped_by_strategy.get(strategy)
        if grouped_trades is None:
            continue
        metrics = calculate_trade_metrics(grouped_trades)
        if metrics["trade_count"] == 0:
            continue
        rolling_mode = calculate_rolling_mode(grouped_trades)
        rows.append(
            {
                "strategy": strategy,
                "trade_count": metrics["trade_count"],
                "total_realized_pnl": calculate_total_realized_pnl(grouped_trades),
                "win_rate": metrics["win_rate"],
                "expectancy": metrics["expectancy"],
                "profit_factor": metrics["profit_factor"],
                "average_win": metrics["average_win"],
                "average_loss": metrics["average_loss"],
                "average_win_r": metrics["average_win_r"],
                "average_loss_r": metrics["average_loss_r"],
                "r_ratio": metrics["r_ratio"],
                "average_win_hold": metrics["average_win_hold"],
                "average_loss_hold": metrics["average_loss_hold"],
                "rolling_mode_exp": rolling_mode["rolling_mode_exp"],
                "mode_adjusted_score": rolling_mode["mode_adjusted_score"],
                "mode": rolling_mode["mode"],
                "action": rolling_mode["action"],
            }
        )

    if not rows:
        return pd.DataFrame(columns=STRATEGY_METRIC_COLUMNS)
    return pd.DataFrame(rows, columns=STRATEGY_METRIC_COLUMNS)


def calculate_strategy_attribution(closed_trades: pd.DataFrame, *, market_regime: str | None = None) -> pd.DataFrame:
    if closed_trades.empty:
        return pd.DataFrame(columns=STRATEGY_ATTRIBUTION_COLUMNS)

    trades = closed_trades.copy()
    strategy_values = trades["strategy"] if "strategy" in trades.columns else pd.Series([""] * len(trades), index=trades.index)
    trades["strategy_group"] = strategy_values.apply(display_strategy)
    rows = []
    grouped_by_strategy = dict(tuple(trades.groupby("strategy_group", sort=False, dropna=False)))
    for strategy in [*STRATEGY_OPTIONS, UNCLASSIFIED_STRATEGY]:
        grouped_trades = grouped_by_strategy.get(strategy)
        if grouped_trades is None:
            continue
        rows.append(_strategy_attribution_row(strategy, grouped_trades, market_regime=market_regime))

    if not rows:
        return pd.DataFrame(columns=STRATEGY_ATTRIBUTION_COLUMNS)
    return pd.DataFrame(rows, columns=STRATEGY_ATTRIBUTION_COLUMNS)


def portfolio_attribution_note(strategy_attribution: pd.DataFrame | None) -> str:
    deteriorating = _deteriorating_strategy_names(strategy_attribution)
    if len(deteriorating) < 2:
        return ""
    return (
        f"Multiple strategies deteriorating simultaneously: {', '.join(deteriorating)}. "
        "Review market regime and trading behavior before setup-specific adjustments."
    )


def _strategy_attribution_row(
    strategy: str,
    grouped_trades: pd.DataFrame,
    *,
    market_regime: str | None = None,
) -> dict[str, str]:
    ordered = grouped_trades.copy()
    ordered["_original_order"] = range(len(ordered))
    ordered["sell_date"] = pd.to_datetime(ordered.get("sell_date"), errors="coerce")
    ordered = ordered.sort_values(["sell_date", "_original_order"], kind="mergesort")

    trade_count = len(ordered)
    if trade_count < STRATEGY_MODE_WINDOW:
        metrics = calculate_trade_metrics(ordered)
        return {
            "strategy": strategy,
            "mode": "Unknown",
            "mode_basis": f"Need {STRATEGY_MODE_WINDOW} valid R trades",
            "trend": f"Need {STRATEGY_MODE_WINDOW} trades",
            "trend_driver": f"Need {STRATEGY_MODE_WINDOW} trades",
            "evidence": _low_sample_evidence(trade_count, metrics),
            "playbook": _attribution_playbook("Unknown", f"Need {STRATEGY_MODE_WINDOW} trades", "", market_regime),
        }

    recent = ordered.tail(STRATEGY_MODE_WINDOW)
    recent_metrics = calculate_trade_metrics(recent)
    rolling_mode = calculate_rolling_mode(ordered)
    prior = ordered.iloc[-(STRATEGY_MODE_WINDOW * 2) : -STRATEGY_MODE_WINDOW]
    prior_metrics = calculate_trade_metrics(prior) if len(prior) >= STRATEGY_MODE_WINDOW else None
    deltas = {
        "profit_factor": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "profit_factor"),
        "expectancy_r": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "expectancy_r"),
        "win_rate": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "win_rate"),
        "average_win_r": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "average_win_r"),
        "average_loss_r": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "average_loss_r"),
        "loss_streak": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "loss_streak"),
        "average_risk_in_atr": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "average_risk_in_atr"),
        "hold_time_ratio": _metric_delta_from_optional_prior(recent_metrics, prior_metrics, "hold_time_ratio"),
    }
    mode = rolling_mode["mode"]
    mode_basis = _mode_basis(rolling_mode)
    trend = _attribution_trend(mode, deltas)
    regime_driver = _regime_driver(recent, mode, trend)
    trend_driver = regime_driver or _trend_metric_driver(mode, trend, deltas, recent_metrics)
    evidence = _attribution_evidence(recent, prior, recent_metrics, prior_metrics, deltas, regime_driver)
    playbook = _attribution_playbook(mode, trend_driver, trend, market_regime)

    return {
        "strategy": strategy,
        "mode": mode,
        "mode_basis": mode_basis,
        "trend": trend,
        "trend_driver": trend_driver,
        "evidence": evidence,
        "playbook": playbook,
    }


def _mode_basis(rolling_mode: dict[str, str]) -> str:
    mode_basis = rolling_mode.get("mode_basis")
    if mode_basis:
        return mode_basis
    rolling_exp = rolling_mode.get("rolling_mode_exp")
    adjusted_score = rolling_mode.get("mode_adjusted_score")
    if rolling_exp == "N/A" or not rolling_exp:
        return f"Need {STRATEGY_MODE_WINDOW} valid R trades"
    if adjusted_score == "N/A" or not adjusted_score:
        return f"{STRATEGY_MODE_WINDOW}R {rolling_exp}"
    return f"{STRATEGY_MODE_WINDOW}R {rolling_exp} | Adj {adjusted_score}"


def _metric_delta_from_optional_prior(
    recent_metrics: dict[str, float | int | None],
    prior_metrics: dict[str, float | int | None] | None,
    key: str,
) -> float | None:
    if prior_metrics is None:
        return None
    return _metric_delta(recent_metrics.get(key), prior_metrics.get(key))


def _metric_delta(recent_value: Any, prior_value: Any) -> float | None:
    if recent_value is None or prior_value is None or pd.isna(recent_value) or pd.isna(prior_value):
        return None
    return round(float(recent_value) - float(prior_value), 2)


def _attribution_trend(mode: str, deltas: dict[str, float | None]) -> str:
    expectancy_delta = deltas.get("expectancy_r")
    if expectancy_delta is None:
        return f"Need {STRATEGY_MODE_WINDOW * 2} trades"
    if _delta_at_or_above(expectancy_delta, ATTRIBUTION_TREND_THRESHOLD):
        label = "Recovering" if mode in {"Weak", "Failing"} else "Improving"
        return f"{label} ({expectancy_delta:+.2f}R)"
    if _delta_at_or_below(expectancy_delta, -ATTRIBUTION_TREND_THRESHOLD):
        return f"Weakening ({expectancy_delta:+.2f}R)"
    return f"Flat ({expectancy_delta:+.2f}R)"


def _trend_metric_driver(
    mode: str,
    trend: str,
    deltas: dict[str, float | None],
    metrics: dict[str, float | int | None],
) -> str:
    trend_label = _trend_label(trend)
    if trend_label in {"Improving", "Recovering"}:
        candidates = [
            ("Loss control", deltas.get("average_loss_r"), 0.20),
            ("Winner size", deltas.get("average_win_r"), 0.20),
            ("Hit rate", deltas.get("win_rate"), 10.0),
            ("Profit factor", deltas.get("profit_factor"), 0.30),
            ("Expectancy R", deltas.get("expectancy_r"), 0.25),
        ]
        drivers = [label for label, delta, threshold in candidates if _delta_at_or_above(delta, threshold)]
        return _join_drivers(drivers)

    if trend_label == "Weakening":
        candidates = [
            ("Loss control", deltas.get("average_loss_r"), -0.20),
            ("Winner size", deltas.get("average_win_r"), -0.20),
            ("Hit rate", deltas.get("win_rate"), -10.0),
            ("Loss streak", deltas.get("loss_streak"), 2.0),
            ("Profit factor", deltas.get("profit_factor"), -0.30),
            ("Expectancy R", deltas.get("expectancy_r"), -ATTRIBUTION_TREND_THRESHOLD),
        ]
        drivers = []
        if _hold_time_driver_active(metrics, deltas):
            drivers.append("Hold time")
        for label, delta, threshold in candidates:
            if label == "Loss streak":
                if _delta_at_or_above(delta, threshold):
                    drivers.append(label)
            elif _delta_at_or_below(delta, threshold):
                drivers.append(label)
        return _join_drivers(drivers)

    if mode in {"Working", "Caution"}:
        if deltas.get("expectancy_r") is None:
            return _current_strengths_driver(metrics)
        return _flat_metric_driver(deltas)
    return "No clear driver"


def _regime_driver(recent: pd.DataFrame, mode: str, trend: str) -> str | None:
    if "market_regime" not in recent.columns:
        return None

    tagged = recent.copy()
    tagged["market_regime"] = tagged["market_regime"].fillna("").astype(str).str.strip()
    tagged = tagged[tagged["market_regime"] != ""]
    if len(tagged) < 6:
        return None

    overall = calculate_trade_metrics(tagged).get("expectancy_r")
    if overall is None or pd.isna(overall):
        return None

    best_regime = None
    worst_regime = None
    best_expectancy = None
    worst_expectancy = None
    for regime, rows in tagged.groupby("market_regime", sort=False):
        if len(rows) < 3:
            continue
        expectancy = calculate_trade_metrics(rows).get("expectancy_r")
        if expectancy is None or pd.isna(expectancy):
            continue
        if best_expectancy is None or expectancy > best_expectancy:
            best_regime = regime
            best_expectancy = expectancy
        if worst_expectancy is None or expectancy < worst_expectancy:
            worst_regime = regime
            worst_expectancy = expectancy

    trend_label = _trend_label(trend)
    if (mode in {"Weak", "Failing"} or trend_label == "Weakening") and worst_regime and worst_expectancy is not None:
        other_expectancy = _expectancy_excluding_regime(tagged, worst_regime)
        if other_expectancy is not None and worst_expectancy <= other_expectancy - 0.25:
            return f"Regime filter: {worst_regime}"
    if best_regime and best_expectancy is not None:
        other_expectancy = _expectancy_excluding_regime(tagged, best_regime)
        if other_expectancy is not None and best_expectancy >= other_expectancy + 0.25:
            return f"Regime strength: {best_regime}"
    return None


def _expectancy_excluding_regime(tagged: pd.DataFrame, regime: str) -> float | None:
    others = tagged[tagged["market_regime"] != regime]
    if len(others) < 3:
        return None
    expectancy = calculate_trade_metrics(others).get("expectancy_r")
    if expectancy is None or pd.isna(expectancy):
        return None
    return float(expectancy)


def _regime_warmup_status(recent: pd.DataFrame, regime_driver: str | None) -> str:
    if regime_driver or "market_regime" not in recent.columns:
        return ""

    tagged = recent.copy()
    tagged["market_regime"] = tagged["market_regime"].fillna("").astype(str).str.strip()
    tagged = tagged[tagged["market_regime"] != ""]
    if len(tagged) < 6:
        return f"Regime attribution: need {6 - len(tagged)} more tagged trades"

    regime_counts = tagged.groupby("market_regime", sort=False).size()
    if int((regime_counts >= 3).sum()) < 2:
        return "Regime attribution: need at least 3 tagged trades in 2 regimes"
    return ""


def _attribution_evidence(
    recent: pd.DataFrame,
    prior: pd.DataFrame,
    recent_metrics: dict[str, float | int | None],
    prior_metrics: dict[str, float | int | None] | None,
    deltas: dict[str, float | None],
    regime_driver: str | None,
) -> str:
    if prior_metrics is None:
        pf_text = _metric_value("PF", recent_metrics.get("profit_factor"))
        exp_text = _metric_value("Exp R", recent_metrics.get("expectancy_r"))
        win_text = _percent_metric_value("Win rate", recent_metrics.get("win_rate"))
        winner_size_text = _metric_value("Avg win R", recent_metrics.get("average_win_r"))
        loss_text = _metric_value("Avg loss R", recent_metrics.get("average_loss_r"))
        risk_text = _metric_value("Risk/ATR", recent_metrics.get("average_risk_in_atr"))
    else:
        pf_text = _metric_comparison("PF", recent_metrics.get("profit_factor"), prior_metrics.get("profit_factor"))
        exp_text = _metric_comparison("Exp R", recent_metrics.get("expectancy_r"), prior_metrics.get("expectancy_r"))
        win_text = _percent_metric_comparison(
            "Win rate",
            recent_metrics.get("win_rate"),
            prior_metrics.get("win_rate"),
            deltas.get("win_rate"),
        )
        winner_size_text = _metric_comparison_with_delta(
            "Avg win R",
            recent_metrics.get("average_win_r"),
            prior_metrics.get("average_win_r"),
            deltas.get("average_win_r"),
        )
        loss_text = _metric_comparison_with_delta(
            "Avg loss R",
            recent_metrics.get("average_loss_r"),
            prior_metrics.get("average_loss_r"),
            deltas.get("average_loss_r"),
        )
        risk_text = _risk_in_atr_text(
            recent_metrics.get("average_risk_in_atr"),
            prior_metrics.get("average_risk_in_atr"),
            deltas.get("average_risk_in_atr"),
        )
    hold_text = _hold_time_ratio_text(
        recent_metrics.get("hold_time_ratio"),
        None if prior_metrics is None else prior_metrics.get("hold_time_ratio"),
        deltas.get("hold_time_ratio"),
    )
    window_text = _window_span_text(recent, prior if prior_metrics is not None else None)
    regime_text = _regime_warmup_status(recent, regime_driver)
    return " | ".join(
        part
        for part in [
            pf_text,
            exp_text,
            win_text,
            winner_size_text,
            loss_text,
            risk_text,
            hold_text,
            window_text,
            regime_text,
        ]
        if part
    )


def _low_sample_evidence(trade_count: int, metrics: dict[str, float | int | None]) -> str:
    parts = [f"{trade_count} closed trades"]
    expectancy = _metric_value("Exp R", metrics.get("expectancy_r"))
    if expectancy:
        parts.append(expectancy)
        parts.append("directional only until 15 valid R trades")
    return " | ".join(parts)


def _metric_value(label: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{label} {_format_metric_value(value)}"


def _metric_comparison(label: str, recent_value: Any, prior_value: Any) -> str:
    if recent_value is None or prior_value is None or pd.isna(recent_value) or pd.isna(prior_value):
        return ""
    return f"{label} {_format_metric_value(recent_value)} vs {_format_metric_value(prior_value)}"


def _metric_comparison_with_delta(label: str, recent_value: Any, prior_value: Any, delta: float | None) -> str:
    comparison = _metric_comparison(label, recent_value, prior_value)
    if not comparison:
        return ""
    if delta is None or pd.isna(delta):
        return comparison
    return f"{comparison} ({delta:+.2f})"


def _percent_metric_value(label: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{label} {float(value):.1f}%"


def _percent_metric_comparison(label: str, recent_value: Any, prior_value: Any, delta: float | None) -> str:
    if recent_value is None or prior_value is None or pd.isna(recent_value) or pd.isna(prior_value):
        return ""
    comparison = f"{label} {float(recent_value):.1f}% vs {float(prior_value):.1f}%"
    if delta is None or pd.isna(delta):
        return comparison
    return f"{comparison} ({delta:+.1f} pts)"


def _risk_in_atr_text(recent_value: Any, prior_value: Any, delta: float | None) -> str:
    if recent_value is None or pd.isna(recent_value):
        return ""
    if prior_value is None or pd.isna(prior_value):
        return _metric_value("Risk/ATR", recent_value)
    if delta is None:
        return _metric_comparison("Risk/ATR", recent_value, prior_value)
    return f"Risk/ATR {_format_metric_value(recent_value)} vs {_format_metric_value(prior_value)} ({delta:+.2f})"


def _hold_time_ratio_text(recent_value: Any, prior_value: Any, delta: float | None) -> str:
    if recent_value is None or pd.isna(recent_value):
        return ""
    if prior_value is None or pd.isna(prior_value):
        return _metric_value("Hold ratio", recent_value)
    if delta is None:
        return _metric_comparison("Hold ratio", recent_value, prior_value)
    return f"Hold ratio {_format_metric_value(recent_value)} vs {_format_metric_value(prior_value)} ({delta:+.2f})"


def _format_metric_value(value: Any) -> str:
    return f"{float(value):.2f}"


def _trend_label(trend: str) -> str:
    return trend.split(" ", maxsplit=1)[0]


def _attribution_playbook(mode: str, trend_driver: str, trend: str, market_regime: str | None) -> str:
    regime = str(market_regime or "").strip().upper()
    trend_label = _trend_label(trend)
    if mode == "Working" and regime == "NO-GO":
        return "Strategy is working, but NO-GO regime limits sizing. Wait for GO conditions for full deployment."
    if mode == "Working" and trend_label == "Weakening":
        driver_detail = "" if trend_driver == "No clear driver" else f" ({trend_driver})"
        return f"Early warning: expectancy is declining{driver_detail}. Monitor closely and reduce size proactively if trend continues."
    if mode == "Caution" and trend_label == "Weakening":
        return "Trend weakening from Caution. Reduce to Weak-level sizing proactively rather than waiting for mode to drop."
    if mode == "Caution" and trend_label in {"Improving", "Recovering"}:
        return "Trend is recovering. Watch for Working status before resuming normal sizing."
    if mode in {"Working", "Caution"}:
        return "Keep using Market Regime and Strategy Mode sizing while monitoring the listed drivers."
    if mode == "Unknown":
        return "Insufficient valid R-trade history; maintain minimum sizing until 15 valid R trades are available."
    if trend_driver == "No clear driver":
        return "Keep risk reduced until mode improves."
    if trend_driver.startswith("Regime filter"):
        return "Tighten entry regime filter before resuming normal activity."
    driver_advice = _driver_advice(trend_driver)
    if driver_advice:
        return driver_advice
    return "Keep risk reduced until mode improves."


def _join_drivers(drivers: list[str]) -> str:
    return " + ".join(drivers) if drivers else "No clear driver"


def _flat_metric_driver(deltas: dict[str, float | None]) -> str:
    candidates = [
        ("Loss control", deltas.get("average_loss_r"), 0.20),
        ("Winner size", deltas.get("average_win_r"), 0.20),
        ("Hit rate", deltas.get("win_rate"), 10.0),
        ("Profit factor", deltas.get("profit_factor"), 0.30),
    ]
    drivers = [label for label, delta, threshold in candidates if _delta_at_or_above(delta, threshold)]
    if not drivers:
        return "No clear driver"
    return f"Contributors: {_join_drivers(drivers)}"


def _current_strengths_driver(metrics: dict[str, float | int | None]) -> str:
    candidates = [
        ("Hit rate", metrics.get("win_rate"), 50.0),
        ("Winner size", metrics.get("average_win_r"), 1.50),
        ("Loss control", metrics.get("average_loss_r"), -0.75),
        ("Profit factor", metrics.get("profit_factor"), 1.30),
        ("Expectancy R", metrics.get("expectancy_r"), STRATEGY_MODE_WORKING_THRESHOLD),
    ]
    drivers = [label for label, value, threshold in candidates if _metric_at_or_above(value, threshold)]
    if not drivers:
        return "No clear driver"
    return f"Current strengths: {_join_drivers(drivers)}"


def _driver_includes(trend_driver: str, label: str) -> bool:
    return label in _driver_parts(trend_driver)


def _driver_parts(trend_driver: str) -> list[str]:
    normalized = trend_driver.replace("Contributors:", "").replace("Current strengths:", "")
    return [part.strip() for part in normalized.split("+")]


def _driver_advice(trend_driver: str) -> str:
    advice_by_driver = [
        ("Hold time", "Losses are being held significantly longer than wins. Enforce time-based or rule-based exits on losing positions."),
        ("Loss control", "Tighten stop discipline and avoid entries where risk can expand."),
        ("Winner size", "Require cleaner reward/risk and avoid taking profits too quickly."),
        ("Hit rate", "Increase entry selectivity and reduce marginal setups."),
        ("Loss streak", "Probe only or pause until the setup stabilizes."),
    ]
    advice = [message for driver, message in advice_by_driver if _driver_includes(trend_driver, driver)]
    return " ".join(advice[:3])


def _metric_at_or_above(value: Any, threshold: float) -> bool:
    return value is not None and not pd.isna(value) and float(value) >= threshold


def _hold_time_driver_active(metrics: dict[str, float | int | None], deltas: dict[str, float | None]) -> bool:
    ratio = metrics.get("hold_time_ratio")
    delta = deltas.get("hold_time_ratio")
    return (
        ratio is not None
        and not pd.isna(ratio)
        and float(ratio) >= ATTRIBUTION_HOLD_TIME_RATIO_THRESHOLD
        and _delta_at_or_above(delta, ATTRIBUTION_HOLD_TIME_RATIO_DELTA_THRESHOLD)
    )


def _deteriorating_strategy_names(strategy_attribution: pd.DataFrame | None) -> list[str]:
    if strategy_attribution is None or strategy_attribution.empty:
        return []
    names = []
    for _, row in strategy_attribution.iterrows():
        strategy = str(row.get("strategy") or "").strip()
        mode = str(row.get("mode") or "").strip()
        trend = str(row.get("trend") or "").strip()
        if not strategy or mode == "Unknown" or trend.startswith("Need "):
            continue
        if mode in {"Weak", "Failing"} or _trend_label(trend) == "Weakening":
            names.append(strategy)
    return names


def _window_span_text(recent: pd.DataFrame, prior: pd.DataFrame | None) -> str:
    recent_text = _single_window_span_text("Recent", recent)
    prior_text = _single_window_span_text("Prior", prior) if prior is not None else ""
    return " | ".join(part for part in [recent_text, prior_text] if part)


def _single_window_span_text(label: str, window: pd.DataFrame | None) -> str:
    if window is None or window.empty or "sell_date" not in window.columns:
        return ""
    dates = pd.to_datetime(window["sell_date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    days = int((dates.max() - dates.min()).days) + 1
    trade_word = "trade" if len(window) == 1 else "trades"
    day_word = "day" if days == 1 else "days"
    return f"{label}: {len(window)} {trade_word} over {days} {day_word}"


def _risk_in_atr_series(trades: pd.DataFrame) -> pd.Series:
    atr = pd.to_numeric(trades.get("atr"), errors="coerce")
    buy_price = pd.to_numeric(trades.get("buy_price"), errors="coerce")
    planned_stop = pd.to_numeric(trades.get("planned_stop"), errors="coerce")
    stop_loss_percent = ((buy_price - planned_stop) / buy_price) * 100
    risk_in_atr = stop_loss_percent / atr
    return risk_in_atr.where((atr > 0) & (buy_price > 0) & (planned_stop < buy_price))


def _delta_at_or_above(delta: float | None, threshold: float) -> bool:
    return delta is not None and delta >= threshold


def _delta_at_or_below(delta: float | None, threshold: float) -> bool:
    return delta is not None and delta <= threshold


def _calculate_rolling_mode_exp(closed_trades: pd.DataFrame) -> float | None:
    scores = _calculate_rolling_mode_scores(closed_trades)
    if scores is None:
        return None
    return scores.rolling_exp


def _calculate_rolling_mode_scores(closed_trades: pd.DataFrame) -> RollingModeScores | None:
    if len(closed_trades) == 0:
        return None

    trades = closed_trades.copy()
    trades["sell_date"] = pd.to_datetime(trades.get("sell_date"), errors="coerce")
    trades = trades.sort_values("sell_date", kind="mergesort").tail(STRATEGY_MODE_LOOKBACK).reset_index(drop=True)

    for column in ["realized_pnl", "buy_price", "planned_stop", "quantity"]:
        trades[column] = pd.to_numeric(trades.get(column), errors="coerce")

    trades["initial_risk"] = (trades["buy_price"] - trades["planned_stop"]) * trades["quantity"]
    trades["r_multiple"] = trades["realized_pnl"] / trades["initial_risk"]
    valid_mask = (trades["initial_risk"] > 0) & trades["r_multiple"].notna()

    selected_indices: list[Any] = []
    examined_count = 0
    skipped_count = 0
    for index, is_valid in reversed(list(valid_mask.items())):
        examined_count += 1
        if is_valid:
            selected_indices.append(index)
            if len(selected_indices) == STRATEGY_MODE_WINDOW:
                break
        else:
            skipped_count += 1

    if len(selected_indices) != STRATEGY_MODE_WINDOW:
        valid_count = int(valid_mask.sum())
        return RollingModeScores(
            rolling_exp=None,
            adjusted_score=None,
            valid_count=valid_count,
            examined_count=len(trades),
            skipped_count=len(trades) - valid_count,
            lookback_count=len(trades),
        )

    valid_r_multiples = trades.loc[list(reversed(selected_indices)), "r_multiple"]

    rolling_exp = float(valid_r_multiples.mean())
    adjusted_score = rolling_exp - (
        STRATEGY_MODE_ADJUSTMENT_K * float(valid_r_multiples.std()) / math.sqrt(STRATEGY_MODE_WINDOW)
    )
    return RollingModeScores(
        rolling_exp=rolling_exp,
        adjusted_score=adjusted_score,
        valid_count=len(valid_r_multiples),
        examined_count=examined_count,
        skipped_count=skipped_count,
        lookback_count=len(trades),
    )


def parse_robinhood_csv(source: str | Path | TextIO | Any) -> ImportResult:
    rows = _read_csv_rows(source)
    if not rows:
        return ImportResult(_empty_transactions(), _empty_issues(), _empty_issues())

    header = rows[0]
    if header != ROBINHOOD_COLUMNS:
        return ImportResult(
            _empty_transactions(),
            _empty_issues(),
            pd.DataFrame(
                [{"row_number": 1, "issue": "Unexpected Robinhood CSV header.", "raw_row": repr(header)}],
                columns=IMPORT_ISSUE_COLUMNS,
            ),
        )

    transactions: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows[1:], start=2):
        if _is_robinhood_footer_row(row):
            continue

        if len(row) != len(ROBINHOOD_COLUMNS):
            malformed_rows.append(
                {
                    "row_number": row_number,
                    "issue": f"Expected {len(ROBINHOOD_COLUMNS)} columns, found {len(row)}.",
                    "raw_row": repr(row),
                }
            )
            continue

        record = dict(zip(ROBINHOOD_COLUMNS, row, strict=True))
        trans_code = str(record["Trans Code"]).strip()
        if trans_code not in TRADE_CODES:
            ignored_rows.append(
                {
                    "row_number": row_number,
                    "issue": f"Ignored transaction code: {trans_code or 'blank'}.",
                    "raw_row": repr(row),
                }
            )
            continue

        activity_date = pd.to_datetime(record["Activity Date"], errors="coerce")
        process_date = pd.to_datetime(record["Process Date"], errors="coerce")
        settle_date = pd.to_datetime(record["Settle Date"], errors="coerce")
        quantity = _parse_number(record["Quantity"])
        price = _parse_money(record["Price"])
        amount = _parse_money(record["Amount"])
        symbol = str(record["Instrument"]).upper().strip()

        if pd.isna(activity_date) or not symbol or quantity is None or price is None or amount is None:
            malformed_rows.append(
                {
                    "row_number": row_number,
                    "issue": "Missing or invalid trade field.",
                    "raw_row": repr(row),
                }
            )
            continue

        transactions.append(
            {
                "activity_date": activity_date.date().isoformat(),
                "process_date": "" if pd.isna(process_date) else process_date.date().isoformat(),
                "settle_date": "" if pd.isna(settle_date) else settle_date.date().isoformat(),
                "symbol": symbol,
                "description": record["Description"],
                "trans_code": trans_code,
                "quantity": quantity,
                "price": price,
                "amount": amount,
            }
        )

    return ImportResult(
        pd.DataFrame(transactions, columns=TRANSACTION_COLUMNS),
        pd.DataFrame(ignored_rows, columns=IMPORT_ISSUE_COLUMNS),
        pd.DataFrame(malformed_rows, columns=IMPORT_ISSUE_COLUMNS),
    )


def derive_fifo_trades(
    transactions: pd.DataFrame,
    planned_stops: pd.DataFrame | None = None,
    *,
    as_of: date | None = None,
) -> TradeDerivation:
    if transactions.empty:
        return TradeDerivation(
            _empty_closed_trades(),
            _empty_closed_trades(),
            _empty_open_lots(),
            _empty_unmatched_sells(),
            0,
            _empty_planned_stop_issues(),
            _empty_missing_planned_stop_rows(),
            _empty_lot_grouping_audit(),
        )

    planned_stop_lookup, planned_stop_issues = _planned_stop_lookup(planned_stops)
    normalized = _normalize_transactions(transactions)
    buy_lot_matches, lot_grouping_audit = _buy_lot_matches(normalized, planned_stops, planned_stop_lookup)
    lots: defaultdict[str, deque[dict[str, Any]]] = defaultdict(deque)
    trade_groups: dict[str, dict[str, Any]] = {}
    closed_trades: list[dict[str, Any]] = []
    exit_matches: list[dict[str, Any]] = []
    unmatched_sells: list[dict[str, Any]] = []

    for row_index, row in normalized.iterrows():
        symbol = row["symbol"]
        quantity = float(row["quantity"])
        if row["trans_code"] == "Buy":
            buy_date = row["activity_date"]
            lot_match = buy_lot_matches.get(
                row_index,
                {
                    "logical_trade_id": f"unmatched:{row_index}",
                    "logical_quantity": quantity,
                    "planned_stop": None,
                    "strategy": "",
                    "atr": None,
                    "market_regime": "",
                },
            )
            logical_trade_id = str(lot_match["logical_trade_id"])
            if logical_trade_id not in trade_groups:
                trade_groups[logical_trade_id] = {
                    "symbol": symbol,
                    "buy_date": buy_date,
                    "quantity": float(lot_match["logical_quantity"]),
                    "remaining_quantity": float(lot_match["logical_quantity"]),
                    "planned_stop": lot_match["planned_stop"],
                    "strategy": lot_match["strategy"],
                    "atr": lot_match["atr"],
                    "market_regime": lot_match["market_regime"],
                    "exit_matches": [],
                    "closed": False,
                }
            lots[symbol].append(
                {
                    "symbol": symbol,
                    "buy_date": buy_date,
                    "quantity": quantity,
                    "original_quantity": quantity,
                    "buy_price": float(row["price"]),
                    "logical_trade_id": logical_trade_id,
                    "exit_matches": [],
                    "planned_stop": lot_match["planned_stop"],
                    "strategy": lot_match["strategy"],
                    "atr": lot_match["atr"],
                    "market_regime": lot_match["market_regime"],
                }
            )
            continue

        remaining_quantity = quantity
        sell_price = float(row["price"])
        sell_date = row["activity_date"]
        while remaining_quantity > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            matched_quantity = min(remaining_quantity, lot["quantity"])
            buy_price = float(lot["buy_price"])
            buy_amount = round(matched_quantity * buy_price, 2)
            sell_amount = round(matched_quantity * sell_price, 2)
            realized_pnl = round(sell_amount - buy_amount, 2)
            realized_pnl_percent = round((realized_pnl / buy_amount) * 100, 2) if buy_amount else None

            exit_match = {
                "symbol": symbol,
                "buy_date": lot["buy_date"],
                "sell_date": sell_date,
                "quantity": _clean_quantity(matched_quantity),
                "planned_stop": lot["planned_stop"],
                "strategy": lot["strategy"],
                "atr": lot["atr"],
                "market_regime": lot["market_regime"],
                "buy_price": buy_price,
                "buy_amount": buy_amount,
                "sell_price": sell_price,
                "sell_amount": sell_amount,
                "realized_pnl": realized_pnl,
                "realized_pnl_percent": realized_pnl_percent,
                "hold_days": weekday_hold_count(lot["buy_date"], as_of=_to_date(sell_date)),
            }
            exit_matches.append(exit_match)
            lot["exit_matches"].append(exit_match)
            trade_group = trade_groups[lot["logical_trade_id"]]
            trade_group["exit_matches"].append(exit_match)
            trade_group["remaining_quantity"] -= matched_quantity

            lot["quantity"] -= matched_quantity
            remaining_quantity -= matched_quantity
            if trade_group["remaining_quantity"] <= 1e-9 and not trade_group["closed"]:
                closed_trades.append(_aggregate_closed_trade(trade_group))
                trade_group["closed"] = True
            if lot["quantity"] <= 1e-9:
                lots[symbol].popleft()

        if remaining_quantity > 1e-9:
            unmatched_sells.append(
                {
                    "symbol": symbol,
                    "sell_date": sell_date,
                    "quantity": _clean_quantity(remaining_quantity),
                    "sell_price": sell_price,
                }
            )

    open_lots = []
    for symbol in sorted(lots):
        for lot in lots[symbol]:
            if lot["quantity"] <= 1e-9:
                continue
            quantity = _clean_quantity(lot["quantity"])
            open_lots.append(
                {
                    "symbol": symbol,
                    "buy_date": lot["buy_date"],
                    "quantity": quantity,
                    "planned_stop": lot["planned_stop"],
                    "strategy": lot["strategy"],
                    "atr": lot["atr"],
                    "market_regime": lot["market_regime"],
                    "buy_price": float(lot["buy_price"]),
                    "cost_basis": round(float(lot["quantity"]) * float(lot["buy_price"]), 2),
                    "hold_days": weekday_hold_count(lot["buy_date"], as_of=as_of),
                }
            )

    closed_frame = pd.DataFrame(closed_trades, columns=CLOSED_TRADE_COLUMNS)
    exit_match_frame = pd.DataFrame(exit_matches, columns=CLOSED_TRADE_COLUMNS)
    open_frame = pd.DataFrame(open_lots, columns=OPEN_LOT_COLUMNS)
    unmatched_frame = pd.DataFrame(unmatched_sells, columns=UNMATCHED_SELL_COLUMNS)
    missing_planned_stops = _missing_stop_count(closed_frame) + _missing_stop_count(open_frame)
    missing_planned_stop_rows = _missing_planned_stop_rows(closed_frame, open_frame)

    return TradeDerivation(
        closed_frame,
        exit_match_frame,
        open_frame,
        unmatched_frame,
        missing_planned_stops,
        planned_stop_issues,
        missing_planned_stop_rows,
        lot_grouping_audit,
    )


def _aggregate_closed_trade(lot: dict[str, Any]) -> dict[str, Any]:
    matches = lot["exit_matches"]
    quantity = sum(float(match["quantity"]) for match in matches)
    buy_amount = round(sum(float(match["buy_amount"]) for match in matches), 2)
    sell_amount = round(sum(float(match["sell_amount"]) for match in matches), 2)
    realized_pnl = round(sell_amount - buy_amount, 2)
    realized_pnl_percent = round((realized_pnl / buy_amount) * 100, 2) if buy_amount else None
    buy_price = round(buy_amount / quantity, 2) if quantity else None
    sell_price = round(sell_amount / quantity, 2) if quantity else None
    sell_date = max(str(match["sell_date"]) for match in matches)

    return {
        "symbol": lot["symbol"],
        "buy_date": lot["buy_date"],
        "sell_date": sell_date,
        "quantity": _clean_quantity(quantity),
        "planned_stop": lot["planned_stop"],
        "strategy": lot["strategy"],
        "atr": lot["atr"],
        "market_regime": lot["market_regime"],
        "buy_price": buy_price,
        "buy_amount": buy_amount,
        "sell_price": sell_price,
        "sell_amount": sell_amount,
        "realized_pnl": realized_pnl,
        "realized_pnl_percent": realized_pnl_percent,
        "hold_days": weekday_hold_count(lot["buy_date"], as_of=_to_date(sell_date)),
    }


def _read_csv_rows(source: str | Path | TextIO | Any) -> list[list[str]]:
    if isinstance(source, str | Path):
        with Path(source).open(newline="", encoding="utf-8-sig") as file:
            return list(csv.reader(file))

    if hasattr(source, "getvalue"):
        value = source.getvalue()
        if isinstance(value, bytes):
            value = value.decode("utf-8-sig")
        return list(csv.reader(str(value).splitlines()))

    return list(csv.reader(source))


def _is_robinhood_footer_row(row: list[str]) -> bool:
    values = [str(value).strip() for value in row]
    if not any(values):
        return True

    text = " ".join(value for value in values if value).lower()
    return "the data provided is for informational purposes only" in text


def _normalize_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    normalized = transactions.copy()
    normalized["activity_date"] = pd.to_datetime(normalized["activity_date"], errors="coerce").dt.date.astype(str)
    normalized["symbol"] = normalized["symbol"].fillna("").astype(str).str.upper().str.strip()
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["side_rank"] = normalized["trans_code"].map({"Buy": 0, "Sell": 1}).fillna(9)
    return normalized.sort_values(["activity_date", "symbol", "side_rank"], kind="mergesort").reset_index(drop=True)


def normalize_strategy(value: Any) -> str:
    strategy = str(value or "").strip()
    return strategy if strategy in STRATEGY_OPTIONS else ""


def display_strategy(value: Any) -> str:
    strategy = normalize_strategy(value)
    return strategy or UNCLASSIFIED_STRATEGY


def display_trade_context_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "atr" in frame.columns:
        frame["atr"] = frame["atr"].apply(_display_optional_number)
    if "market_regime" in frame.columns:
        frame["market_regime"] = frame["market_regime"].apply(_display_optional_text)
    return frame


def _display_optional_number(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "N/A"
    return f"{float(number):.2f}"


def _display_optional_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    text = str(value).strip()
    return text if text and text.lower() not in {"none", "nan"} else "N/A"


def _buy_lot_matches(
    normalized: pd.DataFrame,
    planned_stops: pd.DataFrame | None,
    planned_stop_lookup: dict[tuple[str, str, float], dict[str, float | str | None]],
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    matches: dict[int, dict[str, Any]] = {}
    unmatched_buys: list[tuple[int, pd.Series]] = []

    buys = normalized[normalized["trans_code"] == "Buy"]
    for row_index, row in buys.iterrows():
        symbol = str(row["symbol"])
        buy_date = str(row["activity_date"])
        quantity = float(row["quantity"])
        exact_key = (symbol, buy_date, _quantity_key(quantity))
        exact_match = planned_stop_lookup.get(exact_key)
        if exact_match is not None:
            matches[row_index] = {
                "logical_trade_id": f"exact:{row_index}",
                "logical_quantity": quantity,
                **exact_match,
            }
            continue
        unmatched_buys.append((row_index, row))

    planned_entries = _planned_stop_entries(planned_stops)
    audit_rows: list[dict[str, Any]] = []
    grouped_buys: defaultdict[tuple[str, str, float], list[tuple[int, pd.Series]]] = defaultdict(list)
    for row_index, row in unmatched_buys:
        price = pd.to_numeric(row.get("price"), errors="coerce")
        if pd.isna(price):
            continue
        grouped_buys[(str(row["symbol"]), str(row["activity_date"]), _quantity_key(price))].append((row_index, row))

    for (symbol, buy_date, price), rows in grouped_buys.items():
        if len(rows) <= 1:
            continue
        total_quantity = sum(float(row["quantity"]) for _, row in rows)
        candidates = [
            entry
            for entry in planned_entries
            if entry["symbol"] == symbol
            and entry["buy_date"] == buy_date
            and _quantity_key(entry["quantity"]) == _quantity_key(total_quantity)
            and entry["planned_stop"] is not None
        ]
        if len(candidates) != 1:
            continue

        planned = candidates[0]
        logical_trade_id = f"split:{symbol}:{buy_date}:{price:g}:{_quantity_key(total_quantity):g}"
        for row_index, _ in rows:
            matches[row_index] = {
                "logical_trade_id": logical_trade_id,
                "logical_quantity": total_quantity,
                "planned_stop": planned["planned_stop"],
                "strategy": planned["strategy"],
                "atr": planned["atr"],
                "market_regime": planned["market_regime"],
            }
        split_quantities = ", ".join(str(_clean_quantity(float(row["quantity"]))) for _, row in rows)
        audit_rows.append(
            {
                "symbol": symbol,
                "buy_date": buy_date,
                "planned_quantity": _clean_quantity(total_quantity),
                "split_quantities": split_quantities,
                "buy_price": float(price),
                "planned_stop": planned["planned_stop"],
                "strategy": planned["strategy"],
                "atr": planned["atr"],
                "market_regime": planned["market_regime"],
                "reason": "Same symbol/date/price buy lots summed to one planned position.",
            }
        )

    return matches, pd.DataFrame(audit_rows, columns=LOT_GROUPING_AUDIT_COLUMNS)


def _planned_stop_entries(planned_stops: pd.DataFrame | None) -> list[dict[str, Any]]:
    if planned_stops is None or planned_stops.empty:
        return []

    entries = []
    for _, row in planned_stops.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        buy_date = pd.to_datetime(row.get("buy_date"), errors="coerce")
        quantity = pd.to_numeric(row.get("quantity"), errors="coerce")
        if pd.isna(buy_date) or pd.isna(quantity):
            continue
        planned_stop = pd.to_numeric(row.get("planned_stop"), errors="coerce")
        atr = pd.to_numeric(row.get("atr"), errors="coerce")
        entries.append(
            {
                "symbol": symbol,
                "buy_date": buy_date.date().isoformat(),
                "quantity": float(quantity),
                "planned_stop": None if pd.isna(planned_stop) else float(planned_stop),
                "strategy": normalize_strategy(row.get("strategy")),
                "atr": None if pd.isna(atr) else float(atr),
                "market_regime": _normalize_market_regime_or_blank(row.get("market_regime")),
            }
        )
    return entries


def _planned_stop_lookup(
    planned_stops: pd.DataFrame | None,
) -> tuple[dict[tuple[str, str, float], dict[str, float | str | None]], pd.DataFrame]:
    if planned_stops is None or planned_stops.empty:
        return {}, _empty_planned_stop_issues()

    stops_by_key: defaultdict[tuple[str, str, float], set[float]] = defaultdict(set)
    strategies_by_key: defaultdict[tuple[str, str, float], set[str]] = defaultdict(set)
    atrs_by_key: defaultdict[tuple[str, str, float], set[float]] = defaultdict(set)
    regimes_by_key: defaultdict[tuple[str, str, float], set[str]] = defaultdict(set)
    for _, row in planned_stops.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        buy_date = pd.to_datetime(row.get("buy_date"), errors="coerce")
        if pd.isna(buy_date):
            continue
        quantity = pd.to_numeric(row.get("quantity"), errors="coerce")
        if pd.isna(quantity):
            continue
        key = (symbol, buy_date.date().isoformat(), _quantity_key(quantity))

        planned_stop = pd.to_numeric(row.get("planned_stop"), errors="coerce")
        if not pd.isna(planned_stop):
            stops_by_key[key].add(float(planned_stop))

        strategy = normalize_strategy(row.get("strategy"))
        if strategy:
            strategies_by_key[key].add(strategy)
        atr = pd.to_numeric(row.get("atr"), errors="coerce")
        if not pd.isna(atr):
            atrs_by_key[key].add(float(atr))
        market_regime = _normalize_market_regime_or_blank(row.get("market_regime"))
        if market_regime:
            regimes_by_key[key].add(market_regime)

    lookup = {}
    issue_rows: list[dict[str, object]] = []
    for key in set(stops_by_key) | set(strategies_by_key) | set(atrs_by_key) | set(regimes_by_key):
        stops = stops_by_key[key]
        strategies = strategies_by_key[key]
        atrs = atrs_by_key[key]
        regimes = regimes_by_key[key]
        if len(stops) > 1:
            symbol, buy_date, quantity = key
            stop_values = ", ".join(f"{stop:g}" for stop in sorted(stops))
            issue_rows.append(
                {
                    "symbol": symbol,
                    "buy_date": buy_date,
                    "quantity": _clean_quantity(quantity),
                    "issue": "Conflicting planned stops for this lot key.",
                    "detail": f"planned_stop values: {stop_values}",
                }
            )
        lookup[key] = {
            "planned_stop": next(iter(stops)) if len(stops) == 1 else None,
            "strategy": next(iter(strategies)) if len(strategies) == 1 else "",
            "atr": next(iter(atrs)) if len(atrs) == 1 else None,
            "market_regime": next(iter(regimes)) if len(regimes) == 1 else "",
        }
    return lookup, pd.DataFrame(issue_rows, columns=PLANNED_STOP_ISSUE_COLUMNS)


def _normalize_market_regime_or_blank(value: Any) -> str:
    regime = str(value or "").strip().upper()
    return regime if regime in {"GO", "SELECTIVE GO", "NO-GO"} else ""


def _parse_money(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    text = re.sub(r"[^0-9.-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return None
    return float(text)


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    return float(text)


def _to_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _clean_quantity(value: float) -> float | int:
    if float(value).is_integer():
        return int(value)
    return round(float(value), 6)


def _quantity_key(value: Any) -> float:
    return round(float(value), 8)


def _missing_stop_count(frame: pd.DataFrame) -> int:
    if frame.empty or "planned_stop" not in frame.columns:
        return 0
    return int(frame["planned_stop"].isna().sum())


def _missing_planned_stop_rows(closed_frame: pd.DataFrame, open_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not closed_frame.empty and "planned_stop" in closed_frame.columns:
        for _, row in closed_frame[closed_frame["planned_stop"].isna()].iterrows():
            rows.append(
                {
                    "status": "Closed",
                    "symbol": row.get("symbol"),
                    "buy_date": row.get("buy_date"),
                    "sell_date": row.get("sell_date"),
                    "quantity": row.get("quantity"),
                    "buy_price": row.get("buy_price"),
                    "detail": "Closed trade is missing a usable planned stop.",
                }
            )
    if not open_frame.empty and "planned_stop" in open_frame.columns:
        for _, row in open_frame[open_frame["planned_stop"].isna()].iterrows():
            rows.append(
                {
                    "status": "Open",
                    "symbol": row.get("symbol"),
                    "buy_date": row.get("buy_date"),
                    "sell_date": "",
                    "quantity": row.get("quantity"),
                    "buy_price": row.get("buy_price"),
                    "detail": "Open lot is missing a usable planned stop.",
                }
            )
    return pd.DataFrame(rows, columns=MISSING_PLANNED_STOP_COLUMNS)


def _empty_trade_metrics() -> dict[str, float | int | None]:
    return {
        "trade_count": 0,
        "win_count": 0,
        "loss_count": 0,
        "breakeven_count": 0,
        "win_rate": None,
        "win_loss_ratio": None,
        "average_win_r": None,
        "average_loss_r": None,
        "r_ratio": None,
        "expectancy_r": None,
        "expectancy": None,
        "profit_factor": None,
        "average_win": None,
        "average_win_percent": None,
        "average_win_hold": None,
        "win_streak": 0,
        "top_win": None,
        "average_loss": None,
        "average_loss_percent": None,
        "average_loss_hold": None,
        "hold_time_ratio": None,
        "loss_streak": 0,
        "top_loss": None,
    }


def _mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    value = values.mean()
    if pd.isna(value):
        return None
    return float(value)


def _safe_round(value: Any, *, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _longest_streak(trades: pd.DataFrame, *, winning: bool) -> int:
    if trades.empty:
        return 0

    ordered = trades.copy()
    ordered["_original_order"] = range(len(ordered))
    ordered = ordered.sort_values(["sell_date", "_original_order"])
    best_streak = 0
    current_streak = 0

    for pnl in ordered["realized_pnl"]:
        is_match = pnl > 0 if winning else pnl < 0
        if is_match:
            current_streak += 1
        else:
            current_streak = 0
        best_streak = max(best_streak, current_streak)

    return best_streak


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def _empty_closed_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=CLOSED_TRADE_COLUMNS)


def _empty_open_lots() -> pd.DataFrame:
    return pd.DataFrame(columns=OPEN_LOT_COLUMNS)


def _empty_unmatched_sells() -> pd.DataFrame:
    return pd.DataFrame(columns=UNMATCHED_SELL_COLUMNS)


def _empty_planned_stop_issues() -> pd.DataFrame:
    return pd.DataFrame(columns=PLANNED_STOP_ISSUE_COLUMNS)


def _empty_missing_planned_stop_rows() -> pd.DataFrame:
    return pd.DataFrame(columns=MISSING_PLANNED_STOP_COLUMNS)


def _empty_lot_grouping_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=LOT_GROUPING_AUDIT_COLUMNS)


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame(columns=IMPORT_ISSUE_COLUMNS)
