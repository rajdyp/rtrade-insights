from __future__ import annotations

from typing import Any

import pandas as pd

from stock_calculator.robinhood import STRATEGY_OPTIONS


MARKET_REGIME_OPTIONS = ["GO", "SELECTIVE GO", "NO-GO"]
STRATEGY_MODE_UNKNOWN = "Unknown"

RISK_PERCENT_MATRIX = {
    "GO": {
        "Working": 1.00,
        "Caution": 0.50,
        "Weak": 0.25,
        "Failing": 0.12,
        "Unknown": 0.12,
    },
    "SELECTIVE GO": {
        "Working": 0.50,
        "Caution": 0.25,
        "Weak": 0.12,
        "Failing": 0.06,
        "Unknown": 0.06,
    },
    "NO-GO": {
        "Working": 0.25,
        "Caution": 0.12,
        "Weak": 0.06,
        "Failing": 0.00,
        "Unknown": 0.00,
    },
}


def normalize_market_regime(value: Any, fallback: str = "GO") -> str:
    regime = str(value or "").strip().upper()
    if regime in MARKET_REGIME_OPTIONS:
        return regime
    return fallback if fallback in MARKET_REGIME_OPTIONS else "GO"


def strategy_mode_for_selection(strategy_metrics: pd.DataFrame | None, strategy: str) -> str:
    if strategy_metrics is None or strategy_metrics.empty:
        return STRATEGY_MODE_UNKNOWN

    strategy_name = str(strategy or "").strip()
    rows = strategy_metrics[strategy_metrics["strategy"] == strategy_name] if "strategy" in strategy_metrics else pd.DataFrame()
    if rows.empty or "mode" not in rows:
        return STRATEGY_MODE_UNKNOWN

    mode = str(rows.iloc[0]["mode"] or "").strip()
    if mode in RISK_PERCENT_MATRIX["GO"]:
        return mode
    return STRATEGY_MODE_UNKNOWN


def suggested_risk_percent(market_regime: str, strategy_mode: str, fallback: float) -> float:
    regime = normalize_market_regime(market_regime)
    mode = strategy_mode if strategy_mode in RISK_PERCENT_MATRIX[regime] else STRATEGY_MODE_UNKNOWN
    try:
        return RISK_PERCENT_MATRIX[regime][mode]
    except KeyError:
        return fallback


def default_strategy() -> str:
    return STRATEGY_OPTIONS[0]
