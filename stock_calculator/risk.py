from __future__ import annotations

import pandas as pd

from stock_calculator.robinhood import STRATEGY_OPTIONS
from stock_calculator.sizing_policy import (
    EXPOSURE_RECOMMENDATION_MATRIX,
    MARKET_REGIME_OPTIONS,
    STRATEGY_MODE_UNKNOWN,
    STRATEGY_MODES,
    normalize_market_regime,
    suggested_exposure,
)


def strategy_mode_for_selection(strategy_metrics: pd.DataFrame | None, strategy: str) -> str:
    if strategy_metrics is None or strategy_metrics.empty:
        return STRATEGY_MODE_UNKNOWN

    strategy_name = str(strategy or "").strip()
    rows = strategy_metrics[strategy_metrics["strategy"] == strategy_name] if "strategy" in strategy_metrics else pd.DataFrame()
    if rows.empty or "mode" not in rows:
        return STRATEGY_MODE_UNKNOWN

    mode = str(rows.iloc[0]["mode"] or "").strip()
    if mode in STRATEGY_MODES:
        return mode
    return STRATEGY_MODE_UNKNOWN


def default_strategy() -> str:
    return STRATEGY_OPTIONS[0]
