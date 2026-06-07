from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


CONFIG_PATH = Path("config.toml")
DEFAULT_PORTFOLIO_AMOUNT = 20_000.0
DEFAULT_RISK_PERCENT = 0.5
DEFAULT_MARKET_REGIME = "GO"
DEFAULT_MAX_SYMBOL_EXPOSURE_PERCENT = 20.0
DEFAULT_ADD_ON_UNREALIZED_PROFIT_PRESERVE_PERCENT = 50.0
DEFAULT_IEX_SIZING_PRICE_BUFFER_PERCENT = 0.25
DEFAULT_IEX_SIZING_PRICE_BUFFER_MIN = 0.05
DEFAULT_IEX_SIZING_PRICE_BUFFER_MAX = 0.10


@dataclass(frozen=True)
class AppConfig:
    portfolio_amount: float = DEFAULT_PORTFOLIO_AMOUNT
    sizing_portfolio_amount: float = DEFAULT_PORTFOLIO_AMOUNT
    risk_percent: float = DEFAULT_RISK_PERCENT
    market_regime: str = DEFAULT_MARKET_REGIME
    max_symbol_exposure_percent: float = DEFAULT_MAX_SYMBOL_EXPOSURE_PERCENT
    add_on_unrealized_profit_preserve_percent: float = DEFAULT_ADD_ON_UNREALIZED_PROFIT_PRESERVE_PERCENT
    iex_sizing_price_buffer_percent: float = DEFAULT_IEX_SIZING_PRICE_BUFFER_PERCENT
    iex_sizing_price_buffer_min: float = DEFAULT_IEX_SIZING_PRICE_BUFFER_MIN
    iex_sizing_price_buffer_max: float = DEFAULT_IEX_SIZING_PRICE_BUFFER_MAX


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()

    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return AppConfig()

    defaults = parsed.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    portfolio_amount = _positive_float(defaults.get("portfolio_amount"), DEFAULT_PORTFOLIO_AMOUNT)
    sizing_portfolio_amount = _positive_float(defaults.get("sizing_portfolio_amount"), portfolio_amount)

    return AppConfig(
        portfolio_amount=portfolio_amount,
        sizing_portfolio_amount=sizing_portfolio_amount,
        risk_percent=_positive_float(defaults.get("risk_percent"), DEFAULT_RISK_PERCENT),
        market_regime=_market_regime(defaults.get("market_regime"), DEFAULT_MARKET_REGIME),
        max_symbol_exposure_percent=_positive_float(
            defaults.get("max_symbol_exposure_percent"),
            DEFAULT_MAX_SYMBOL_EXPOSURE_PERCENT,
        ),
        add_on_unrealized_profit_preserve_percent=_bounded_percent(
            defaults.get("add_on_unrealized_profit_preserve_percent"),
            DEFAULT_ADD_ON_UNREALIZED_PROFIT_PRESERVE_PERCENT,
        ),
        iex_sizing_price_buffer_percent=_positive_float(
            defaults.get("iex_sizing_price_buffer_percent"),
            DEFAULT_IEX_SIZING_PRICE_BUFFER_PERCENT,
        ),
        iex_sizing_price_buffer_min=_positive_float(
            defaults.get("iex_sizing_price_buffer_min"),
            DEFAULT_IEX_SIZING_PRICE_BUFFER_MIN,
        ),
        iex_sizing_price_buffer_max=_positive_float(
            defaults.get("iex_sizing_price_buffer_max"),
            DEFAULT_IEX_SIZING_PRICE_BUFFER_MAX,
        ),
    )


def _positive_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback

    if number <= 0:
        return fallback
    return number


def _market_regime(value: Any, fallback: str) -> str:
    regime = str(value or "").strip().upper()
    return regime if regime in {"GO", "SELECTIVE GO", "NO-GO"} else fallback


def _bounded_percent(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback

    if number < 0 or number > 100:
        return fallback
    return number
