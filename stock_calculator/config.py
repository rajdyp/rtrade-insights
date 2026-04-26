from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


CONFIG_PATH = Path("config.toml")
DEFAULT_PORTFOLIO_AMOUNT = 20_000.0
DEFAULT_RISK_PERCENT = 0.5


@dataclass(frozen=True)
class AppConfig:
    portfolio_amount: float = DEFAULT_PORTFOLIO_AMOUNT
    sizing_portfolio_amount: float = DEFAULT_PORTFOLIO_AMOUNT
    risk_percent: float = DEFAULT_RISK_PERCENT


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
    )


def _positive_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback

    if number <= 0:
        return fallback
    return number
