import pandas as pd

from stock_calculator.robinhood import STRATEGY_OPTIONS
from stock_calculator.strategy_attribution_display import (
    strategy_attribution_display_strategies,
    strategy_attribution_strategy_frame,
)


def frame_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return frame.to_dict("records")


def test_strategy_attribution_display_strategies_keeps_configured_order_and_appends_new_strategy():
    attribution = pd.DataFrame([{"strategy": "BO"}, {"strategy": "New Strategy"}])

    assert strategy_attribution_display_strategies(attribution) == [*STRATEGY_OPTIONS, "New Strategy"]


def test_strategy_attribution_strategy_frame_reports_missing_configured_strategy():
    attribution = pd.DataFrame([{"strategy": "EP", "mode": "Working"}])

    frame = strategy_attribution_strategy_frame(attribution, "4% BO")

    assert frame.columns.tolist() == ["Metric", "4% BO"]
    assert frame_records(frame) == [{"Metric": "Status", "4% BO": "No attribution yet"}]


def test_strategy_attribution_strategy_frame_keeps_warmup_rows_when_attribution_exists():
    attribution = pd.DataFrame(
        [
            {
                "strategy": "EP",
                "mode": "Unknown",
                "mode_basis": "Need 15 valid R trades",
                "trend": "Need 15 trades",
                "trend_driver": "Need 15 trades",
                "playbook": "Insufficient valid R-trade history; maintain minimum sizing until 15 valid R trades are available.",
                "evidence": "14 closed trades | Exp R 0.40 | directional only until 15 valid R trades",
            }
        ]
    )

    frame = strategy_attribution_strategy_frame(attribution, "EP")

    assert frame.columns.tolist() == ["Metric", "EP"]
    assert frame_records(frame) == [
        {"Metric": "Mode", "EP": "Unknown"},
        {"Metric": "Mode Basis", "EP": "Need 15 valid R trades"},
        {"Metric": "Trend", "EP": "Need 15 trades"},
        {"Metric": "Trend Driver", "EP": "Need 15 trades"},
        {"Metric": "Playbook", "EP": "Not enough R history; stay minimum size until 15 valid R trades."},
        {"Metric": "Performance", "EP": "Exp R: 0.40"},
        {"Metric": "Confidence", "EP": "14 closed trades  |  Directional only until 15 valid R trades"},
    ]


def test_strategy_attribution_strategy_frame_explains_trend_driver_from_evidence():
    attribution = pd.DataFrame(
        [
            {
                "strategy": "BO",
                "mode": "Weak",
                "mode_basis": "15R -0.04R | Adj -0.16R",
                "trend": "Recovering (+0.44R)",
                "trend_driver": "Loss control + Winner size + Hit rate + Profit factor + Expectancy R + Hold time",
                "playbook": (
                    "Require cleaner reward/risk and avoid taking profits too quickly. "
                    "Increase entry selectivity and reduce marginal setups."
                ),
                "evidence": (
                    "PF 0.68 vs 0.22 | Exp R -0.04 vs -0.48 | Win rate 46.7% vs 33.3% (+13.4 pts) | "
                    "Avg win R 1.00 vs 0.60 (+0.40) | Avg loss R -0.50 vs -0.65 (+0.15) | "
                    "Risk/ATR 1.40 vs 1.10 (+0.30) | "
                    "Hold ratio 0.27 vs 0.42 (-0.15) | Recent: 15 trades over 91 days | "
                    "Prior: 15 trades over 14 days | Regime attribution: need 6 more tagged trades"
                ),
            }
        ]
    )

    frame = strategy_attribution_strategy_frame(attribution, "BO")
    assert frame.columns.tolist() == ["Metric", "BO"]
    values = {row["Metric"]: row["BO"] for row in frame_records(frame)}

    assert values["Trend Driver"] == (
        "Loss control (Avg loss R: +0.15) + Winner size (Avg win R: +0.40) + "
        "Hit rate (Win rate: +13.4 pts) + PF (PF: +0.46) + "
        "Expectancy R (Exp R: +0.44) + Hold time (Hold ratio: -0.15)"
    )
    assert values["Mode Basis"] == "15R: -0.04R  |  Adj: -0.16R"
    assert values["Playbook"] == (
        "Require cleaner risk/reward; let winners work.  |  Be more selective; skip marginal setups."
    )
    assert values["Performance"] == (
        "Exp R: -0.04 vs -0.48 (+0.44)  |  PF: 0.68 vs 0.22 (+0.46)  |  "
        "Win rate: 46.7% vs 33.3% (+13.4 pts)"
    )
    assert values["Risk Behavior"] == (
        "Avg win R: 1.00 vs 0.60 (+0.40)  |  Avg loss R: -0.50 vs -0.65 (+0.15)  |  "
        "Risk/ATR: 1.40 vs 1.10 (+0.30)  |  Hold ratio: 0.27 vs 0.42 (-0.15)"
    )
    assert values["Confidence"] == (
        "Recent: 15 trades over 91 days  |  Prior: 15 trades over 14 days  |  "
        "Regime: Need 6 more tagged trades"
    )


def test_strategy_attribution_strategy_frame_keeps_single_value_risk_atr_without_no_prior_note():
    attribution = pd.DataFrame(
        [
            {
                "strategy": "EP",
                "mode": "Working",
                "mode_basis": "15R +0.40R | Adj +0.35R",
                "trend": "Need 30 trades",
                "trend_driver": "Current strengths: Winner size",
                "playbook": "Keep using Market Regime and Strategy Mode sizing while monitoring the listed drivers.",
                "evidence": "PF 1.30 | Exp R 0.40 | Avg win R 2.27 | Risk/ATR 1.19",
            }
        ]
    )

    frame = strategy_attribution_strategy_frame(attribution, "EP")
    values = {row["Metric"]: row["EP"] for row in frame_records(frame)}

    assert values["Mode Basis"] == "15R: +0.40R  |  Adj: +0.35R"
    assert values["Performance"] == "Exp R: 0.40  |  PF: 1.30"
    assert values["Risk Behavior"] == "Avg win R: 2.27  |  Risk/ATR: 1.19"
    assert "(no prior)" not in values["Risk Behavior"]


def test_strategy_attribution_strategy_frame_sentence_cases_regime_context():
    attribution = pd.DataFrame(
        [
            {
                "strategy": "EP",
                "mode": "Caution",
                "mode_basis": "15R +0.10R | Adj +0.08R",
                "trend": "Flat (+0.00R)",
                "trend_driver": "No clear driver",
                "playbook": "Keep using Market Regime and Strategy Mode sizing while monitoring the listed drivers.",
                "evidence": "Regime attribution: need at least 3 tagged trades in 2 regimes",
            }
        ]
    )

    frame = strategy_attribution_strategy_frame(attribution, "EP")
    values = {row["Metric"]: row["EP"] for row in frame_records(frame)}

    assert values["Confidence"] == "Regime: Need at least 3 tagged trades in 2 regimes"
