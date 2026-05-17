from __future__ import annotations

import re
from typing import Any

import pandas as pd

from stock_calculator.robinhood import STRATEGY_OPTIONS


STRATEGY_ATTRIBUTION_DISPLAY_ROWS = [
    ("Mode", "mode"),
    ("Mode Basis", "mode_basis"),
    ("Trend", "trend"),
    ("Trend Driver", "trend_driver"),
    ("Playbook", "playbook"),
]

STRATEGY_ATTRIBUTION_EVIDENCE_GROUPS = [
    "Performance",
    "Risk Behavior",
    "Confidence",
]

STRATEGY_ATTRIBUTION_EVIDENCE_GROUP_PARTS = {
    "Performance": ["Exp R", "PF", "Win rate"],
    "Risk Behavior": ["Avg win R", "Avg loss R", "Risk/ATR", "Hold ratio"],
    "Confidence": ["Recent", "Prior", "Regime", "Trades", "Status", "Other"],
}

DRIVER_EVIDENCE_LABELS = {
    "Loss control": "Avg loss R",
    "Winner size": "Avg win R",
    "Hit rate": "Win rate",
    "Profit factor": "PF",
    "PF": "PF",
    "Expectancy R": "Exp R",
    "Hold time": "Hold ratio",
}
DISPLAY_SEPARATOR = "  |  "
PLAYBOOK_ACTION_MESSAGES = [
    "Losses are being held significantly longer than wins. Enforce time-based or rule-based exits on losing positions.",
    "Tighten stops; avoid trades where risk can spread.",
    "Require cleaner risk/reward; let winners work.",
    "Be more selective; skip marginal setups.",
    "Probe only or pause until the setup stabilizes.",
]


def strategy_attribution_display_strategies(strategy_attribution: pd.DataFrame | None) -> list[str]:
    strategies = list(STRATEGY_OPTIONS)
    if strategy_attribution is None or strategy_attribution.empty:
        return strategies

    existing = [str(strategy) for strategy in strategy_attribution.get("strategy", [])]
    for strategy in existing:
        if strategy and strategy not in strategies:
            strategies.append(strategy)
    return strategies


def strategy_attribution_strategy_frame(strategy_attribution: pd.DataFrame | None, strategy: str) -> pd.DataFrame:
    source_row = strategy_attribution_rows_by_strategy(strategy_attribution).get(strategy)
    if source_row is None:
        return pd.DataFrame([{"Metric": "Status", strategy: "No attribution yet"}], columns=["Metric", strategy])

    rows = []
    for label, key in STRATEGY_ATTRIBUTION_DISPLAY_ROWS:
        rows.append(
            {
                "Metric": label,
                strategy: strategy_attribution_display_value(source_row.get(key), key=key, source_row=source_row),
            }
        )

    evidence_values = strategy_attribution_evidence_values(source_row.get("evidence"))
    for group in STRATEGY_ATTRIBUTION_EVIDENCE_GROUPS:
        value = evidence_values.get(group)
        if value:
            rows.append({"Metric": group, strategy: value})

    return pd.DataFrame(rows, columns=["Metric", strategy])


def strategy_attribution_rows_by_strategy(strategy_attribution: pd.DataFrame | None) -> dict[str, pd.Series]:
    if strategy_attribution is None or strategy_attribution.empty:
        return {}
    return {
        str(row.get("strategy") or ""): row
        for _, row in strategy_attribution.iterrows()
    }


def strategy_attribution_display_value(value: object, *, key: str, source_row: pd.Series | None = None) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if key == "mode_basis":
        return display_separator_text(text)
    if key == "trend_driver":
        return explained_trend_driver(text, source_row)
    if key == "playbook":
        return compact_strategy_attribution_playbook(text)
    return text


def display_separator_text(text: str) -> str:
    if "|" in text:
        return DISPLAY_SEPARATOR.join(mode_basis_part_text(part.strip()) for part in text.split("|"))
    return DISPLAY_SEPARATOR.join(part.strip() for part in text.split("|"))


def mode_basis_part_text(part: str) -> str:
    for label in ["15R", "Adj"]:
        prefix = f"{label} "
        if part.startswith(prefix):
            return f"{label}: {part.removeprefix(prefix).strip()}"
    return part


def compact_strategy_attribution_playbook(text: str) -> str:
    replacements = {
        "Trend is recovering. Watch for Working status before resuming normal sizing.": (
            "Recovering; wait for Working before sizing back up."
        ),
        "Insufficient valid R-trade history; maintain minimum sizing until 15 valid R trades are available.": (
            "Not enough R history; stay minimum size until 15 valid R trades."
        ),
        "Require cleaner reward/risk and avoid taking profits too quickly.": (
            "Require cleaner risk/reward; let winners work."
        ),
        "Increase entry selectivity and reduce marginal setups.": (
            "Be more selective; skip marginal setups."
        ),
        "Tighten stop discipline and avoid entries where risk can expand.": (
            "Tighten stops; avoid trades where risk can spread."
        ),
    }
    for original, compact in replacements.items():
        text = text.replace(original, compact)
    return separate_playbook_actions(text)


def separate_playbook_actions(text: str) -> str:
    actions = []
    remaining = text.strip()
    while remaining:
        matched_action = next((action for action in PLAYBOOK_ACTION_MESSAGES if remaining.startswith(action)), None)
        if matched_action is None:
            return text
        actions.append(matched_action)
        remaining = remaining.removeprefix(matched_action).strip()
    return DISPLAY_SEPARATOR.join(actions)


def explained_trend_driver(trend_driver: str, source_row: pd.Series | None) -> str:
    evidence_by_label = trend_driver_evidence_by_label(source_row)
    prefix = ""
    driver_text = trend_driver
    for candidate_prefix in ["Contributors:", "Current strengths:"]:
        if driver_text.startswith(candidate_prefix):
            prefix = f"{candidate_prefix} "
            driver_text = driver_text.removeprefix(candidate_prefix).strip()
            break

    parts = [part.strip() for part in driver_text.split("+") if part.strip()]
    if not parts:
        return compact_trend_driver_label(trend_driver)

    explained_parts = []
    for part in parts:
        display_label = compact_trend_driver_label(part)
        evidence_label = DRIVER_EVIDENCE_LABELS.get(part, DRIVER_EVIDENCE_LABELS.get(display_label))
        evidence = evidence_by_label.get(evidence_label or "")
        explained_parts.append(f"{display_label} ({evidence})" if evidence else display_label)
    return f"{prefix}{' + '.join(explained_parts)}"


def compact_trend_driver_label(text: str) -> str:
    return text.replace("Profit factor", "PF").replace("profit factor", "PF")


def trend_driver_evidence_by_label(source_row: pd.Series | None) -> dict[str, str]:
    if source_row is None:
        return {}
    evidence = source_row.get("evidence")
    values = {}
    for item in strategy_attribution_evidence_items(evidence):
        _, label, text = strategy_attribution_evidence_part(item)
        summary = trend_driver_evidence_summary(label, text)
        if summary:
            values[label] = summary
    return values


def trend_driver_evidence_summary(label: str, text: str) -> str:
    if label in {"Avg win R", "Avg loss R", "Exp R", "PF"}:
        delta = comparison_delta(text, label)
        if delta is not None:
            return f"{label}: {delta:+.2f}"
        return text
    if label == "Win rate":
        delta = parenthesized_delta_text(text)
        if delta:
            return f"{label}: {delta}"
        return text
    if label == "Hold ratio":
        delta = parenthesized_delta_text(text)
        if delta:
            return f"{label}: {delta}"
        return text
    return text


def comparison_delta(text: str, label: str) -> float | None:
    match = re.match(rf"{re.escape(label)}:?\s+([+-]?\d+(?:\.\d+)?)\s+vs\s+([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return round(float(match.group(1)) - float(match.group(2)), 2)


def parenthesized_delta_text(text: str) -> str:
    parenthesized_delta = re.search(r"\(([+-]\d+(?:\.\d+)?(?: pts)?)\)$", text)
    return parenthesized_delta.group(1) if parenthesized_delta else ""


def strategy_attribution_evidence_items(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(" | ") if part.strip()]


def strategy_attribution_evidence_part(part: str) -> tuple[str, str, str]:
    sample_labels = ["Recent", "Prior"]
    for source_label in sample_labels:
        prefix = f"{source_label}:"
        if part.startswith(prefix):
            return (
                "Confidence",
                source_label,
                compact_sample_window_text(source_label, part.removeprefix(prefix).strip()),
            )

    regime_prefix = "Regime attribution:"
    if part.startswith(regime_prefix):
        return "Confidence", "Regime", compact_regime_context_text(part.removeprefix(regime_prefix).strip())

    performance_labels = ["Exp R", "PF", "Win rate"]
    for label in performance_labels:
        prefix = f"{label} "
        if part.startswith(prefix):
            text = comparison_text_with_delta(part, label) if label in {"Exp R", "PF"} else label_value_text(part, label)
            return "Performance", label, text

    risk_labels = ["Avg win R", "Avg loss R", "Risk/ATR", "Hold ratio"]
    for label in risk_labels:
        prefix = f"{label} "
        if part.startswith(prefix):
            return "Risk Behavior", label, label_value_text(part, label)

    if part.endswith("closed trades"):
        return "Confidence", "Trades", part
    if part.startswith("directional only"):
        return "Confidence", "Status", sentence_case_text(part)
    return "Confidence", "Other", part


def compact_sample_window_text(label: str, value: str) -> str:
    return f"{label}: {value}"


def compact_regime_context_text(value: str) -> str:
    return f"Regime: {sentence_case_text(value)}"


def sentence_case_text(value: str) -> str:
    text = value.strip()
    return text[:1].upper() + text[1:] if text else text


def strategy_attribution_evidence_values(value: object) -> dict[str, str]:
    items = strategy_attribution_evidence_items(value)
    if not items:
        return {"Confidence": "No evidence available."}

    grouped_items = {
        group: {part: [] for part in STRATEGY_ATTRIBUTION_EVIDENCE_GROUP_PARTS[group]}
        for group in STRATEGY_ATTRIBUTION_EVIDENCE_GROUPS
    }
    for item in items:
        group, part_label, text = strategy_attribution_evidence_part(item)
        grouped_items[group][part_label].append(text)

    values = {}
    for group in STRATEGY_ATTRIBUTION_EVIDENCE_GROUPS:
        parts = []
        for part_label in STRATEGY_ATTRIBUTION_EVIDENCE_GROUP_PARTS[group]:
            parts.extend(grouped_items[group][part_label])
        if parts:
            values[group] = DISPLAY_SEPARATOR.join(parts)
    return values


def comparison_text_with_delta(text: str, label: str) -> str:
    if re.search(r"\([+-]\d+(?:\.\d+)?\)$", text):
        return label_value_text(text, label)
    delta = comparison_delta(text, label)
    if delta is None:
        return label_value_text(text, label)
    return label_value_text(f"{text} ({delta:+.2f})", label)


def label_value_text(text: str, label: str) -> str:
    prefix = f"{label} "
    if text.startswith(prefix):
        return f"{label}: {text.removeprefix(prefix).strip()}"
    return text
