from __future__ import annotations

import altair as alt
import pandas as pd


BAR_STEP_PIXELS = 68
MIN_BAR_STEP_PIXELS = 12
MAX_PLOT_WIDTH_PIXELS = 1400
CHART_HEIGHT_PIXELS = 560
BAR_PADDING_INNER = 0.4
# The plot is deliberately large, so the themed defaults need a matching step up to stay legible.
LABEL_FONT_SIZE = 13
TITLE_FONT_SIZE = 14
LOSS_LABEL = "Loss (< 0%)"
MIXED_LABEL = "Spans 0%"
GAIN_LABEL = "Gain (≥ 0%)"
OUTCOME_COLORS = {
    LOSS_LABEL: "#dc2626",
    MIXED_LABEL: "#7a7060",
    GAIN_LABEL: "#0d9488",
}


def build_trade_return_distribution_chart(bins: pd.DataFrame) -> alt.Chart:
    plot_bins = _plot_bins(bins)
    outcomes = [label for label in OUTCOME_COLORS if label in set(plot_bins["outcome"])]
    return (
        alt.Chart(plot_bins)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "axis_label:N",
                sort=None,
                title="Completed Trade Return (%)",
                scale=alt.Scale(paddingInner=BAR_PADDING_INNER),
                axis=alt.Axis(
                    labelAngle=0,
                    labelPadding=6,
                    labelOverlap="greedy",
                    labelFontSize=LABEL_FONT_SIZE,
                    titleFontSize=TITLE_FONT_SIZE,
                    titlePadding=12,
                ),
            ),
            y=alt.Y(
                "trade_count:Q",
                title="Completed Trades",
                axis=alt.Axis(
                    tickMinStep=1,
                    format="d",
                    labelFontSize=LABEL_FONT_SIZE,
                    titleFontSize=TITLE_FONT_SIZE,
                    titlePadding=12,
                ),
            ),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=outcomes, range=[OUTCOME_COLORS[label] for label in outcomes]),
                legend=(
                    alt.Legend(
                        title=None,
                        orient="top",
                        direction="horizontal",
                        labelFontSize=LABEL_FONT_SIZE,
                    )
                    if len(outcomes) > 1
                    else None
                ),
            ),
            tooltip=[
                alt.Tooltip("bin_label:N", title="Return Range"),
                alt.Tooltip("trade_count:Q", title="Trades", format="d"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(
            title="Distribution of Completed Trade Returns",
            width=alt.Step(_bar_step(len(plot_bins))),
            height=CHART_HEIGHT_PIXELS,
        )
    )


def _plot_bins(bins: pd.DataFrame) -> pd.DataFrame:
    total_trades = int(bins["trade_count"].sum())
    return bins.assign(
        axis_label=[_axis_label(row) for row in bins.itertuples()],
        outcome=[_outcome(row) for row in bins.itertuples()],
        share=bins["trade_count"] / total_trades if total_trades else 0.0,
    )


def _axis_label(row) -> str:
    """A tick reads as the bin's lower edge; end bars name their open-ended cutoff instead."""
    if row.overflow == "lower":
        return f"< {row.bin_end:g}"
    if row.overflow == "upper":
        return f"≥ {row.bin_start:g}"
    return f"{row.bin_start:g}"


def _outcome(row) -> str:
    """Classify a bar by the returns its range can hold, never by its stored bin edge alone.

    End bars are open-ended, so a cutoff on the far side of breakeven leaves the bar holding
    both winners and losers; colouring such a bar red or green would state an outcome the
    data does not support.
    """
    if row.overflow == "lower":
        return LOSS_LABEL if row.bin_end <= 0 else MIXED_LABEL
    if row.overflow == "upper":
        return GAIN_LABEL if row.bin_start >= 0 else MIXED_LABEL
    return GAIN_LABEL if row.bin_start >= 0 else LOSS_LABEL


def _bar_step(bin_count: int) -> int:
    """Keep the plot near its natural width; squeeze the band only when a range runs long.

    Ranges stay short once outlier fencing applies, but a sample too small to fence spreads
    raw extremes across many bins, which at the natural step would overflow the panel.
    """
    if bin_count <= 0:
        return BAR_STEP_PIXELS
    return max(min(BAR_STEP_PIXELS, MAX_PLOT_WIDTH_PIXELS // bin_count), MIN_BAR_STEP_PIXELS)
