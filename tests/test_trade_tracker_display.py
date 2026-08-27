import pandas as pd

from stock_calculator.trade_tracker_display import (
    BAR_STEP_PIXELS,
    CHART_HEIGHT_PIXELS,
    GAIN_LABEL,
    LOSS_LABEL,
    MAX_PLOT_WIDTH_PIXELS,
    MIN_BAR_STEP_PIXELS,
    MIXED_LABEL,
    OUTCOME_COLORS,
    build_trade_return_distribution_chart,
)


def _bins(rows):
    return pd.DataFrame(rows, columns=["bin_start", "bin_end", "bin_label", "trade_count", "overflow"])


def _core_bin(bin_start, trade_count):
    bin_end = bin_start + 2.0
    return {
        "bin_start": bin_start,
        "bin_end": bin_end,
        "bin_label": f"[{bin_start:g}%, {bin_end:g}%)",
        "trade_count": trade_count,
        "overflow": None,
    }


def test_trade_return_distribution_chart_renders_a_single_layer_with_visible_axes():
    spec = build_trade_return_distribution_chart(_bins([_core_bin(-2.0, 2), _core_bin(0.0, 3)])).to_dict()

    # The layered predecessor declared axis=None on the rule and annotation layers, which
    # suppressed both shared axes and left the chart with no ticks, gridlines, or titles.
    assert "layer" not in spec
    assert spec["mark"] == {"type": "bar", "cornerRadiusEnd": 4}
    assert spec["encoding"]["x"]["axis"] is not None
    assert spec["encoding"]["y"]["axis"] is not None
    assert spec["encoding"]["x"]["title"] == "Completed Trade Return (%)"
    assert spec["encoding"]["y"]["title"] == "Completed Trades"
    # Counts are whole trades, so the axis must not invent fractional ticks.
    assert spec["encoding"]["y"]["axis"]["format"] == "d"
    assert spec["encoding"]["y"]["axis"]["tickMinStep"] == 1
    assert spec["height"] == CHART_HEIGHT_PIXELS


def test_trade_return_distribution_chart_keeps_bin_order_and_compact_width():
    bins = _bins([_core_bin(-4.0, 1), _core_bin(-2.0, 0), _core_bin(0.0, 5)])

    spec = build_trade_return_distribution_chart(bins).to_dict()
    dataset = spec["datasets"][spec["data"]["name"]]

    # A nominal domain sorts alphabetically unless sorting is switched off, which would
    # scramble "-4", "-2", "0" into the wrong order along the axis.
    assert spec["encoding"]["x"]["field"] == "axis_label"
    assert spec["encoding"]["x"]["sort"] is None
    assert [row["axis_label"] for row in dataset] == ["-4", "-2", "0"]
    assert spec["width"] == {"step": BAR_STEP_PIXELS}


def test_trade_return_distribution_chart_labels_and_colors_overflow_bars():
    bins = _bins(
        [
            {
                "bin_start": -14.0,
                "bin_end": -12.0,
                "bin_label": "< -12%",
                "trade_count": 1,
                "overflow": "lower",
            },
            _core_bin(-12.0, 0),
            _core_bin(0.0, 26),
            {
                "bin_start": 10.0,
                "bin_end": 12.0,
                "bin_label": "≥ 10%",
                "trade_count": 16,
                "overflow": "upper",
            },
        ]
    )

    spec = build_trade_return_distribution_chart(bins).to_dict()
    dataset = spec["datasets"][spec["data"]["name"]]

    assert [row["axis_label"] for row in dataset] == ["< -12", "-12", "0", "≥ 10"]
    assert [row["outcome"] for row in dataset] == [LOSS_LABEL, LOSS_LABEL, GAIN_LABEL, GAIN_LABEL]
    assert spec["encoding"]["color"]["scale"] == {
        "domain": [LOSS_LABEL, GAIN_LABEL],
        "range": [OUTCOME_COLORS[LOSS_LABEL], OUTCOME_COLORS[GAIN_LABEL]],
    }
    assert spec["encoding"]["color"]["legend"]["orient"] == "top"


def test_trade_return_distribution_chart_marks_an_end_bar_that_spans_breakeven():
    # An outlier fence can land above breakeven after a run of large winners, leaving the
    # lower end bar holding both losers and small winners.
    bins = _bins(
        [
            {
                "bin_start": 14.0,
                "bin_end": 16.0,
                "bin_label": "< 16%",
                "trade_count": 3,
                "overflow": "lower",
            },
            _core_bin(16.0, 5),
        ]
    )

    spec = build_trade_return_distribution_chart(bins).to_dict()
    dataset = spec["datasets"][spec["data"]["name"]]

    assert [row["outcome"] for row in dataset] == [MIXED_LABEL, GAIN_LABEL]
    assert spec["encoding"]["color"]["scale"]["range"] == [
        OUTCOME_COLORS[MIXED_LABEL],
        OUTCOME_COLORS[GAIN_LABEL],
    ]


def test_trade_return_distribution_chart_drops_the_legend_for_a_single_outcome():
    spec = build_trade_return_distribution_chart(_bins([_core_bin(0.0, 4), _core_bin(2.0, 1)])).to_dict()

    assert spec["encoding"]["color"]["scale"]["domain"] == [GAIN_LABEL]
    assert spec["encoding"]["color"]["legend"] is None


def test_trade_return_distribution_chart_reports_trade_share_in_the_tooltip():
    spec = build_trade_return_distribution_chart(_bins([_core_bin(-2.0, 1), _core_bin(0.0, 3)])).to_dict()
    dataset = spec["datasets"][spec["data"]["name"]]

    assert [tooltip["field"] for tooltip in spec["encoding"]["tooltip"]] == [
        "bin_label",
        "trade_count",
        "share",
    ]
    assert [row["share"] for row in dataset] == [0.25, 0.75]


def test_trade_return_distribution_chart_narrows_the_band_for_a_long_unfenced_range():
    # A sample too small to fence outliers spreads raw extremes across many bins; at the
    # natural step the plot would run past the panel it sits in.
    wide_bins = _bins([_core_bin(-72.0 + index * 2.0, 0) for index in range(38)])

    spec = build_trade_return_distribution_chart(wide_bins).to_dict()
    step = spec["width"]["step"]

    assert MIN_BAR_STEP_PIXELS <= step < BAR_STEP_PIXELS
    assert step * 38 <= MAX_PLOT_WIDTH_PIXELS
