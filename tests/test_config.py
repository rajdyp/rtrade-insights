from stock_calculator.config import AppConfig, load_config


def test_load_config_uses_defaults_when_file_is_missing(tmp_path):
    config = load_config(tmp_path / "missing.toml")

    assert config == AppConfig(
        portfolio_amount=20_000.0,
        sizing_portfolio_amount=20_000.0,
        risk_percent=0.5,
        market_regime="GO",
        max_symbol_exposure_percent=20.0,
        iex_sizing_price_buffer_percent=0.25,
        iex_sizing_price_buffer_min=0.05,
        iex_sizing_price_buffer_max=0.10,
    )


def test_load_config_reads_valid_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[defaults]
portfolio_amount = 50000.0
sizing_portfolio_amount = 40000.0
risk_percent = 1.25
market_regime = "SELECTIVE GO"
max_symbol_exposure_percent = 15.5
iex_sizing_price_buffer_percent = 0.75
iex_sizing_price_buffer_min = 0.03
iex_sizing_price_buffer_max = 0.20
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.portfolio_amount == 50_000.0
    assert config.sizing_portfolio_amount == 40_000.0
    assert config.risk_percent == 1.25
    assert config.market_regime == "SELECTIVE GO"
    assert config.max_symbol_exposure_percent == 15.5
    assert config.iex_sizing_price_buffer_percent == 0.75
    assert config.iex_sizing_price_buffer_min == 0.03
    assert config.iex_sizing_price_buffer_max == 0.20


def test_load_config_defaults_sizing_portfolio_to_baseline_portfolio(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[defaults]
portfolio_amount = 50000.0
risk_percent = 1.25
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.portfolio_amount == 50_000.0
    assert config.sizing_portfolio_amount == 50_000.0
    assert config.risk_percent == 1.25


def test_load_config_falls_back_per_invalid_field(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[defaults]
portfolio_amount = -1
sizing_portfolio_amount = -2
risk_percent = 0.75
max_symbol_exposure_percent = -4
iex_sizing_price_buffer_percent = -0.5
iex_sizing_price_buffer_min = 0
iex_sizing_price_buffer_max = -0.25
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.portfolio_amount == 20_000.0
    assert config.sizing_portfolio_amount == 20_000.0
    assert config.risk_percent == 0.75
    assert config.max_symbol_exposure_percent == 20.0
    assert config.iex_sizing_price_buffer_percent == 0.25
    assert config.iex_sizing_price_buffer_min == 0.05
    assert config.iex_sizing_price_buffer_max == 0.10


def test_load_config_falls_back_invalid_sizing_portfolio_to_baseline_portfolio(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[defaults]
portfolio_amount = 50000.0
sizing_portfolio_amount = -2
risk_percent = 0.75
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.portfolio_amount == 50_000.0
    assert config.sizing_portfolio_amount == 50_000.0
    assert config.risk_percent == 0.75


def test_load_config_falls_back_for_invalid_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[defaults", encoding="utf-8")

    config = load_config(path)

    assert config == AppConfig(
        portfolio_amount=20_000.0,
        sizing_portfolio_amount=20_000.0,
        risk_percent=0.5,
        market_regime="GO",
        max_symbol_exposure_percent=20.0,
        iex_sizing_price_buffer_percent=0.25,
        iex_sizing_price_buffer_min=0.05,
        iex_sizing_price_buffer_max=0.10,
    )


def test_load_config_falls_back_for_invalid_market_regime(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[defaults]
market_regime = "BAD"
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.market_regime == "GO"
