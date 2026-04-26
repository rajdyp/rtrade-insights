# Stock Calculator

Local Streamlit app for spreadsheet-style stock position sizing and portfolio risk tracking.

## Run

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open the URL Streamlit prints, usually `http://localhost:8501`.

Light and dark mode are controlled through Streamlit's settings menu.

## Test

```bash
.venv/bin/python -m pytest
```

## Data

Editable source data lives in `data/positions.csv`. The app keeps calculated fields generated from the source columns instead of saving formula outputs back into the input file.

Durable entry stops live in `data/planned_stops.csv`. Robinhood trade reporting reads planned stops from this ledger by exact symbol, buy date, and quantity, so closed trades keep their original stop after active rows are removed from `positions.csv`.

Cleaned Robinhood upload transactions are stored locally in `data/robinhood_transactions.csv`. Later Robinhood uploads append new transactions and skip duplicate rows so overlapping exports do not double-count metrics.

## Config

Calculator defaults live in `config.toml`:

```toml
[defaults]
portfolio_amount = 20000.0
risk_percent = 0.5
```

Each saved position still stores the portfolio and risk values used for that row.
