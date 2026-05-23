# rTrade Insights

Personal Streamlit dashboard and local workflow tools for trade planning, position sizing, and portfolio review.

## What It Does

- Sizes positions from entry price, stop price, portfolio size, and risk percent.
- Tracks active positions and calculated risk metrics in an editable Streamlit table.
- Imports Robinhood CSV reports to derive FIFO exit matches, open lots, closed trades, realized P/L, and strategy metrics.
- Preserves planned stops, strategy tags, ATR %, and market-regime context for later trade analysis.
- Ranks new trade candidates by strategy through a local CLI or FastAPI endpoint.

## Quickstart

Requirements: Python 3.11+.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the URL Streamlit prints, usually `http://localhost:8501`.

## Configuration

Default sizing values live in `config.toml`:

```toml
[defaults]
portfolio_amount = 20000.0
sizing_portfolio_amount = 20000.0
risk_percent = 0.5
market_regime = "GO"
max_symbol_exposure_percent = 20.0
iex_sizing_price_buffer_percent = 0.25
iex_sizing_price_buffer_min = 0.05
iex_sizing_price_buffer_max = 0.10
```

`market_regime` supports `GO`, `SELECTIVE GO`, and `NO-GO`.

## Data Storage

By default, the app stores local runtime data under `data/`:

- `positions.csv`: editable active-position source data.
- `positions_archive.csv`: permanent latest-snapshot archive of every position added through the app.
- `planned_stops.csv`: durable entry stop, strategy, ATR %, and market-regime context.
- `robinhood_transactions.csv`: cleaned imported Robinhood transactions with duplicate uploads skipped.

`positions_archive.csv` is not an event log and does not track open/closed status. Deleting a row from active
positions leaves its archive row intact; editing an active position updates the matching archive snapshot.

Treat `data/` as user-local runtime data. It is ignored by git.

### Google Sheets Storage

When Streamlit secrets include both `[google_sheets]` and `[gcp_service_account]`, the app uses Google Sheets instead of local CSV files. Share the sheet with the service account `client_email`.

Required worksheet tabs are created automatically if missing:

- `positions`
- `positions_archive`
- `planned_stops`
- `robinhood_transactions`

Minimal secrets shape:

```toml
[google_sheets]
spreadsheet_id = "your-google-sheet-id"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```

For local testing, put secrets in `.streamlit/secrets.toml`. For Streamlit Community Cloud, paste the same TOML into the app secrets settings and deploy with entrypoint `app.py`. Do not commit `.streamlit/secrets.toml`, service account JSON, or `data/`.

### Pull Google Sheets into Local CSV

If you want local testing to keep using `data/*.csv`, leave the Google Sheets sections in `.streamlit/secrets.toml`
disabled. Enabling those app secrets makes the Streamlit app read and write Google Sheets directly.

Use the standalone sync helper to refresh local CSV snapshots from Google Sheets without changing the app backend:

```bash
.venv/bin/python tools/sync_gs.py pull
```

The helper reads its own ignored config from `.sync/google_sheets.toml`:

```toml
[google_sheets]
spreadsheet_id = "your-google-sheet-id"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```

Preview a sync without writing files:

```bash
.venv/bin/python tools/sync_gs.py pull --dry-run
```

To use a different config or destination directory:

```bash
.venv/bin/python tools/sync_gs.py pull --config path/to/google_sheets.toml --data-dir data
```

## CLI

Use the module entrypoint from the virtualenv:

```bash
.venv/bin/python -m stock_calculator.cli --help
```

Open one Google AI Mode research tab per ticker:

```bash
.venv/bin/python -m stock_calculator.cli research --file research_tickers.txt
```

`research_tickers.txt` contains ticker symbols separated by lines or whitespace:

```text
PINS
APP
NVDA
```

Rank grouped candidates:

```bash
.venv/bin/python -m stock_calculator.cli rank --file rank_candidates.txt
```

`rank_candidates.txt` uses strategy headers and `SYMBOL PRICE LOD ATR%` rows:

```text
5% BO
PINS 21.16 20.69 5.2

EP
NVDA 100 95 5
```

Use `--format csv` or `--format json` for machine-readable output.

To fill missing price, stop, or ATR % from Alpaca, set credentials outside the repo and pass `--enrich`:

```bash
export APCA_API_KEY_ID="your_key_id"
export APCA_API_SECRET_KEY="your_secret_key"
.venv/bin/python -m stock_calculator.cli rank --file rank_candidates.txt --enrich
```

With `--enrich`, compact rows can omit Alpaca-derived values:

```text
BO
RIGL 27.83 4.54

EP
RIGL 27.83 4.54
```

For all strategies, positional low values mean LOD/reference low; stop is calculated as `LOD - min(max($0.10, price * 0.2%), $1.00)`. Add `SL:<value>` anywhere after the symbol to use an exact manual stop loss instead:

```text
EP
ROIV 29.10 3.21
ROIV 31.55 29.10 3.21 SL:29
ROIV SL:29 3.21
```

The default Alpaca feed is `iex`; use `--feed delayed_sip` or `--feed sip` only if your Alpaca plan supports it.
When enrichment fills price from the default `iex` feed, ranking uses a conservative sizing price to reduce oversizing from stale or thin prints: `raw_price + min(max(raw_price * 0.25%, $0.05), $0.10)`. Manual prices, `delayed_sip`, and `sip` are sized exactly.

## Local API

Start the local ranking API:

```bash
.venv/bin/python -m stock_calculator.cli serve
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Rank candidates through `/rank`:

```bash
curl -X POST "http://127.0.0.1:8000/rank?format=table" \
  --data-binary $'5% BO\nPINS 21.16 20.69 5.2\n\nEP\nNVDA 100 95 5\n'
```

Supported formats are `table`, `csv`, and `json`. The API calculates rankings only; it does not save positions.

## Tests

```bash
.venv/bin/python -m pytest
```
