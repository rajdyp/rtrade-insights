# rTrade Insights

Personal Streamlit dashboard and local workflow tools for trade planning, position sizing, and portfolio review.

## What It Does

- Sizes positions from entry price, stop price, portfolio size, and risk percent, then applies a selected exposure cap.
- Tracks active positions and calculated risk metrics in an editable Streamlit table.
- Imports Robinhood CSV reports to derive FIFO exit matches, open lots, closed trades, realized P/L, and strategy metrics.
- Summarizes completed trades by month and charts their yearly return distribution.
- Preserves planned stops, strategy tags, ATR %, and market-regime context for later trade analysis.
- Ranks new trade candidates by strategy through a local CLI or FastAPI endpoint.

## Quickstart

Requirements: Python 3.11+.

```bash
python3 -m venv .venv
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
risk_percent = 1.0
market_regime = "GO"
max_symbol_exposure_percent = 20.0
add_on_unrealized_profit_preserve_percent = 50.0
iex_sizing_price_buffer_percent = 0.25
iex_sizing_price_buffer_min = 0.05
iex_sizing_price_buffer_max = 0.10
```

`market_regime` supports `GO`, `SELECTIVE GO`, and `NO-GO`.

## Position Sizing

The configured `risk_percent` is a global maximum-risk budget (1% in the checked-in configuration), not an editable
per-trade input. The calculator first determines whole shares from that budget and the stop-loss distance. It then
limits those shares to the selected `Exposure` tier:

- `Full`: 20% of the position's portfolio value.
- `Half`: 10%.
- `Quarter`: 5%.
- `Probe`: 2.5%.

Final shares are the smaller of the risk-based shares and exposure-capped shares. `Total Risk` is the actual stop risk
of those final shares, after whole-share rounding and any exposure cap. Saved active positions default to `Full` when
loading a legacy row with no Exposure value. Exposure is stored with the position and displayed read-only after the
position is added; the other saved source inputs remain editable.

The New Position form and candidate ranking suggest Exposure from Market Regime and the selected strategy's mode:

| Strategy mode | GO | SELECTIVE GO | NO-GO |
| --- | --- | --- | --- |
| Working | Full | Half | Quarter |
| Caution | Half | Quarter | Probe |
| Weak | Quarter | Probe | No Trade |
| Failing | Probe | No Trade | No Trade |
| Unknown | Probe | No Trade | No Trade |

The form allows a manual Exposure override. Changing Regime, Strategy, or the calculated strategy mode resets the
selection to the matrix recommendation; price, stop, ATR, and portfolio edits preserve the override. `No Trade`
produces stop metrics but zero shares, position size, and Total Risk, and cannot be saved as an active position.

This first version deliberately does not add a separate wide-stop rule or a portfolio-wide heat cap. A wide stop can
therefore make the 1% maximum-risk calculation the binding constraint, while a tight stop can make Exposure bind.
The existing aggregate per-symbol Exposure Limit remains a concentration warning rather than a sizing constraint.

When IEX enrichment adds a sizing-price buffer, that buffered price is used for both risk sizing and the exposure cap.

## Data Storage

By default, the app stores local runtime data under `data/`:

- `positions.csv`: editable active-position source data.
- `positions_archive.csv`: permanent latest-snapshot archive of every position added through the app, including its Exposure tier.
- `campaign_overrides.csv`: manual Campaign View `Current Shares` and `Campaign Stop` overrides by symbol.
- `planned_stops.csv`: durable entry stop, strategy, ATR %, and market-regime context.
- `robinhood_transactions.csv`: cleaned imported Robinhood transactions with duplicate uploads skipped.

`positions_archive.csv` is not an event log and does not track open/closed status. Deleting a row from active
positions leaves its archive row intact; editing an active position updates the matching archive snapshot.

The Exposure column is appended to both position schemas for backward compatibility. Legacy active rows with a blank
or missing Exposure load as `Full`. Legacy archive rows retain a blank Exposure: a populated Exposure marks snapshots
written with the new sizing semantics, where `risk_amount`/`Total Risk` is actual final stop risk. On Google Sheets,
the app automatically grows worksheet row and column capacity before writing the expanded schemas and never shrinks it.
Persisted `No Trade` or unknown nonblank Exposure values are rejected instead of silently coerced.

Treat `data/` as user-local runtime data. It is ignored by git.

### Google Sheets Storage

When Streamlit secrets include both `[google_sheets]` and `[gcp_service_account]`, the app uses Google Sheets instead of local CSV files. Share the sheet with the service account `client_email`.

Required worksheet tabs are created automatically if missing:

- `positions`
- `positions_archive`
- `campaign_overrides`
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
4% BO
PINS 21.16 20.69 5.2

EP
NVDA 100 95 5

Pullback
AAPL 200 195 4.5
```

Use `--format csv` or `--format json` for machine-readable output.

Ranking outputs now include `exposure`. It appears beside the sizing fields in the human table and is appended as the
last field in CSV and each JSON candidate row so existing machine-column positions stay unchanged. `risk_percent` is
the configured maximum risk for every candidate. A matrix `No Trade` result keeps stop metrics, reports zero sizing,
and is identified by `exposure` plus `validation_error`; consumers should not infer the policy gate from a zero risk
percentage.

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
  --data-binary $'4% BO\nPINS 21.16 20.69 5.2\n\nEP\nNVDA 100 95 5\n'
```

Supported formats are `table`, `csv`, and `json`. The API calculates rankings only; it does not save positions.

## Tests

```bash
.venv/bin/python -m pytest
```
