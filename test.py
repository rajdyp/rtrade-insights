import os
import requests

key = os.environ.get("MASSIVE_API_KEY", "").strip()
if not key:
    raise SystemExit("MASSIVE_API_KEY is not set in this shell")

endpoint = "https://api.massive.com/v2/snapshot/locale/us/markets/stocks/tickers"
attempts = [
    ("bearer_header", {"headers": {"Authorization": f"Bearer {key}"}, "params": {"tickers": "AAPL"}}),
    ("apiKey_param", {"headers": {}, "params": {"tickers": "AAPL", "apiKey": key}}),
]

for name, kwargs in attempts:
    print(f"\n== {name} ==")
    try:
        response = requests.get(endpoint, timeout=15, **kwargs)
    except Exception as exc:
        print(f"request_error={type(exc).__name__}: {exc}")
        continue

    print(f"status={response.status_code}")
    print(f"url={endpoint}")
    print(f"content_type={response.headers.get('content-type', '')}")

    try:
        payload = response.json()
    except Exception:
        print("json=false")
        print("body_prefix=" + response.text[:300].replace("\n", " "))
        continue

    print("json=true")
    if not isinstance(payload, dict):
        print(f"root_type={type(payload).__name__}")
        continue

    print("top_keys=" + ",".join(payload.keys()))
    tickers = payload.get("tickers")
    print(f"tickers_type={type(tickers).__name__}")

    if isinstance(tickers, list):
        print(f"tickers_len={len(tickers)}")
        if tickers and isinstance(tickers[0], dict):
            sample = tickers[0]
            print("sample_keys=" + ",".join(sample.keys()))
            for field in ("ticker", "lastTrade", "lastQuote", "min", "day"):
                value = sample.get(field)
                if isinstance(value, dict):
                    print(f"{field}_keys=" + ",".join(value.keys()))
                else:
                    print(f"{field}_type={type(value).__name__}")
    elif isinstance(tickers, dict):
        print("tickers_dict_keys_sample=" + ",".join(list(tickers.keys())[:5]))

    if response.status_code >= 400:
        print("error_body_prefix=" + response.text[:300].replace("\n", " "))