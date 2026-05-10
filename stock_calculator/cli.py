from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stock_calculator.ranking import rank_candidates, render_rank_result
from stock_calculator.research import (
    DEFAULT_CHROME,
    DEFAULT_START_URL,
    DEFAULT_TEMPLATE,
    open_research_tabs,
    read_ticker_file,
)


SUPPORTED_FORMATS = ("table", "csv", "json")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rTrade Insights local workflow tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    research = subparsers.add_parser("research", help="Open Google AI Mode research tabs.")
    research.add_argument("tickers", nargs="*", help="Ticker symbols, for example: AMD NVDA")
    research.add_argument("--file", type=Path, help="Ticker-only file, with symbols separated by whitespace or lines.")
    research.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between tabs. Default: 1.0")
    research.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="Query template. Must contain {ticker}. Default: %(default)r",
    )
    research.add_argument(
        "--profile",
        type=Path,
        default=Path(".browser-profile"),
        help="Persistent Chrome profile directory. Default: .browser-profile",
    )
    research.add_argument("--start-url", default=DEFAULT_START_URL, help="AI Mode start URL.")
    research.add_argument("--chrome", default=DEFAULT_CHROME, help="Chrome executable path.")
    research.add_argument("--no-keep-open", action="store_true", help="Close Chrome when tabs are opened.")
    research.set_defaults(handler=_handle_research)

    rank = subparsers.add_parser("rank", help="Rank grouped candidates from a file.")
    rank.add_argument("--file", type=Path, required=True, help="Grouped candidate file with SYMBOL PRICE STOP ATR%%.")
    rank.add_argument("--format", choices=SUPPORTED_FORMATS, default="table", help="Output format. Default: table")
    rank.add_argument("--enrich", action="store_true", help="Fetch missing price, stop, and ATR%% from Alpaca market data.")
    rank.add_argument(
        "--feed",
        choices=("iex", "delayed_sip", "sip"),
        default="iex",
        help="Alpaca market data feed for --enrich. Default: iex",
    )
    rank.set_defaults(handler=_handle_rank)

    serve = subparsers.add_parser("serve", help="Start the local /rank HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_handle_serve)

    return parser


def _handle_research(args: argparse.Namespace) -> int:
    if "{ticker}" not in args.template:
        raise ValueError("--template must contain {ticker}.")

    tickers = list(args.tickers)
    if args.file:
        tickers.extend(read_ticker_file(args.file))

    return open_research_tabs(
        tickers,
        profile=args.profile,
        delay=args.delay,
        template=args.template,
        start_url=args.start_url,
        chrome=args.chrome,
        keep_open=not args.no_keep_open,
    )


def _handle_rank(args: argparse.Namespace) -> int:
    if not args.file.exists():
        raise ValueError(f"Ranking file does not exist: {args.file}")

    text = args.file.read_text(encoding="utf-8")
    print(render_rank_result(rank_candidates(text, enrich=args.enrich, feed=args.feed), args.format), end="")
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("stock_calculator.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
