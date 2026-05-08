from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path


DEFAULT_TEMPLATE = "What's up with {ticker} stock today?"
DEFAULT_START_URL = "https://www.google.com/ai"
FALLBACK_START_URL = "https://www.google.com/aimode"
DEFAULT_CHROME = "/usr/bin/google-chrome"

TEXTBOX_SELECTORS = (
    "textarea:visible",
    "[contenteditable='true']:visible",
    "[role='textbox']:visible",
    "input[type='text']:visible",
    "input[name='q']:visible",
)

PlaywrightError = Exception
PlaywrightTimeoutError = TimeoutError
sync_playwright = None


def read_ticker_file(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"Ticker file does not exist: {path}")

    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tickers.extend(line.split())
    return normalize_tickers(tickers)


def normalize_tickers(tickers: list[str]) -> list[str]:
    normalized_tickers = []
    seen = set()
    for ticker in tickers:
        normalized = ticker.strip().upper()
        if not normalized or normalized in seen:
            continue
        normalized_tickers.append(normalized)
        seen.add(normalized)
    return normalized_tickers


def find_chrome(path: str) -> str:
    if Path(path).exists():
        return path

    discovered = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if discovered:
        return discovered

    raise ValueError(
        f"Chrome executable was not found at {path}. "
        "Install Chrome or pass --chrome /path/to/google-chrome."
    )


def open_research_tabs(
    tickers: list[str],
    *,
    profile: Path = Path(".browser-profile"),
    delay: float = 1.0,
    template: str = DEFAULT_TEMPLATE,
    start_url: str = DEFAULT_START_URL,
    chrome: str = DEFAULT_CHROME,
    keep_open: bool = True,
) -> int:
    if "{ticker}" not in template:
        raise ValueError("--template must contain {ticker}.")

    normalized_tickers = normalize_tickers(tickers)
    if not normalized_tickers:
        raise ValueError("Provide at least one ticker or use --file research_tickers.txt.")

    chrome_path = find_chrome(chrome)
    profile.mkdir(parents=True, exist_ok=True)
    _load_playwright()

    print(f"Opening {len(normalized_tickers)} AI Mode tab(s) with profile: {profile}")
    print("Use Ctrl+C in this terminal when you are done, or press Enter at the final prompt.")

    try:
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile),
                executable_path=chrome_path,
                headless=False,
                args=["--start-maximized"],
                no_viewport=True,
            )

            for index, ticker in enumerate(normalized_tickers, start=1):
                query = template.format(ticker=ticker)
                print(f"[{index}/{len(normalized_tickers)}] {ticker}: {query}")
                try:
                    _submit_with_login_retry(context, start_url, query)
                except PlaywrightError as exc:
                    print(f"Failed to submit {ticker}: {exc}", file=sys.stderr)
                    continue

                if delay > 0 and index < len(normalized_tickers):
                    time.sleep(delay)

            if keep_open:
                input("\nTabs are open. Press Enter to close Chrome and exit...")

            context.close()
    except KeyboardInterrupt:
        print("\nExiting.")
        return 130

    return 0


def _load_playwright() -> None:
    global PlaywrightError, PlaywrightTimeoutError, sync_playwright

    try:
        from playwright.sync_api import (
            Error as ImportedPlaywrightError,
            TimeoutError as ImportedPlaywrightTimeoutError,
            sync_playwright as imported_sync_playwright,
        )
    except ModuleNotFoundError:
        print(
            "Playwright is not installed. Run:\n"
            "  .venv/bin/python -m pip install playwright",
            file=sys.stderr,
        )
        raise SystemExit(1)

    PlaywrightError = ImportedPlaywrightError
    PlaywrightTimeoutError = ImportedPlaywrightTimeoutError
    sync_playwright = imported_sync_playwright


def _wait_for_textbox(page, timeout_ms: int = 12000):
    last_error: Exception | None = None
    for selector in TEXTBOX_SELECTORS:
        locator = page.locator(selector).last
        try:
            locator.wait_for(state="visible", timeout=timeout_ms)
            return locator
        except PlaywrightTimeoutError as exc:
            last_error = exc

    raise PlaywrightTimeoutError(
        "Could not find the AI Mode input box. "
        "Google may have changed the page, login may be required, "
        "or this account may not have AI Mode access."
    ) from last_error


def _open_ai_mode_page(context, start_url: str):
    page = context.new_page()
    page.goto(start_url, wait_until="domcontentloaded")

    try:
        _wait_for_textbox(page, timeout_ms=8000)
        return page
    except PlaywrightTimeoutError:
        if start_url.rstrip("/") == FALLBACK_START_URL.rstrip("/"):
            raise
        page.goto(FALLBACK_START_URL, wait_until="domcontentloaded")
        _wait_for_textbox(page, timeout_ms=12000)
        return page


def _submit_query(page, query: str) -> None:
    textbox = _wait_for_textbox(page)
    textbox.click()
    textbox.fill(query)
    textbox.press("Enter")


def _submit_with_login_retry(context, start_url: str, query: str):
    page = _open_ai_mode_page(context, start_url)
    try:
        _submit_query(page, query)
        return page
    except PlaywrightTimeoutError:
        print(
            "\nCould not find the AI Mode input box. "
            "If Chrome is asking for Google login or consent, complete it in the "
            "browser window, then press Enter here to retry.",
            file=sys.stderr,
        )
        input()
        page.reload(wait_until="domcontentloaded")
        _submit_query(page, query)
        return page
