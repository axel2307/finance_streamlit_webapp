"""Offline regressions for the Streamlit entry point and market-data formats."""

from itertools import product
from pathlib import Path
import ftplib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import yfinance as yf


APP_PATH = Path(__file__).resolve().parents[1] / "finance_streamlit_webapp.py"

NASDAQ_DIRECTORY = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc.|Q|N|N|100|N|N
MSFT|Microsoft Corporation|Q|N|N|100|N|N
NA|Nano Labs Ltd|G|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
TEST|Test security|Q|Y|N|100|N|N
HALT|Non-normal financial status|Q|N|D|100|N|N
|Missing symbol|Q|N|N|100|N|N
EMPTYNAME||Q|N|N|100|N|N
File Creation Time: 101020261500|||||||
"""

OTHER_DIRECTORY = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
IBM|International Business Machines|N|IBM|N|100|N|IBM
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
TEST2|Other test security|N|TEST2|N|100|Y|TEST2
MISSING|Other missing symbol|N|MISSING|N|100|N|
EMPTYNAME2||N|EMPTYNAME2|N|100|N|EMPTYNAME2
File Creation Time: 101020261500|||||||
"""

# Expected user-visible records, independently specified from the CSV parser.
VALID_ASSETS = [
    ("AAPL", "Apple Inc.", "Stock", "NASDAQ"),
    ("MSFT", "Microsoft Corporation", "Stock", "NASDAQ"),
    ("NA", "Nano Labs Ltd", "Stock", "NASDAQ"),
    ("QQQ", "Invesco QQQ Trust", "ETF", "NASDAQ"),
    ("IBM", "International Business Machines", "Stock", "NYSE"),
    ("SPY", "SPDR S&P 500 ETF Trust", "ETF", "NYSE"),
]


class FakeFTP:
    """The public directory transfer, with no socket or credential access."""

    files = {
        "nasdaqlisted.txt": NASDAQ_DIRECTORY.encode(),
        "otherlisted.txt": OTHER_DIRECTORY.encode(),
    }

    def __init__(self, host="", *args, **kwargs):
        assert host == "ftp.nasdaqtrader.com"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.quit()

    def login(self, *args, **kwargs):
        return "230 Logged in"

    def cwd(self, directory):
        assert directory == "symboldirectory"
        return "250 Directory changed"

    def retrbinary(self, command, callback, *args, **kwargs):
        verb, filename = command.split(" ", 1)
        assert verb == "RETR"
        callback(self.files[filename])
        return "226 Transfer complete"

    def quit(self):
        return "221 Goodbye"


def market_prices(tickers):
    """Finite, non-collinear closes in yfinance's default MultiIndex layout."""
    dates = pd.date_range("2025-01-02", periods=24, freq="B", name="Date")
    step = np.arange(len(dates), dtype=float)
    values = []
    columns = []
    for position, ticker in enumerate(tickers):
        daily_returns = (
            0.0015
            + 0.008 * np.sin(step * (0.67 + position * 0.29))
            + 0.004 * np.cos(step * (0.31 + position * 0.13))
        )
        values.append((100 + 25 * position) * np.cumprod(1 + daily_returns))
        columns.append(("Close", ticker))
    return pd.DataFrame(
        np.column_stack(values),
        index=dates,
        columns=pd.MultiIndex.from_tuples(columns, names=["Price", "Ticker"]),
    )


def treasury_yield(layout="multiindex", value=4.2):
    dates = pd.DatetimeIndex(["2025-02-04"], name="Date")
    if layout == "plain":
        return pd.DataFrame({"Close": [value]}, index=dates)
    columns = pd.MultiIndex.from_tuples(
        [("Close", "^TNX")], names=["Price", "Ticker"]
    )
    return pd.DataFrame([[value]], index=dates, columns=columns)


@pytest.fixture
def offline_app(monkeypatch, tmp_path):
    """Isolate files, cached directories, figures and both network providers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ftplib, "FTP", FakeFTP)
    st.cache_data.clear()
    st.cache_resource.clear()
    calls = []
    responses = {"yield": treasury_yield(), "prices": None}

    def download(*, tickers, period, interval, **kwargs):
        calls.append({"tickers": tickers, "period": period, "interval": interval})
        if tickers == "^TNX":
            return responses["yield"].copy()
        assert isinstance(tickers, list), "Unexpected market-data request"
        if responses["prices"] is not None:
            return responses["prices"].copy()
        return market_prices(tickers)

    monkeypatch.setattr(yf, "download", download)
    random_generator = np.random.default_rng(12345)
    monkeypatch.setattr(np.random, "dirichlet", random_generator.dirichlet)

    def create_app():
        app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        assert not app.exception, [exception.message for exception in app.exception]
        return app

    yield create_app, responses, calls
    plt.close("all")
    st.cache_data.clear()
    st.cache_resource.clear()


def assert_no_exception(app):
    assert not app.exception, [exception.message for exception in app.exception]


def select_and_go(app, symbols):
    app.sidebar.multiselect[0].set_value(symbols)
    app.sidebar.button[0].click().run()
    return app


def test_startup_preserves_na_ticker_and_excludes_invalid_directory_rows(offline_app):
    create_app, _, calls = offline_app
    app = create_app()

    assert app.sidebar.multiselect[0].options == ["AAPL", "IBM", "MSFT", "NA", "QQQ", "SPY"]
    assert app.sidebar.selectbox[0].value == "1y"
    assert app.sidebar.selectbox[1].value == "1d"
    assert app.title[0].value == "Portfolio Optimization and Financial Analytics Tool"
    assert not calls


@pytest.mark.parametrize(
    "asset_class,exchange,search_by",
    list(product(["All", "Stock", "ETF"], ["All", "NASDAQ", "NYSE"], ["Symbol", "Security Name"])),
)
def test_filter_combinations(offline_app, asset_class, exchange, search_by):
    create_app, _, calls = offline_app
    app = create_app()
    app.sidebar.radio[0].set_value(asset_class)
    app.sidebar.radio[1].set_value(exchange)
    app.sidebar.radio[2].set_value(search_by)
    app.run()

    assert_no_exception(app)
    expected = sorted(
        asset[0 if search_by == "Symbol" else 1]
        for asset in VALID_ASSETS
        if (asset_class == "All" or asset[2] == asset_class)
        and (exchange == "All" or asset[3] == exchange)
    )
    assert app.sidebar.multiselect[0].options == expected
    assert not calls


def test_go_without_selection_warns_without_downloading(offline_app):
    create_app, _, calls = offline_app
    app = create_app()
    app.sidebar.button[0].click().run()

    assert_no_exception(app)
    assert [warning.value for warning in app.warning] == ["You must select an asset first"]
    assert not calls


def test_na_ticker_can_be_selected_and_downloaded(offline_app):
    create_app, _, calls = offline_app
    app = select_and_go(create_app(), ["NA"])

    assert_no_exception(app)
    assert calls == [{"tickers": ["NA"], "period": "1y", "interval": "1d"}]
    assert any("Download Time Series" in element.value for element in app.markdown)
    assert not app.error


@pytest.mark.parametrize("layout", ["multiindex", "plain"])
def test_two_assets_complete_with_scalar_treasury_yield(offline_app, layout):
    create_app, responses, calls = offline_app
    responses["yield"] = treasury_yield(layout=layout)
    app = select_and_go(create_app(), ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    assert calls[-1] == {"tickers": "^TNX", "period": "1d", "interval": "1d"}
    assert any("Risk Free Rate: 4.20%" in element.value for element in app.markdown)
    assert any("Efficient Frontier and Portfolio Optimization" in element.value for element in app.markdown)
    metrics = [table.value for table in app.table if "Sharpe Ratio" in table.value.columns]
    assert len(metrics) == 1
    assert np.isfinite(metrics[0][["Return", "Std Dev", "Sharpe Ratio"]].to_numpy()).all()


@pytest.mark.parametrize("failure", ["empty", "nan", "infinite", "missing_close"])
def test_unavailable_treasury_yield_stops_cleanly(offline_app, failure):
    create_app, responses, _ = offline_app
    if failure == "empty":
        responses["yield"] = treasury_yield().iloc[:0]
    elif failure == "missing_close":
        responses["yield"] = treasury_yield(layout="plain").rename(columns={"Close": "Open"})
    else:
        responses["yield"] = treasury_yield(value=np.nan if failure == "nan" else np.inf)
    app = select_and_go(create_app(), ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert app.error
    assert not any("Sharpe Ratio" in table.value.columns for table in app.table)
    assert not any("Efficient Frontier and Portfolio Optimization" in element.value for element in app.markdown)


@pytest.mark.parametrize("symbols", [["NA"], ["AAPL", "MSFT"]])
def test_empty_price_download_stops_cleanly(offline_app, symbols):
    create_app, responses, calls = offline_app
    responses["prices"] = market_prices(symbols).iloc[:0]
    app = select_and_go(create_app(), symbols)

    assert_no_exception(app)
    assert app.error
    assert len(calls) == 1
    assert not any("Download Time Series" in element.value for element in app.markdown)
