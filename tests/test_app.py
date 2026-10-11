"""Offline regressions for the Streamlit entry point and market-data formats."""

from itertools import product
from pathlib import Path
import base64
import ftplib
from io import StringIO
import re

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
    ("SPY", "SPDR S&P 500 ETF Trust", "ETF", "NYSE Arca"),
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
        [("Close", "^IRX")], names=["Price", "Ticker"]
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
    responses = {"yield": treasury_yield(), "prices": None, "plots": []}

    # Keep rendering through Streamlit while retaining numeric chart evidence.
    # Production closes figures after rendering, so capture the data at that boundary.
    original_pyplot = st.pyplot

    def capture_plot(figure=None, *args, **kwargs):
        plotted = figure if hasattr(figure, "get_axes") else plt.gcf()
        for axis in plotted.get_axes():
            responses["plots"].append({
                "title": axis.get_title(),
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "collections": [
                    {"label": item.get_label(), "offsets": np.asarray(item.get_offsets()).copy()}
                    for item in axis.collections
                ],
            })
        return original_pyplot(figure, *args, **kwargs)

    monkeypatch.setattr(st, "pyplot", capture_plot)

    def download(*, tickers, period, interval, **kwargs):
        calls.append({"tickers": tickers, "period": period, "interval": interval})
        if tickers == "^IRX":
            if isinstance(responses["yield"], Exception):
                raise responses["yield"]
            return responses["yield"].copy()
        assert isinstance(tickers, list), "Unexpected market-data request"
        assert kwargs.get("auto_adjust") is True, "Asset returns require explicitly adjusted prices"
        if responses["prices"] is not None:
            if isinstance(responses["prices"], Exception):
                raise responses["prices"]
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


def assert_no_portfolio_results(app):
    assert_no_exception(app)
    assert app.error, "Invalid input or a failed provider must be explained to the user"
    assert not any("Sharpe Ratio" in table.value.columns for table in app.table)
    assert not any("Portfolio Metrics" in element.value for element in app.markdown)
    assert not any("Efficient Frontier and Portfolio Optimization" in element.value for element in app.markdown)


def downloaded_returns(app, interval="1d"):
    label = f"Download Returns ({interval})"
    links = [element.value for element in app.markdown if label in element.value]
    assert len(links) == 1, "Expose one returns CSV with the selected interval in its label"
    match = re.search(r'base64,([^\"]+)"', links[0])
    assert match is not None
    csv_text = base64.b64decode(match.group(1)).decode()
    result = pd.read_csv(StringIO(csv_text), index_col="Date", parse_dates=["Date"])
    return result


def assert_valid_downloads_remain_available(app, expected_return_rows=23):
    assert any("Download Time Series" in element.value for element in app.markdown)
    returns = downloaded_returns(app)
    assert len(returns) == expected_return_rows
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert np.isfinite(returns.to_numpy()).all()


def test_startup_preserves_na_ticker_and_excludes_invalid_directory_rows(offline_app):
    create_app, _, calls = offline_app
    app = create_app()

    assert app.sidebar.multiselect[0].options == ["AAPL", "IBM", "MSFT", "NA", "QQQ", "SPY"]
    assert app.sidebar.selectbox[0].value == "1y"
    assert app.sidebar.selectbox[1].value == "1d"
    assert app.title[0].value == "Portfolio Analytics"
    assert not calls


@pytest.mark.parametrize(
    "asset_class,exchange,search_by",
    list(product(
        ["All", "Stock", "ETF"],
        ["All", "NASDAQ", "NYSE", "NYSE Arca"],
        ["Symbol", "Security Name"],
    )),
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
    assert [warning.value for warning in app.warning] == [
        "Select at least one asset, then choose Analyze portfolio."
    ]
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
    assert calls[-1] == {"tickers": "^IRX", "period": "5d", "interval": "1d"}
    assert any("annual simple investment yield): 4.304%" in element.value for element in app.markdown)
    assert any("Efficient Frontier and Portfolio Optimization" in element.value for element in app.markdown)
    metrics = [table.value for table in app.table if "Sharpe Ratio" in table.value.columns]
    assert len(metrics) == 1
    assert np.isfinite(metrics[0][["Annual Return", "Annual Volatility", "Sharpe Ratio"]].to_numpy()).all()
    assert metrics[0].iloc[0]["Portfolio"] == "Long-only Max Sharpe Ratio"


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
    assert_valid_downloads_remain_available(app)


@pytest.mark.parametrize("layout", ["multiindex", "plain", "nested"])
def test_ambiguous_treasury_columns_preserve_valid_price_and_return_csvs(offline_app, layout):
    create_app, responses, calls = offline_app
    yield_data = treasury_yield(layout=layout)
    responses["yield"] = pd.concat([yield_data, yield_data], axis=1)
    if layout == "nested":
        responses["yield"].columns = pd.MultiIndex.from_tuples([
            ("Close", "^IRX", "a"), ("Close", "^IRX", "b"),
        ])
    prices = market_prices(["AAPL", "MSFT"])
    responses["prices"] = prices
    app = select_and_go(create_app(), ["AAPL", "MSFT"])

    assert_no_portfolio_results(app)
    assert any("The Treasury bill proxy (^IRX) is unavailable:" in element.value for element in app.error)
    assert calls[-1]["tickers"] == "^IRX"
    assert_valid_downloads_remain_available(app)
    expected_returns = prices["Close"].pct_change(fill_method=None).dropna()
    pd.testing.assert_frame_equal(
        downloaded_returns(app), expected_returns, check_freq=False, check_names=False,
    )
    price_links = [element.value for element in app.markdown if "Download Time Series" in element.value]
    assert len(price_links) == 1
    match = re.search(r'base64,([^\"]+)"', price_links[0])
    assert match is not None
    price_csv = pd.read_csv(
        StringIO(base64.b64decode(match.group(1)).decode()),
        index_col="Date", parse_dates=["Date"],
    )
    pd.testing.assert_frame_equal(
        price_csv, prices["Close"], check_freq=False, check_names=False,
    )


@pytest.mark.parametrize("symbols", [["NA"], ["AAPL", "MSFT"]])
def test_object_numeric_price_dates_stop_before_export(offline_app, symbols):
    create_app, responses, calls = offline_app
    prices = market_prices(symbols)
    prices.index = pd.Index(range(1, len(prices) + 1), dtype=object)
    responses["prices"] = prices
    app = select_and_go(create_app(), symbols)

    assert_no_portfolio_results(app)
    assert "Closing prices must have dates, not row numbers." in [
        element.value for element in app.error
    ]
    assert len(calls) == 1
    assert not any("Download" in element.value for element in app.markdown)
    assert not responses["plots"]


@pytest.mark.parametrize("symbols", [["NA"], ["AAPL", "MSFT"]])
def test_empty_price_download_stops_cleanly(offline_app, symbols):
    create_app, responses, calls = offline_app
    responses["prices"] = market_prices(symbols).iloc[:0]
    app = select_and_go(create_app(), symbols)

    assert_no_exception(app)
    assert app.error
    assert len(calls) == 1
    assert not any("Download Time Series" in element.value for element in app.markdown)


@pytest.mark.parametrize("interval", ["1d", "1wk", "1mo", "3mo"])
def test_returns_csv_retains_first_return_dates_and_interval(offline_app, interval):
    """BUG01/02: every valid observation survives the app's actual CSV export."""
    create_app, responses, _ = offline_app
    prices = market_prices(["AAPL", "MSFT"])
    responses["prices"] = prices
    app = create_app()
    app.sidebar.selectbox[1].set_value(interval)
    app = select_and_go(app, ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    expected = prices["Close"].pct_change(fill_method=None).dropna()
    actual = downloaded_returns(app, interval)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False, check_names=False)
    assert actual.index.name == "Date"
    assert actual.index[0] == prices.index[1]
    assert len(actual) == len(prices) - 1


def test_partial_prices_use_only_observed_adjacent_returns(offline_app):
    """BUG03: one interior gap is allowed, without synthesizing a zero return."""
    create_app, responses, _ = offline_app
    prices = market_prices(["AAPL", "MSFT"])
    prices.loc[prices.index[8], ("Close", "MSFT")] = np.nan
    responses["prices"] = prices
    app = select_and_go(create_app(), ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    expected = prices["Close"].pct_change(fill_method=None).dropna()
    actual = downloaded_returns(app)
    pd.testing.assert_frame_equal(actual, expected, check_freq=False, check_names=False)
    assert prices.index[8] not in actual.index
    assert prices.index[9] not in actual.index
    assert len(actual) == len(prices) - 3


@pytest.mark.parametrize(
    "case",
    ["one_price", "two_prices", "three_prices", "all_nan_asset", "no_overlap", "identical_series"],
)
def test_unusable_multiasset_data_stops_before_optimization(offline_app, case):
    """BUG03/04: insufficient/common data and singular covariance are actionable errors."""
    create_app, responses, _ = offline_app
    prices = market_prices(["AAPL", "MSFT"])
    if case == "one_price":
        prices = prices.iloc[:1]
    elif case == "two_prices":
        prices = prices.iloc[:2]
    elif case == "three_prices":
        prices = prices.iloc[:3]
    elif case == "all_nan_asset":
        prices[("Close", "MSFT")] = np.nan
    elif case == "no_overlap":
        prices.loc[prices.index[:12], ("Close", "MSFT")] = np.nan
        prices.loc[prices.index[12:], ("Close", "AAPL")] = np.nan
    else:
        prices[("Close", "MSFT")] = prices[("Close", "AAPL")]
    responses["prices"] = prices
    app = select_and_go(create_app(), ["AAPL", "MSFT"])

    assert_no_portfolio_results(app)
    assert not any("Frontier" in plot["title"] for plot in responses["plots"])
    if case in {"three_prices", "identical_series"}:
        assert_valid_downloads_remain_available(app, expected_return_rows=len(prices) - 1)


def test_weights_metrics_and_frontier_marker_describe_same_long_only_portfolio(offline_app):
    """BUG06: rendered tables and the plotted star describe one feasible allocation."""
    create_app, responses, _ = offline_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_no_exception(app)
    assert not app.error

    text_elements = list(app.markdown) + list(app.caption) + list(app.info)
    assert any("Long-only: weights are between 0 and 1 and sum to 1." in item.value for item in text_elements)
    weights_tables = [
        element.value for element in list(app.dataframe) + list(app.table)
        if "Opt Weight" in element.value.columns and "Symbol" in element.value.columns
    ]
    assert len(weights_tables) == 1
    weights = weights_tables[0].set_index("Symbol")["Opt Weight"].reindex(["AAPL", "MSFT"]).to_numpy()
    assert np.isfinite(weights).all()
    assert (weights >= -1e-8).all() and (weights <= 1 + 1e-8).all()
    assert weights.sum() == pytest.approx(1, abs=1e-8)

    returns = downloaded_returns(app)
    expected_return = float(weights @ (returns.mean().to_numpy() * 252))
    expected_risk = float(np.sqrt(weights @ (returns.cov().to_numpy() * 252) @ weights))
    # Compare annual arithmetic moments with the full annual simple IRX proxy.
    risk_free = 365 * 0.042 / (360 - 91 * 0.042)
    metrics = [table.value for table in app.table if "Sharpe Ratio" in table.value.columns]
    assert len(metrics) == 1
    displayed = metrics[0].iloc[0]
    assert displayed["Portfolio"] == "Long-only Max Sharpe Ratio"
    assert displayed["Annual Return"] == pytest.approx(expected_return, rel=1e-7, abs=1e-10)
    assert displayed["Annual Volatility"] == pytest.approx(expected_risk, rel=1e-7, abs=1e-10)
    assert displayed["Sharpe Ratio"] == pytest.approx((expected_return - risk_free) / expected_risk, rel=1e-7)

    markers = [
        collection["offsets"] for plot in responses["plots"] for collection in plot["collections"]
        if collection["label"] == "Max Sharpe Ratio Portfolio"
    ]
    assert len(markers) == 1
    np.testing.assert_allclose(markers[0], [[expected_risk, expected_return]], rtol=1e-7, atol=1e-10)


def test_optimizer_failure_is_explained_before_results(offline_app, monkeypatch):
    """BUG07: the app honors an unsuccessful validation at the optimizer boundary."""
    import portfolio_modes_v1 as portfolio

    create_app, responses, _ = offline_app
    message = "The optimizer did not converge. Try a different asset selection."

    def failed_optimization(*args, **kwargs):
        raise portfolio.PortfolioError(message)

    monkeypatch.setattr(portfolio, "optimize_portfolio", failed_optimization)
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_no_portfolio_results(app)
    assert message in [element.value for element in app.error]
    assert not any("Frontier" in plot["title"] for plot in responses["plots"])
    assert_valid_downloads_remain_available(app)


@pytest.mark.parametrize("failure", [TimeoutError("controlled FTP timeout"), ftplib.error_perm("550 unavailable")])
def test_catalog_provider_failure_stops_cleanly(offline_app, monkeypatch, failure):
    """BUG09: failures raised by the catalogue provider do not expose a traceback."""
    create_app, _, calls = offline_app

    def fail_transfer(self, command, callback, *args, **kwargs):
        raise failure

    monkeypatch.setattr(FakeFTP, "retrbinary", fail_transfer)
    app = create_app()
    assert_no_portfolio_results(app)
    assert not calls


def test_invalid_catalog_stops_cleanly(offline_app, monkeypatch):
    create_app, _, calls = offline_app
    monkeypatch.setattr(FakeFTP, "files", {**FakeFTP.files, "nasdaqlisted.txt": b"bad|schema\nwrong|columns\n"})
    app = create_app()
    assert_no_portfolio_results(app)
    assert not calls


@pytest.mark.parametrize("provider", ["prices", "yield"])
def test_yahoo_exception_stops_cleanly(offline_app, provider):
    """BUG10: both external Yahoo calls have a recoverable error boundary."""
    create_app, responses, calls = offline_app
    responses[provider] = TimeoutError("controlled Yahoo timeout")
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_no_portfolio_results(app)
    assert len(calls) == (1 if provider == "prices" else 2)


def test_single_asset_yahoo_exception_stops_cleanly(offline_app):
    create_app, responses, calls = offline_app
    responses["prices"] = TimeoutError("controlled Yahoo timeout")
    app = select_and_go(create_app(), ["NA"])
    assert_no_portfolio_results(app)
    assert len(calls) == 1


def test_kde_describes_returns_and_selected_interval(offline_app):
    """BUG11: negative/positive returns are labeled as returns, including their horizon."""
    create_app, responses, _ = offline_app
    app = create_app()
    app.sidebar.selectbox[1].set_value("1mo")
    app = select_and_go(app, ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    title = "Distribution of Returns by Asset (KDE)"
    assert any(title in element.value for element in app.markdown)
    kde_axes = [plot for plot in responses["plots"] if plot["title"] == "Return distributions"]
    assert len(kde_axes) == 1
    assert kde_axes[0]["xlabel"] == "Return (1mo)"
    assert not any("Distribution of Volatility" in element.value for element in app.markdown)


def test_other_exchange_codes_keep_independent_filters(offline_app, monkeypatch):
    """BUG12: a known code retains its market and an unknown code stays explicit."""
    create_app, _, calls = offline_app
    extra_rows = (
        "AAA|BATS security|Z|AAA|Y|100|N|AAA\n"
        "ACU|NYSE MKT security|A|ACU|N|100|N|ACU\n"
        "IEXS|IEX security|V|IEXS|N|100|N|IEXS\n"
        "UNKN|Unknown market security|F|UNKN|N|100|N|UNKN\n"
    )
    directory = OTHER_DIRECTORY.replace("File Creation Time:", extra_rows + "File Creation Time:")
    monkeypatch.setattr(FakeFTP, "files", {**FakeFTP.files, "otherlisted.txt": directory.encode()})
    app = create_app()
    expected_markets = {"NYSE": ["IBM"], "NYSE Arca": ["SPY"], "BATS": ["AAA"], "NYSE MKT": ["ACU"], "IEX": ["IEXS"], "Unknown (F)": ["UNKN"]}
    for market, symbols in expected_markets.items():
        assert market in app.sidebar.radio[1].options
        app.sidebar.radio[1].set_value(market).run()
        assert_no_exception(app)
        assert app.sidebar.multiselect[0].options == symbols
    assert not calls
