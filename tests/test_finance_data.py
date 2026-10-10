"""Offline regressions for catalog identity and dated, unfilled returns."""

from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pytest

from finance_data import DataError, EXCHANGE_LABELS, extract_close_prices, parse_catalog, prepare_returns


NASDAQ = """Symbol|Security Name|ETF|Test Issue|Financial Status
NA|Nano Labs Ltd|N|N|N
AAPL|Apple Inc.|N|N|N
TEST|Test entry|N|Y|N
HALT|Non-normal entry|N|N|D
|Missing symbol|N|N|N
MISSINGNAME||N|N|N
File Creation Time: 1009202621:31||||
"""
OTHER = """NASDAQ Symbol|Security Name|ETF|Test Issue|Exchange
IBM|International Business Machines|N|N|N
SPY|State Street SPDR S&P 500 ETF Trust|Y|N|P
AAAA|Amplius Aggressive Asset Allocation ETF|Y|N|Z
AAAU|Goldman Sachs Physical Gold ETF Shares|Y|N|Z
ACU|Acme United Corporation|N|N|A
IEXTEST|Controlled IEX fixture|N|N|V
UNKNOWN|Controlled unknown-code fixture|Y|N|F
TEST|Test entry|N|Y|P
|Missing symbol|N|N|N
MISSINGNAME||N|N|N
File Creation Time: 1009202621:31||||
"""


def close_fixture():
    return pd.DataFrame(
        {"AAPL": [100, 110, 111.1, 113.322], "MSFT": [200, 210, 216.3, 211.974]},
        index=pd.date_range("2026-10-01", periods=4, freq="B", name="Date"),
    )


def as_download(close):
    frame = close.copy()
    frame.columns = pd.MultiIndex.from_tuples(
        [("Close", symbol) for symbol in frame.columns], names=["Price", "Ticker"],
    )
    return frame


@pytest.mark.parametrize("wrap", [lambda value: value, lambda value: value.encode(), StringIO, lambda value: BytesIO(value.encode())])
def test_catalog_preserves_na_and_source_exchanges(wrap):
    catalog = parse_catalog(wrap(NASDAQ), wrap(OTHER)).set_index("Symbol")
    assert catalog["Exchange"].to_dict() == {
        "NA": "NASDAQ", "AAPL": "NASDAQ", "IBM": "NYSE", "SPY": "NYSE Arca",
        "AAAA": "BATS", "AAAU": "BATS", "ACU": "NYSE MKT", "IEXTEST": "IEX",
        "UNKNOWN": "Unknown (F)",
    }
    assert catalog.loc["UNKNOWN", "Exchange Code"] == "F"
    assert catalog.loc["SPY", "Exchange Code"] == "P"
    assert list(catalog.loc[catalog["Exchange"] == "NYSE"].index) == ["IBM"]
    assert EXCHANGE_LABELS["N"] == "NYSE"


@pytest.mark.parametrize("nasdaq,other,match", [
    ("", OTHER, "could not be read"),
    ("Symbol|Security Name\nNA|Nano Labs\n", OTHER, "missing columns"),
    (NASDAQ, "NASDAQ Symbol|Security Name\nIBM|IBM\n", "missing columns"),
    ("Symbol|Security Name|ETF|Test Issue|Financial Status\n", "NASDAQ Symbol|Security Name|ETF|Test Issue|Exchange\n", "no valid assets"),
])
def test_bad_catalog_fails_with_actionable_error(nasdaq, other, match):
    with pytest.raises(DataError, match=match):
        parse_catalog(nasdaq, other)


@pytest.mark.parametrize("reverse_levels", [False, True])
def test_extract_multiindex_preserves_dates_and_requested_order(reverse_levels):
    download = as_download(close_fixture())
    if reverse_levels:
        download = download.swaplevel(axis=1)
    original = download.copy(deep=True)
    close = extract_close_prices(download, ["MSFT", "AAPL"])
    pd.testing.assert_frame_equal(close, close_fixture()[["MSFT", "AAPL"]].astype(float))
    pd.testing.assert_frame_equal(download, original)


def test_extract_plain_single_asset_and_sort_dates():
    download = close_fixture()[["AAPL"]].rename(columns={"AAPL": "Close"}).iloc[::-1]
    close = extract_close_prices(download, ["AAPL"])
    pd.testing.assert_frame_equal(close, close_fixture()[["AAPL"]].astype(float))


def test_returns_keep_first_valid_observation_and_csv_dates():
    close = extract_close_prices(as_download(close_fixture()), ["AAPL", "MSFT"])
    result = prepare_returns(close)
    expected = pd.DataFrame(
        {"AAPL": [0.10, 0.01, 0.02], "MSFT": [0.05, 0.03, -0.02]},
        index=pd.DatetimeIndex(["2026-10-02", "2026-10-05", "2026-10-06"], name="Date"),
    )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)
    np.testing.assert_allclose(result.mean(), [0.04333333333333333, 0.02])
    exported = pd.read_csv(StringIO(result.to_csv(index=True)), index_col="Date", parse_dates=True)
    pd.testing.assert_frame_equal(exported, expected, check_freq=False)


@pytest.mark.parametrize("count,match", [(1, "No common returns"), (2, "At least two common returns")])
def test_short_history_is_rejected(count, match):
    with pytest.raises(DataError, match=match):
        prepare_returns(close_fixture().iloc[:count])


def test_three_prices_produce_two_returns():
    result = prepare_returns(close_fixture().iloc[:3])
    assert len(result) == 2
    np.testing.assert_allclose(result["AAPL"], [0.1, 0.01])


def test_absent_asset_is_named_in_error():
    with pytest.raises(DataError, match="MSFT"):
        extract_close_prices(as_download(close_fixture()[["AAPL"]]), ["AAPL", "MSFT"])


def test_all_nan_asset_is_named_in_error():
    close = close_fixture()
    close["MSFT"] = np.nan
    with pytest.raises(DataError, match="MSFT"):
        extract_close_prices(as_download(close), ["AAPL", "MSFT"])


def test_no_overlap_is_rejected_without_fabricating_returns():
    close = close_fixture()
    close.loc[close.index[:2], "MSFT"] = np.nan
    close.loc[close.index[2:], "AAPL"] = np.nan
    with pytest.raises(DataError, match="No common returns"):
        prepare_returns(extract_close_prices(as_download(close), ["AAPL", "MSFT"]))


def test_partial_history_keeps_only_observed_common_returns_without_filling():
    close = pd.DataFrame(
        {"AAPL": [100, 110, 121, 133.1, 146.41, 161.051], "MSFT": [200, 220, np.nan, 242, 266.2, 292.82]},
        index=pd.date_range("2026-10-01", periods=6, freq="B", name="Date"),
    )
    extracted = extract_close_prices(as_download(close), ["MSFT", "AAPL"])
    assert pd.isna(extracted.loc[close.index[2], "MSFT"])
    result = prepare_returns(extracted)
    assert result.index.tolist() == close.index[[1, 4, 5]].tolist()
    assert result.columns.tolist() == ["MSFT", "AAPL"]
    np.testing.assert_allclose(result, 0.1)


@pytest.mark.parametrize("value", [0, -1, np.inf, -np.inf, "bad"])
def test_invalid_available_close_is_rejected(value):
    close = close_fixture().astype(object)
    close.loc[close.index[1], "MSFT"] = value
    with pytest.raises(DataError, match="MSFT"):
        extract_close_prices(as_download(close), ["AAPL", "MSFT"])


@pytest.mark.parametrize("date_index", [pd.RangeIndex(4), pd.Index(["bad"] * 4), pd.DatetimeIndex(["2026-10-01"] * 4), pd.DatetimeIndex(["2026-10-01", "2026-10-02", "2026-10-05", pd.NaT])])
def test_missing_ambiguous_or_invalid_dates_are_rejected(date_index):
    close = close_fixture()
    close.index = date_index
    with pytest.raises(DataError, match="date|row numbers"):
        extract_close_prices(as_download(close), ["AAPL", "MSFT"])


@pytest.mark.parametrize("values", [
    pytest.param([1, 2, 3, 4], id="integers"),
    pytest.param([1.5, 2.5, 3.5, 4.5], id="floats"),
    pytest.param([pd.Timestamp("2026-10-01"), 2, pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")], id="dates-and-integer"),
    pytest.param([pd.Timestamp("2026-10-01"), 2.5, pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")], id="dates-and-float"),
])
def test_object_index_with_numbers_is_rejected_before_date_conversion(values):
    """Numeric labels must not turn into invented nanosecond dates in 1970."""
    close = close_fixture()
    close.index = pd.Index(values, dtype=object)
    with pytest.raises(DataError, match="date|row numbers"):
        extract_close_prices(as_download(close), ["AAPL", "MSFT"])
    with pytest.raises(DataError, match="date|row numbers"):
        prepare_returns(close)


@pytest.mark.parametrize("representation", ["object-text", "string-text", "date-objects", "timestamp-objects"])
def test_non_datetime_date_index_preserves_prices_returns_and_csv(representation):
    """Valid date representations still identify the original observations."""
    expected_prices = close_fixture()
    close = expected_prices.copy()
    dates = expected_prices.index
    if representation == "object-text":
        close.index = pd.Index(dates.strftime("%Y-%m-%d"), dtype=object)
    elif representation == "string-text":
        close.index = pd.Index(dates.strftime("%Y-%m-%d"), dtype="string")
    elif representation == "date-objects":
        close.index = pd.Index(dates.date, dtype=object)
    else:
        close.index = pd.Index(dates.tolist(), dtype=object)

    extracted = extract_close_prices(as_download(close), ["AAPL", "MSFT"])
    assert isinstance(extracted.index, pd.DatetimeIndex)
    assert extracted.index.name == "Date"
    assert extracted.index.tolist() == dates.tolist()
    pd.testing.assert_frame_equal(extracted, expected_prices, check_freq=False, check_index_type=False)

    result = prepare_returns(extracted)
    expected_returns = pd.DataFrame(
        {"AAPL": [0.10, 0.01, 0.02], "MSFT": [0.05, 0.03, -0.02]},
        index=pd.DatetimeIndex(["2026-10-02", "2026-10-05", "2026-10-06"], name="Date"),
    )
    pd.testing.assert_frame_equal(result, expected_returns, check_freq=False, check_index_type=False)
    exported = pd.read_csv(StringIO(result.to_csv(index=True)), index_col="Date", parse_dates=True)
    pd.testing.assert_frame_equal(exported, expected_returns, check_freq=False, check_index_type=False)


def test_nonfinite_computed_returns_are_rejected():
    close = close_fixture()
    close["AAPL"] = [1e-308, 1e308, 1e308, 1e308]
    with pytest.raises(DataError, match="non-finite returns"):
        prepare_returns(close)


def test_singular_series_are_left_for_portfolio_validation():
    close = close_fixture()
    close["MSFT"] = close["AAPL"]
    result = prepare_returns(close)
    assert len(result) == 3
    np.testing.assert_array_equal(result["AAPL"], result["MSFT"])


@pytest.mark.parametrize("case", ["empty", "missing-close", "ambiguous-assets", "duplicate-close"])
def test_invalid_download_layouts_are_rejected(case):
    if case == "empty":
        download, tickers = pd.DataFrame(), ["AAPL"]
    elif case == "missing-close":
        download, tickers = close_fixture().rename(columns={"AAPL": "Open"}), ["AAPL"]
    elif case == "ambiguous-assets":
        download, tickers = close_fixture().rename(columns={"AAPL": "Close"}), ["AAPL", "MSFT"]
    else:
        download, tickers = as_download(close_fixture()), ["AAPL", "MSFT"]
        download.columns = pd.MultiIndex.from_tuples([("Close", "AAPL")] * 2)
    with pytest.raises(DataError):
        extract_close_prices(download, tickers)
