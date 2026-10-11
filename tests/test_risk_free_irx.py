"""Offline checks for the quoted 13-week bill proxy and simple interval units."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from finance_data import DataError
from risk_free_irx_v1 import IrxQuote, interval_risk_free_rate, read_irx_quote


def irx_frame(values=(4.0,), dates=None, layout="plain"):
    if dates is None:
        dates = pd.date_range("2026-10-05", periods=len(values), name="Date")
    result = pd.DataFrame({"Close": list(values)}, index=dates)
    if layout == "price-ticker":
        result.columns = pd.MultiIndex.from_tuples(
            [("Close", "^IRX")], names=["Price", "Ticker"],
        )
    elif layout == "ticker-price":
        result.columns = pd.MultiIndex.from_tuples(
            [("^IRX", "Close")], names=["Ticker", "Price"],
        )
    return result


@pytest.mark.parametrize("layout", ["plain", "price-ticker", "ticker-price"])
def test_four_percent_quote_is_a_level_and_becomes_annual_simple_yield(layout):
    # The prior quote is deliberately far from 4: a percentage-change reader
    # would report -96%, while the requested input is the latest level of 4%.
    data = irx_frame((100.0, 4.0), layout=layout)
    original = data.copy(deep=True)
    quote = read_irx_quote(data)

    assert isinstance(quote, IrxQuote)
    assert quote.discount_quote_percent == pytest.approx(4.0)
    # Independent exact rational for a 4% discount quote and assumed 91 days.
    assert quote.annual_simple_yield == pytest.approx(365 / 8909, rel=1e-13)
    assert quote.annual_simple_yield * 100 == pytest.approx(4.097, abs=0.0001)
    assert quote.as_of == pd.Timestamp("2026-10-06")
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("layout", ["plain", "price-ticker", "ticker-price"])
def test_other_yahoo_price_fields_do_not_replace_the_close_quote(layout):
    data = irx_frame(layout=layout)
    for field, value in (("Open", 3.8), ("High", 4.2), ("Low", 3.7), ("Volume", 0)):
        column = (field, "^IRX") if layout == "price-ticker" else ("^IRX", field)
        data[field if layout == "plain" else column] = value
    quote = read_irx_quote(data)
    assert quote.discount_quote_percent == pytest.approx(4.0)
    assert quote.annual_simple_yield == pytest.approx(365 / 8909, rel=1e-13)


def test_latest_valid_quote_is_selected_chronologically_without_filling():
    data = irx_frame(
        (4.0, np.nan, 4.5, 3.9),
        pd.DatetimeIndex(["2026-10-07", "2026-10-09", "2026-10-06", "2026-10-08"]),
    )
    original = data.copy(deep=True)
    quote = read_irx_quote(data)
    assert quote.discount_quote_percent == pytest.approx(3.9)
    assert quote.as_of == pd.Timestamp("2026-10-08")
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("dates", [
    pd.Index(["2026-10-05", "2026-10-06"], dtype=object),
    pd.Index(["2026-10-05", "2026-10-06"], dtype="string"),
    pd.Index([date(2026, 10, 5), date(2026, 10, 6)], dtype=object),
    pd.DatetimeIndex(["2026-10-05", "2026-10-06"], tz="America/New_York"),
])
def test_valid_date_representations_retain_the_observation_date(dates):
    quote = read_irx_quote(irx_frame((4.1, 4.0), dates))
    assert isinstance(quote.as_of, pd.Timestamp)
    assert quote.as_of == pd.Timestamp(dates[-1])


@pytest.mark.parametrize("value,expected", [(0.0, 0.0), (-1.0, -365 / 36091)])
def test_zero_and_negative_quotes_are_not_rejected_as_equity_prices(value, expected):
    quote = read_irx_quote(irx_frame((value,)))
    assert quote.discount_quote_percent == pytest.approx(value)
    assert quote.annual_simple_yield == pytest.approx(expected, rel=1e-13)


def test_extreme_negative_quote_does_not_overflow_when_the_final_yield_is_finite():
    quote = read_irx_quote(irx_frame((-1e308,)))
    assert quote.discount_quote_percent == -1e308
    assert np.isfinite(quote.annual_simple_yield)
    # At this magnitude, the 360-day term is below float precision and the
    # result approaches the finite limit -365/91; an intermediate 365*d overflows.
    assert quote.annual_simple_yield == pytest.approx(-365 / 91, rel=1e-14)


@pytest.mark.parametrize("interval,expected", [
    ("1d", 1 / 8909),
    ("5d", 5 / 8909),
    ("1mo", 30 / 8909),
    ("3mo", 90 / 8909),
])
def test_four_percent_proxy_uses_simple_interest_for_each_supported_interval(interval, expected):
    quote = read_irx_quote(irx_frame())
    assert interval_risk_free_rate(quote.annual_simple_yield, interval) == pytest.approx(
        expected, rel=1e-13,
    )


@pytest.mark.parametrize("annual", [0.0, -0.01, 0.2])
def test_interval_conversion_is_linear_and_keeps_signed_rates(annual):
    daily = interval_risk_free_rate(annual, "1d")
    quarterly = interval_risk_free_rate(annual, "3mo")
    assert daily == pytest.approx(annual / 365, rel=1e-13)
    assert quarterly == pytest.approx(90 * daily, rel=1e-13)
    if annual:
        # Treating the annual simple estimate as an effective annual yield
        # would compound and produce a different interval rate.
        assert not np.isclose(quarterly, (1 + annual) ** (90 / 365) - 1, rtol=1e-5, atol=0)


@pytest.mark.parametrize("annual", [np.nan, np.inf, -np.inf, "bad", None])
def test_invalid_annual_yield_is_rejected_instead_of_becoming_zero(annual):
    with pytest.raises(DataError) as error:
        interval_risk_free_rate(annual, "1d")
    assert str(error.value).strip()


@pytest.mark.parametrize("interval", ["1wk", "", None])
def test_unsupported_intervals_are_rejected(interval):
    with pytest.raises(DataError) as error:
        interval_risk_free_rate(0.04, interval)
    assert str(error.value).strip()


@pytest.mark.parametrize("data", [None, pd.DataFrame(), pd.Series([4.0]), irx_frame((np.nan, np.nan))])
def test_missing_quote_data_has_no_silent_zero_fallback(data):
    with pytest.raises(DataError) as error:
        read_irx_quote(data)
    assert str(error.value).strip()


@pytest.mark.parametrize("invalid", [
    "bad", True, False, 4 + 1j, np.inf, -np.inf,
    100 * 360 / 91, 396.0, np.finfo(float).max,
])
@pytest.mark.parametrize("position", [0, 1])
def test_invalid_present_quotes_are_rejected_even_when_another_row_is_valid(invalid, position):
    values = [4.0, 4.1]
    values[position] = invalid
    with pytest.raises(DataError) as error:
        read_irx_quote(irx_frame(values))
    assert str(error.value).strip()


@pytest.mark.parametrize("dates", [
    pd.RangeIndex(2),
    pd.Index([1.0, 2.0]),
    pd.Index([1, 2], dtype=object),
    pd.Index([pd.Timestamp("2026-10-05"), 2], dtype=object),
    pd.Index(["bad", "2026-10-06"]),
    pd.DatetimeIndex(["2026-10-05", "2026-10-05"]),
    pd.DatetimeIndex(["2026-10-05", pd.NaT]),
    pd.MultiIndex.from_tuples([("2026-10-05", "a"), ("2026-10-06", "b")]),
])
def test_invalid_missing_duplicate_or_numeric_dates_are_rejected(dates):
    with pytest.raises(DataError) as error:
        read_irx_quote(irx_frame((4.1, 4.0), dates))
    assert str(error.value).strip()


@pytest.mark.parametrize("columns", [
    pd.Index(["Open"]),
    pd.Index(["Close", "Close"]),
    pd.MultiIndex.from_tuples([("Close", "^TNX")]),
    pd.MultiIndex.from_tuples([("Close", "^IRX"), ("Close", "^TNX")]),
    pd.MultiIndex.from_tuples([("Close", "^IRX"), ("Close", "^IRX")]),
    pd.MultiIndex.from_tuples([("Close", "^IRX"), ("^IRX", "Close")]),
    pd.MultiIndex.from_tuples([("Close", "^IRX", "unexpected")]),
])
def test_missing_or_ambiguous_quote_columns_are_rejected(columns):
    data = pd.DataFrame(
        [[4.0] * len(columns)], index=pd.DatetimeIndex(["2026-10-05"]), columns=columns,
    )
    with pytest.raises(DataError) as error:
        read_irx_quote(data)
    assert str(error.value).strip()
