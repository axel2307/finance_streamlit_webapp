"""Yahoo IRX discount quotes converted to an estimated simple investment yield.

The 91-day maturity is a proxy assumption, not a known remaining maturity.
No effective annual compounding or historical rate alignment is implied.
"""

from dataclasses import dataclass
from numbers import Number

import numpy as np
import pandas as pd

from finance_data import DataError


INTERVAL_DAYS = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90}
ASSUMED_MATURITY_DAYS = 91


@dataclass(frozen=True)
class IrxQuote:
    discount_quote_percent: float
    annual_simple_yield: float
    as_of: pd.Timestamp


def _invalid(reason):
    return DataError(f"The Treasury bill proxy (^IRX) is unavailable: {reason}. Please try again later.")


def read_irx_quote(data):
    """Read the latest dated, nonmissing Close level; reject ambiguous data."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise _invalid("no quotes were returned")
    if data.columns.has_duplicates:
        raise _invalid("duplicate quote columns")
    if isinstance(data.columns, pd.MultiIndex):
        if data.columns.nlevels != 2:
            raise _invalid("unsupported quote columns")
        levels = [level for level in range(2) if "Close" in data.columns.get_level_values(level)]
        if len(levels) != 1:
            raise _invalid("missing or ambiguous Close column")
        close = data.xs("Close", level=levels[0], axis=1)
        if list(close.columns) != ["^IRX"]:
            raise _invalid("missing or ambiguous ^IRX quote")
        close = close["^IRX"]
    else:
        if "Close" not in data.columns:
            raise _invalid("missing Close column")
        close = data["Close"]
    if not isinstance(close, pd.Series):
        raise _invalid("ambiguous Close column")

    if not isinstance(close.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(close.index.dtype) or any(isinstance(value, Number) for value in close.index):
            raise _invalid("quote dates are row numbers")
        try:
            dates = pd.to_datetime(close.index, errors="raise")
        except (ValueError, TypeError, OverflowError) as exc:
            raise _invalid("invalid quote dates") from exc
        if not isinstance(dates, pd.DatetimeIndex):
            raise _invalid("inconsistent quote dates")
    else:
        dates = close.index.copy()
    if dates.hasnans or dates.has_duplicates:
        raise _invalid("missing or duplicate quote dates")

    if any(isinstance(value, (bool, np.bool_, complex, np.complexfloating)) for value in close):
        raise _invalid("quotes must be real numbers")
    try:
        values = pd.to_numeric(close, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (ValueError, TypeError, OverflowError) as exc:
        raise _invalid("non-numeric quotes") from exc
    available = ~np.isnan(values)
    if not available.any():
        raise _invalid("no valid Close quote")
    quotes = pd.Series(values[available], index=dates[available]).sort_index()
    if not np.isfinite(quotes.to_numpy()).all():
        raise _invalid("non-finite quotes")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        discounts = quotes.to_numpy() / 100
        denominators = 360 - ASSUMED_MATURITY_DAYS * discounts
        yields = 365 * (discounts / denominators)
    if (denominators <= 0).any() or not np.isfinite(denominators).all() or not np.isfinite(yields).all():
        raise _invalid("the discount quote cannot produce a finite positive bill price")
    return IrxQuote(float(quotes.iloc[-1]), float(yields[-1]), quotes.index[-1])


def interval_risk_free_rate(annual_simple_yield, interval):
    """Prorate a simple annual yield; do not treat it as an effective yield."""
    if interval not in INTERVAL_DAYS:
        raise DataError("The risk-free proxy does not support this return interval.")
    if isinstance(annual_simple_yield, (bool, np.bool_)):
        raise DataError("The risk-free proxy must be a finite simple annual yield.")
    try:
        annual = float(annual_simple_yield)
        rate = annual * (INTERVAL_DAYS[interval] / 365)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataError("The risk-free proxy must be a finite simple annual yield.") from exc
    if not np.isfinite(annual) or not np.isfinite(rate):
        raise DataError("The risk-free proxy must be a finite simple annual yield.")
    return rate
