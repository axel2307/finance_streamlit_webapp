"""Parse market directories and prepare dated prices without filling missing data."""

from io import BytesIO, StringIO

import numpy as np
import pandas as pd


class DataError(ValueError):
    """Market data cannot support the requested analysis."""


EXCHANGE_LABELS = {
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "BATS",
    "A": "NYSE MKT",
    "V": "IEX",
}


def _read_directory(text, required_columns, source):
    if isinstance(text, str):
        text = StringIO(text)
    elif isinstance(text, bytes):
        text = BytesIO(text)
    elif not hasattr(text, "read"):
        raise DataError(f"The {source} directory must be text or an in-memory stream.")
    try:
        frame = pd.read_csv(
            text, sep="|", dtype=str, keep_default_na=False, na_values=[""],
            encoding="utf-8-sig",
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise DataError(f"The {source} directory could not be read. Try again later.") from exc
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise DataError(f"The {source} directory is missing columns: {', '.join(missing)}.")
    for column in required_columns:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    return frame


def parse_catalog(nasdaq_text, other_text):
    """Read directory text/bytes/streams; preserve tickers such as ``NA``.

    ``Exchange Code`` retains the other-listed source code. It is missing for
    the Nasdaq file, which does not contain that field. Unknown codes remain
    visible in labels instead of being assigned to a known exchange.
    """
    nasdaq = _read_directory(
        nasdaq_text,
        ["Symbol", "Security Name", "ETF", "Test Issue", "Financial Status"],
        "Nasdaq",
    )
    other = _read_directory(
        other_text,
        ["NASDAQ Symbol", "Security Name", "ETF", "Test Issue", "Exchange"],
        "other-listed",
    )
    nasdaq = nasdaq.loc[
        (nasdaq["Test Issue"] == "N") & (nasdaq["Financial Status"] == "N"),
        ["Symbol", "Security Name", "ETF"],
    ].copy()
    nasdaq["Exchange"] = "NASDAQ"
    nasdaq["Exchange Code"] = pd.NA

    other = other.loc[
        other["Test Issue"] == "N",
        ["NASDAQ Symbol", "Security Name", "ETF", "Exchange"],
    ].rename(columns={"NASDAQ Symbol": "Symbol", "Exchange": "Exchange Code"})
    other["Exchange"] = other["Exchange Code"].map(
        lambda code: EXCHANGE_LABELS.get(code, f"Unknown ({code})")
        if pd.notna(code) else "Unknown (missing)"
    )
    columns = ["Symbol", "Security Name", "ETF", "Exchange", "Exchange Code"]
    catalog = pd.concat([nasdaq[columns], other[columns]], ignore_index=True)
    catalog = catalog.dropna(subset=["Symbol", "Security Name"]).reset_index(drop=True)
    if catalog.empty:
        raise DataError("The asset directory has no valid assets. Try again later.")
    return catalog


def _validated_prices(close):
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise DataError("No closing prices are available for the selected assets and period.")
    if close.columns.has_duplicates:
        raise DataError("Closing prices contain duplicate asset columns.")
    if not isinstance(close.index, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(close.index.dtype):
            raise DataError("Closing prices must have dates, not row numbers.")
        try:
            dates = pd.to_datetime(close.index, errors="raise")
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataError("Closing prices contain invalid dates.") from exc
        if not isinstance(dates, pd.DatetimeIndex):
            raise DataError("Closing prices must have a single consistent date index.")
    else:
        dates = close.index.copy()
    if dates.hasnans or dates.has_duplicates:
        raise DataError("Closing prices contain missing or duplicate dates.")

    numeric = pd.DataFrame(index=dates.rename("Date"))
    missing_assets = []
    for symbol in close.columns:
        try:
            values = pd.to_numeric(close[symbol], errors="raise").to_numpy(
                dtype=float, na_value=np.nan,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataError(f"Closing prices for {symbol} contain non-numeric values.") from exc
        available = values[~np.isnan(values)]
        if not len(available):
            missing_assets.append(str(symbol))
        elif not np.isfinite(available).all() or (available <= 0).any():
            raise DataError(f"Closing prices for {symbol} must be finite and positive.")
        numeric[symbol] = values
    if missing_assets:
        raise DataError(f"No closing prices are available for: {', '.join(missing_assets)}.")
    return numeric.sort_index()


def extract_close_prices(download, tickers):
    """Extract selected closes in request order from yfinance column layouts.

    Missing observations stay missing. Missing/all-NaN assets, ambiguous
    columns, invalid dates and invalid available prices raise ``DataError``.
    """
    tickers = list(tickers)
    if not tickers:
        raise DataError("Select at least one asset first.")
    if len(tickers) != len(set(tickers)):
        raise DataError("Select each asset only once.")
    if not isinstance(download, pd.DataFrame) or download.empty:
        raise DataError(f"No price data is available for: {', '.join(tickers)}.")
    if isinstance(download.columns, pd.MultiIndex):
        if download.columns.nlevels != 2:
            raise DataError("The price download has an unsupported column layout.")
        price_levels = [
            level for level in range(2)
            if "Close" in download.columns.get_level_values(level)
        ]
        if len(price_levels) != 1:
            raise DataError("The price download is missing an unambiguous Close column.")
        close = download.xs("Close", level=price_levels[0], axis=1)
        if close.columns.has_duplicates:
            raise DataError("The price download contains duplicate closing-price columns.")
        missing = [ticker for ticker in tickers if ticker not in close.columns]
        if missing:
            raise DataError(f"No closing prices are available for: {', '.join(missing)}.")
        close = close.loc[:, tickers].copy()
    else:
        if len(tickers) != 1:
            raise DataError("The price download does not identify the selected assets separately.")
        if "Close" not in download.columns:
            raise DataError(f"No closing prices are available for: {tickers[0]}.")
        if download.columns.has_duplicates:
            raise DataError("The price download contains duplicate columns.")
        close = download.loc[:, ["Close"]].rename(columns={"Close": tickers[0]})
    return _validated_prices(close)


def prepare_returns(close):
    """Compute common observed returns without filling gaps or dropping valid rows.

    At least two finite common returns are required here. Covariance rank and
    numerical conditioning are checked by the portfolio calculation layer.
    """
    close = _validated_prices(close)
    returns = close.pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise DataError("No common returns are available. Choose assets with overlapping histories or a longer period.")
    if len(returns) < 2:
        raise DataError("At least two common returns are required. Choose a longer period or a shorter interval.")
    if not np.isfinite(returns.to_numpy()).all():
        raise DataError("The selected prices produce non-finite returns; analysis cannot continue.")
    return returns
