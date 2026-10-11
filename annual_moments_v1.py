"""Annual arithmetic moments estimated from unmodified periodic simple returns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance_data import DataError


PERIODS_PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12, "3mo": 4}
FREQUENCY_LABELS = {
    "1d": "Daily (252/year)",
    "1wk": "Weekly (52/year)",
    "1mo": "Monthly (12/year)",
    "3mo": "Quarterly (4/year)",
}


@dataclass(frozen=True)
class AnnualMoments:
    mean_returns: np.ndarray
    covariance: np.ndarray
    periods_per_year: int


def annualize_returns(returns, interval):
    """Scale sample mean and covariance by N, without annualizing each row.

    Volatility consequently uses the square-root-of-time convention. These are
    in-sample arithmetic estimates, not a forecast or compounded wealth path.
    """
    if interval not in PERIODS_PER_YEAR:
        raise DataError("Choose a supported return frequency: daily, weekly, monthly or quarterly.")
    if (not isinstance(returns, pd.DataFrame) or returns.shape[0] < 2
            or returns.shape[1] < 1 or returns.columns.has_duplicates):
        raise DataError("Annual estimates require at least two returns and unambiguous asset columns.")
    if (any(pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in returns.dtypes)
            or any(isinstance(value, (bool, np.bool_, complex, np.complexfloating)) for value in returns.to_numpy().flat)):
        raise DataError("Annual estimates require finite real simple returns.")
    try:
        values = returns.to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataError("Annual estimates require finite real simple returns.") from exc
    if not np.isfinite(values).all():
        raise DataError("Annual estimates require finite real simple returns.")
    periods = PERIODS_PER_YEAR[interval]
    with np.errstate(over="ignore", invalid="ignore"):
        means = values.mean(axis=0) * periods
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1)) * periods
    if not np.isfinite(means).all() or not np.isfinite(covariance).all():
        raise DataError("The selected returns cannot produce finite annual estimates.")
    return AnnualMoments(means, covariance, periods)
