"""Independent arithmetic and portfolio regressions for annual return units."""

import numpy as np
import pandas as pd
import pytest

from annual_moments_v1 import AnnualMoments, annualize_returns
from finance_data import DataError
from portfolio_modes_v1 import optimize_portfolio, portfolio_metrics


FREQUENCIES = [("1d", 252), ("1wk", 52), ("1mo", 12), ("3mo", 4)]


def observed_returns():
    # The order is intentional. The sample covariance is independently known:
    # [[.0004, -.0001], [-.0001, .0001]], with means [.01, .02].
    return pd.DataFrame(
        {"MSFT": [-0.01, 0.01, 0.03], "AAPL": [0.03, 0.01, 0.02]},
        index=pd.DatetimeIndex(["2026-10-05", "2026-10-06", "2026-10-07"], name="Date"),
    )


@pytest.mark.parametrize("interval,periods", FREQUENCIES)
def test_arithmetic_mean_and_sample_covariance_are_annualized_in_original_asset_order(interval, periods):
    returns = observed_returns()
    original = returns.copy(deep=True)
    moments = annualize_returns(returns, interval)

    assert isinstance(moments, AnnualMoments)
    assert isinstance(moments.periods_per_year, int)
    assert moments.periods_per_year == periods
    assert isinstance(moments.mean_returns, np.ndarray)
    assert isinstance(moments.covariance, np.ndarray)
    assert moments.mean_returns.shape == (2,)
    assert moments.covariance.shape == (2, 2)
    np.testing.assert_allclose(moments.mean_returns, [0.01 * periods, 0.02 * periods], rtol=1e-13)
    expected_covariance = np.array([[0.0004, -0.0001], [-0.0001, 0.0001]]) * periods
    np.testing.assert_allclose(moments.covariance, expected_covariance, rtol=1e-13)
    np.testing.assert_allclose(
        np.sqrt(np.diag(moments.covariance)), np.array([0.02, 0.01]) * np.sqrt(periods), rtol=1e-13,
    )
    pd.testing.assert_frame_equal(returns, original)


def test_daily_example_has_annual_return_volatility_and_sharpe_with_annual_risk_free():
    # Mean=.001 and sample standard deviation=.01, exactly in decimal units.
    returns = pd.DataFrame({"Asset": [-0.009, 0.001, 0.011]})
    moments = annualize_returns(returns, "1d")
    measured = portfolio_metrics([1.0], moments.mean_returns, moments.covariance, 0.04)
    assert moments.covariance.shape == (1, 1)
    assert measured[0] == pytest.approx(0.252, rel=1e-13)
    assert measured[1] == pytest.approx(0.15874507866387544, rel=1e-13)
    assert measured[2] == pytest.approx(1.3354744712992694, rel=1e-13)
    assert measured[2] == pytest.approx(1.3355, abs=0.00005)


def test_two_observations_use_sample_not_population_covariance():
    moments = annualize_returns(pd.DataFrame({"Asset": [-0.01, 0.01]}), "1mo")
    np.testing.assert_allclose(moments.mean_returns, [0.0], atol=1e-15)
    np.testing.assert_allclose(moments.covariance, [[0.0024]], rtol=1e-13)


def test_zero_variance_is_left_for_portfolio_risk_validation():
    moments = annualize_returns(pd.DataFrame({"Asset": [0.001, 0.001, 0.001]}), "1wk")
    np.testing.assert_allclose(moments.mean_returns, [0.052], rtol=1e-13)
    np.testing.assert_allclose(moments.covariance, [[0.0]], atol=1e-25)


@pytest.mark.parametrize("interval", ["5d", "", "1h", None])
def test_retired_five_day_and_unknown_intervals_are_rejected(interval):
    with pytest.raises(DataError) as error:
        annualize_returns(observed_returns(), interval)
    assert str(error.value).strip()


@pytest.mark.parametrize("returns", [
    None,
    [[0.01], [0.02]],
    pd.Series([0.01, 0.02]),
    pd.DataFrame(),
    pd.DataFrame(index=[0, 1]),
    pd.DataFrame({"Asset": []}),
    pd.DataFrame({"Asset": [0.01]}),
])
def test_missing_or_insufficient_return_matrix_has_an_explanatory_error(returns):
    with pytest.raises(DataError) as error:
        annualize_returns(returns, "1d")
    assert str(error.value).strip()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, "bad", True, 0.01 + 0.02j])
def test_missing_or_invalid_observations_are_not_silently_dropped(invalid):
    returns = observed_returns().astype(object)
    returns.iloc[0, 0] = invalid
    original = returns.copy(deep=True)
    with pytest.raises(DataError) as error:
        annualize_returns(returns, "1d")
    assert str(error.value).strip()
    pd.testing.assert_frame_equal(returns, original)


def test_duplicate_asset_columns_are_rejected():
    returns = observed_returns()
    returns.columns = ["Asset", "Asset"]
    with pytest.raises(DataError) as error:
        annualize_returns(returns, "1d")
    assert str(error.value).strip()


def test_nonfinite_computed_moments_are_rejected():
    returns = pd.DataFrame({"Asset": [-1e308, 0.0, 1e308]})
    with pytest.raises(DataError) as error:
        annualize_returns(returns, "1d")
    assert str(error.value).strip()


@pytest.mark.parametrize("interval,periods", FREQUENCIES)
@pytest.mark.parametrize("allow_short", [False, True], ids=["long-only", "short-allowed"])
@pytest.mark.parametrize("means,covariance,risk_free", [
    pytest.param(
        [0.001, 0.0005], [[0.0001, 0.00018], [0.00018, 0.0004]], 0.0001,
        id="positive-excess-and-short-position",
    ),
    pytest.param([0.001, 0.001], [[0.0001, 0], [0, 0.0004]], 0.001, id="all-excess-zero"),
    pytest.param([0.001, 0.002], [[0.0001, 0], [0, 0.0004]], 0.01, id="all-feasible-excess-negative"),
])
def test_scaling_all_inputs_to_annual_units_preserves_weights_and_scales_sharpe_and_frontier(
    interval, periods, allow_short, means, covariance, risk_free,
):
    means, covariance = np.array(means), np.array(covariance)
    options = {"points": 5, "allow_short": allow_short, "gross_limit": 2.0}
    baseline = optimize_portfolio(means, covariance, risk_free, **options)
    annual = optimize_portfolio(means * periods, covariance * periods, risk_free * periods, **options)

    assert annual.all_excess_returns_nonpositive == baseline.all_excess_returns_nonpositive
    assert annual.allow_short == baseline.allow_short == allow_short
    assert annual.gross_limit == baseline.gross_limit
    for name in ("weights", "min_weights", "frontier_weights"):
        np.testing.assert_allclose(getattr(annual, name), getattr(baseline, name), atol=2e-6, rtol=2e-6)
    for name in ("max_return", "min_return", "frontier_returns"):
        np.testing.assert_allclose(getattr(annual, name), getattr(baseline, name) * periods, atol=1e-10, rtol=3e-6)
    for name in ("max_risk", "min_risk", "frontier_risks"):
        np.testing.assert_allclose(getattr(annual, name), getattr(baseline, name) * np.sqrt(periods), atol=1e-10, rtol=3e-6)
    assert annual.max_sharpe == pytest.approx(baseline.max_sharpe * np.sqrt(periods), rel=3e-6, abs=1e-10)
    if allow_short and risk_free == 0.0001:
        assert annual.weights[1] < -0.1, "The scaling regression must exercise a real short position"


@pytest.mark.parametrize("allow_short", [False, True], ids=["long-only", "short-allowed"])
def test_annual_rf_legitimately_changes_weights_from_previous_calendar_day_conversion(allow_short):
    # The prior app prorated the annual RF by 365 calendar days. Arithmetic
    # daily moments now use 252 observations/year, so unchanged weights are
    # not a valid expectation when the same annual 4% quote is retained.
    means = np.array([0.0003, 0.0006])
    covariance = np.diag([0.0001, 0.0004])
    annual_rf = 0.04
    options = {"points": 3, "allow_short": allow_short, "gross_limit": 2.0}
    previous = optimize_portfolio(means, covariance, annual_rf / 365, **options)
    current = optimize_portfolio(means * 252, covariance * 252, annual_rf, **options)
    # Independent interior tangent solutions: normalize inverse-covariance
    # times excess means; these exact fractions differ for the two RF bases.
    np.testing.assert_allclose(previous.weights, [278 / 457, 179 / 457], atol=2e-6)
    np.testing.assert_allclose(current.weights, [178 / 317, 139 / 317], atol=2e-6)
    assert previous.weights[0] - current.weights[0] > 0.04
