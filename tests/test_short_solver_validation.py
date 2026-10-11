"""A solver success flag cannot bypass the signed-domain certificate."""

from types import SimpleNamespace

import numpy as np
import pytest

import portfolio_modes_v1 as portfolio
from portfolio import PortfolioError, optimize_portfolio


@pytest.mark.parametrize("failure", ["failed", "nonfinite", "wrong-budget"])
def test_short_solver_failures_do_not_produce_results(monkeypatch, failure):
    real_minimize = portfolio.minimize

    def broken(function, initial, **kwargs):
        result = real_minimize(function, initial, **kwargs)
        if failure == "failed":
            return SimpleNamespace(success=False)
        if failure == "nonfinite":
            result.x[0] = np.nan
        else:
            result.x *= 2
            result.fun = function(result.x)
        result.success = True
        return result

    monkeypatch.setattr(portfolio, "minimize", broken)
    with pytest.raises(PortfolioError, match="optimizer could not"):
        optimize_portfolio([0.001, 0.002], np.diag([0.0001, 0.0004]), 0, points=3, allow_short=True)


@pytest.mark.parametrize("stage", [1, 2])
def test_short_feasible_but_suboptimal_success_is_rejected(monkeypatch, stage):
    real_minimize = portfolio.minimize
    calls = 0

    def false_success(function, initial, **kwargs):
        nonlocal calls
        calls += 1
        result = real_minimize(function, initial, **kwargs)
        if calls == stage:
            # Both net-one and positive-Sharpe lifts admit a pure first-asset
            # position. Solve only its two equality coefficients to obtain a
            # feasible, deliberately suboptimal candidate in solver units.
            equality = kwargs["constraints"]["jac"](initial)
            targets = equality @ initial
            result.x = np.zeros_like(initial)
            selected = [0, len(initial) // 2]
            result.x[selected] = np.linalg.solve(equality[:, selected], targets)
            assert np.min(result.x) >= 0
            np.testing.assert_allclose(equality @ result.x, targets, atol=1e-12)
            result.fun = function(result.x)
            result.success = True
        return result

    monkeypatch.setattr(portfolio, "minimize", false_success)
    with pytest.raises(PortfolioError, match="optimizer could not"):
        optimize_portfolio([0.001, 0.002], np.diag([0.0001, 0.0004]), 0, points=3, allow_short=True)


def test_short_three_asset_heterogeneous_covariance_matches_closed_form():
    variances = np.array([1e-10, 1e-8, 1.0])
    inverse = 1 / variances
    expected = inverse / inverse.sum()
    result = optimize_portfolio([0.001] * 3, np.diag(variances), 0, allow_short=True)
    np.testing.assert_allclose(result.min_weights, expected, rtol=3e-6, atol=1e-14)
    np.testing.assert_allclose(result.weights, expected, rtol=3e-6, atol=1e-14)
    assert result.min_risk == pytest.approx(np.sqrt(1 / inverse.sum()), rel=3e-6)
    assert result.frontier_weights.shape == (1, 3)
