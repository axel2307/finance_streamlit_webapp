"""Financial regressions with controlled data; no provider calls."""

from types import SimpleNamespace
from itertools import permutations

import numpy as np
import pytest

import portfolio
from portfolio import PortfolioError, optimize_portfolio, portfolio_metrics


def assert_feasible(weights):
    assert np.isfinite(weights).all()
    assert np.min(weights) >= 0
    assert np.max(weights) <= 1
    assert np.sum(weights) == pytest.approx(1)


def test_negative_excess_refutes_old_formula_and_selects_best_vertex():
    means = np.array([0.0001, 0.0002])
    cov = np.diag([0.0001, 0.0001])
    result = optimize_portfolio(means, cov, 0.0003, points=7)
    np.testing.assert_allclose(result.weights, [0, 1])
    assert result.max_sharpe == pytest.approx(-0.01)
    assert result.all_excess_returns_nonpositive
    old_sharpe = portfolio_metrics([2 / 3, 1 / 3], means, cov, 0.0003)[2]
    assert result.max_sharpe > old_sharpe
    for first_weight in np.linspace(0, 1, 101):
        candidate = portfolio_metrics([first_weight, 1 - first_weight], means, cov, 0.0003)[2]
        assert candidate <= result.max_sharpe + 1e-12


def test_zero_excess_selects_zero_sharpe_vertex():
    result = optimize_portfolio([0.001, 0.002], np.diag([0.01, 0.02]), 0.002, points=3)
    np.testing.assert_array_equal(result.weights, [0, 1])
    assert result.max_sharpe == 0


def test_one_shared_long_only_solution_replaces_short_table_case():
    means = [0.001, 0.0005]
    cov = [[0.0001, 0.00018], [0.00018, 0.0004]]
    result = optimize_portfolio(means, cov, 0.0001, points=9)
    np.testing.assert_allclose(result.weights, [1, 0], atol=1e-7)
    assert result.max_sharpe == pytest.approx(0.09)
    assert portfolio_metrics(result.weights, means, cov, 0.0001) == pytest.approx(
        (result.max_return, result.max_risk, result.max_sharpe)
    )
    for weights in result.frontier_weights:
        assert_feasible(weights)


def test_positive_sharpe_matches_known_interior_solution():
    means = np.array([0.001, 0.002])
    cov = np.diag([0.0001, 0.0004])
    excess = means - 0.0001
    expected = np.linalg.solve(cov, excess)
    expected /= expected.sum()
    result = optimize_portfolio(means, cov, 0.0001, points=5)
    np.testing.assert_allclose(result.weights, expected, atol=2e-6)
    assert not result.all_excess_returns_nonpositive


def test_nonpositive_excess_asset_can_still_provide_a_hedge():
    means = [0.02, -0.001]
    cov = [[0.04, -0.015], [-0.015, 0.01]]
    result = optimize_portfolio(means, cov, 0, points=4)
    assert result.weights[1] > 0.1
    assert result.max_sharpe > portfolio_metrics([1, 0], means, cov, 0)[2]


@pytest.mark.parametrize("mean", [0, 0.001, -0.001])
def test_equal_means_frontier_is_only_minimum_variance(mean):
    result = optimize_portfolio([mean, mean], np.diag([0.0001, 0.0004]), 0.0001)
    assert result.frontier_weights.shape == (1, 2)
    np.testing.assert_allclose(result.min_weights, [0.8, 0.2], atol=1e-7)
    assert result.frontier_risks[0] == pytest.approx(np.sqrt(0.00008))
    assert result.frontier_returns[0] == pytest.approx(mean)
    assert result.frontier_risks[0] < np.sqrt(0.000125)


def test_near_equal_means_do_not_collapse_frontier():
    means = [0.001, 0.001 + 1e-12]
    result = optimize_portfolio(means, np.diag([0.0001, 0.0004]), 0.0001, points=5)
    assert len(result.frontier_returns) == 5
    np.testing.assert_allclose(result.frontier_weights[-1], [0, 1])
    assert np.all(np.diff(result.frontier_returns) >= 0)
    assert np.all(np.diff(result.frontier_risks) >= -1e-10)


def test_tied_maximum_return_endpoint_uses_its_minimum_variance_mix():
    result = optimize_portfolio([0.001, 0.002, 0.002], np.diag([0.01, 0.01, 0.04]), 0, points=5)
    np.testing.assert_allclose(result.frontier_weights[-1], [0, 0.8, 0.2], atol=1e-7)


@pytest.mark.parametrize("factor", [1e-10, 1e10])
def test_return_unit_changes_preserve_weights_and_sharpe(factor):
    means = np.array([0.001, 0.002])
    cov = np.diag([0.0001, 0.0004])
    baseline = optimize_portfolio(means, cov, 0.0001, points=5)
    scaled = optimize_portfolio(means * factor, cov * factor**2, 0.0001 * factor, points=5)
    np.testing.assert_allclose(scaled.weights, baseline.weights, atol=2e-6)
    np.testing.assert_allclose(scaled.frontier_weights, baseline.frontier_weights, atol=2e-6)
    assert scaled.max_sharpe == pytest.approx(baseline.max_sharpe)


@pytest.mark.parametrize("cov", [
    [[0.0001, 0.0001], [0.0001, 0.0001]],
    [[0.0001, 0.0002], [0.0002, 0.0001]],
    [[0, 0], [0, 0]],
])
def test_singular_or_indefinite_covariance_has_functional_error(cov):
    with pytest.raises(PortfolioError, match="independent risk"):
        optimize_portfolio([0.001, 0.002], cov, 0.0001)


@pytest.mark.parametrize("cov", [
    np.cov([[0.01, 0.03], [0.02, 0.01]], rowvar=False),
    [[1, 1 - 1e-16], [1 - 1e-16, 1]],
])
def test_roundoff_does_not_make_rank_deficient_covariance_usable(cov):
    with pytest.raises(PortfolioError, match="independent risk"):
        optimize_portfolio([0.001, 0.002], cov, 0.0001)


@pytest.mark.parametrize("means,cov,rf", [
    ([np.nan, 0.002], np.eye(2), 0),
    ([0.001, 0.002], [[1, np.inf], [np.inf, 1]], 0),
    ([0.001, 0.002], np.eye(2), np.nan),
    ([0.001, 0.002], [[1, 0.5], [0.1, 1]], 0),
    ([0.001, 0.002], np.eye(3), 0),
])
def test_invalid_financial_inputs_are_rejected(means, cov, rf):
    with pytest.raises(PortfolioError):
        optimize_portfolio(means, cov, rf)


@pytest.mark.parametrize("failure", [
    {"success": False, "x": [0.8, 0.2], "fun": 0.1},
    {"success": True, "x": [np.nan, 0.2], "fun": 0.1},
    {"success": True, "x": [0.8, 0.2], "fun": np.nan},
    {"success": True, "x": [0.2, 0.2], "fun": 0.1},
    {"success": True, "x": [-0.1, 1.1], "fun": 0.1},
    {"success": True, "x": [0.8, 0.2, 0], "fun": 0.1},
])
def test_solver_failure_or_invalid_result_never_publishes_metrics(monkeypatch, failure):
    monkeypatch.setattr(portfolio, "minimize", lambda *args, **kwargs: SimpleNamespace(**failure))
    with pytest.raises(PortfolioError, match="optimizer could not"):
        optimize_portfolio([0.001, 0.002], np.diag([0.0001, 0.0004]), 0.0001)


def test_success_flag_cannot_hide_frontier_return_constraint_violation(monkeypatch):
    real_minimize = portfolio.minimize
    def wrong_frontier(function, initial, **kwargs):
        result = real_minimize(function, initial, **kwargs)
        if len(kwargs["constraints"]["fun"](initial)) == 2:
            result.x = np.array([1.0, 0.0])
            result.success = True
        return result
    monkeypatch.setattr(portfolio, "minimize", wrong_frontier)
    with pytest.raises(PortfolioError, match="optimizer could not"):
        optimize_portfolio([0.001, 0.002], np.diag([0.0001, 0.0004]), 0.0001, points=4)


def test_positive_sharpe_equality_is_validated_before_normalizing(monkeypatch):
    real_minimize = portfolio.minimize
    def wrong_scale(function, initial, **kwargs):
        result = real_minimize(function, initial, **kwargs)
        if kwargs["bounds"][0][1] is None:
            result.x *= 2
        return result
    monkeypatch.setattr(portfolio, "minimize", wrong_scale)
    with pytest.raises(PortfolioError, match="optimizer could not"):
        optimize_portfolio([0.001, 0.002], np.diag([0.0001, 0.0004]), 0.0001, points=4)


@pytest.mark.parametrize("stage", ["minimum_variance", "positive_sharpe"])
def test_feasible_but_suboptimal_solver_success_is_rejected(monkeypatch, stage):
    real_minimize = portfolio.minimize
    def false_success(function, initial, **kwargs):
        result = real_minimize(function, initial, **kwargs)
        is_positive_sharpe = kwargs["bounds"][0][1] is None
        if is_positive_sharpe == (stage == "positive_sharpe"):
            # Construct a feasible but suboptimal vertex in the solver's
            # transformed coordinates; its initial point may already be exact.
            equality = kwargs["constraints"]["jac"](initial)
            target = equality @ initial
            result.x = np.zeros_like(initial)
            result.x[0] = target[0] / equality[0, 0]
            result.fun = function(result.x)
            result.success = True
        return result
    monkeypatch.setattr(portfolio, "minimize", false_success)
    with pytest.raises(PortfolioError, match="optimizer could not"):
        optimize_portfolio([0.001, 0.002], np.diag([0.0001, 0.0004]), 0.0001, points=4)


@pytest.mark.parametrize("weights", [[-0.1, 1.1], [0.5, 0.4], [np.nan, 0.5]])
def test_metrics_reject_weights_outside_shared_long_only_policy(weights):
    with pytest.raises(PortfolioError):
        portfolio_metrics(weights, [0.001, 0.002], np.eye(2), 0)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_disparate_variances_recover_exact_minimum_and_sharpe(order):
    # Regression: a globally scaled covariance allowed success at [0.5,0.5,0]
    # with over five times the minimum risk and missed the Sharpe optimum.
    variances = np.array([1e-10, 1e-8, 1.0])[list(order)]
    inverse = 1 / variances
    expected = inverse / inverse.sum()
    expected_risk = np.sqrt(1 / inverse.sum())
    result = optimize_portfolio([0.001] * 3, np.diag(variances), 0)
    np.testing.assert_allclose(result.min_weights, expected, rtol=2e-7, atol=1e-15)
    np.testing.assert_allclose(result.weights, expected, rtol=2e-7, atol=1e-15)
    assert result.min_risk == pytest.approx(expected_risk, rel=1e-9)
    assert result.max_risk == pytest.approx(expected_risk, rel=1e-9)
    assert result.max_sharpe == pytest.approx(0.001 / expected_risk, rel=1e-9)
    np.testing.assert_allclose(result.frontier_risks, [expected_risk], rtol=1e-9)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_disparate_risks_and_distinct_means_match_analytic_frontier(order):
    order = list(order)
    variances = np.array([1e-10, 1e-8, 1.0])[order]
    means = np.array([0.001, 0.002, 0.003])[order]
    covariance = np.diag(variances)
    result = optimize_portfolio(means, covariance, 0, points=5)
    expected_sharpe_weights = means / variances
    expected_sharpe_weights /= expected_sharpe_weights.sum()
    np.testing.assert_allclose(result.weights, expected_sharpe_weights, rtol=2e-7, atol=1e-14)
    for weights, target, risk in zip(result.frontier_weights, result.frontier_returns, result.frontier_risks):
        assert_feasible(weights)
        assert weights @ means == pytest.approx(target, abs=1e-12)
        # For 3 assets and two equalities, enumerate all possible supports.
        # Each support's closed-form QP solution is an independent oracle.
        candidates = []
        for mask in range(1, 8):
            support = np.flatnonzero([(mask >> i) & 1 for i in range(3)])
            submeans = means[support]
            if len(support) == 1:
                if abs(submeans[0] - target) < 1e-12:
                    candidates.append(variances[support[0]])
                continue
            rows = np.vstack((np.ones(len(support)), (submeans - 0.001) / 0.002))
            rhs = np.array([1, (target - 0.001) / 0.002])
            # Whiten columns to avoid using the same global-covariance scaling
            # that caused the defect; solve the equality problem by SVD.
            whitened = rows / np.sqrt(variances[support])
            row_norm = np.linalg.norm(whitened, axis=1)
            candidate = np.linalg.lstsq(whitened / row_norm[:, None], rhs / row_norm, rcond=None)[0]
            original = candidate / np.sqrt(variances[support])
            if np.min(original) >= -1e-9 and np.max(np.abs(rows @ original - rhs)) < 1e-8:
                candidates.append(float(candidate @ candidate))
        assert risk**2 == pytest.approx(min(candidates), rel=2e-6, abs=1e-18)


def test_disparate_correlated_assets_match_closed_form_interior_solution():
    deviations = np.array([1e-5, 1e-4, 1.0])
    correlation = np.array([[1, -0.2, -0.1], [-0.2, 1, -0.15], [-0.1, -0.15, 1]])
    covariance = correlation * np.outer(deviations, deviations)
    inverse_risk = np.linalg.solve(correlation, 1 / deviations) / deviations
    expected = inverse_risk / inverse_risk.sum()
    assert np.min(expected) > 0
    result = optimize_portfolio([0.001] * 3, covariance, 0)
    np.testing.assert_allclose(result.min_weights, expected, rtol=2e-7, atol=1e-14)
    np.testing.assert_allclose(result.weights, expected, rtol=2e-7, atol=1e-14)
    assert result.min_risk == pytest.approx(np.sqrt(expected @ covariance @ expected), rel=1e-9)


def test_relative_certificate_rejects_old_inaccurate_solution_at_tiny_objective():
    covariance = np.diag([1e-10, 1e-8, 1.0])
    with pytest.raises(PortfolioError, match="optimizer could not"):
        portfolio._certify_optimality(covariance, np.array([0.5, 0.5, 0]), np.ones((1, 3)), np.array([1]))
