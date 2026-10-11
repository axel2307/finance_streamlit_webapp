"""Independent numeric oracles for fully invested, gross-bounded short portfolios.

No oracle calls the production optimizer or another numerical optimizer.
Two-asset extrema use a closed-form derivative, and the tied four-asset
endpoint uses an equality-constrained quadratic solution.
"""

import numpy as np
import pytest

from portfolio import PortfolioError, optimize_portfolio, portfolio_metrics


def direct_metrics(weights, means, covariance, risk_free):
    weights = np.asarray(weights, dtype=float)
    ret = float(weights @ means)
    risk = float(np.sqrt(weights @ covariance @ weights))
    return ret, risk, (ret - risk_free) / risk


def assert_short_feasible(weights, gross_limit):
    weights = np.asarray(weights)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1, abs=1e-7)
    assert np.abs(weights).sum() <= gross_limit + 1e-7


def two_asset_oracle(means, covariance, risk_free, gross_limit):
    """All stationary/end points of (d+e*x)/sqrt(A*x*x+B*x+C)."""
    lower, upper = (1 - gross_limit) / 2, (1 + gross_limit) / 2
    a = covariance[0, 0] - 2 * covariance[0, 1] + covariance[1, 1]
    b = 2 * (covariance[0, 1] - covariance[1, 1])
    c = covariance[1, 1]
    d, e = means[1] - risk_free, means[0] - means[1]
    candidates = [lower, upper]
    slope = e * b - 2 * d * a
    if slope != 0:
        stationary = (d * b - 2 * e * c) / slope
        if lower <= stationary <= upper:
            candidates.append(stationary)
    scores = [(d + e * x) / np.sqrt(a * x**2 + b * x + c) for x in candidates]
    best_x = candidates[int(np.argmax(scores))]
    minimum_x = float(np.clip(-b / (2 * a), lower, upper))
    endpoint_x = upper if means[0] > means[1] else lower
    return {
        "max_sharpe": max(scores),
        "weights": np.array([best_x, 1 - best_x]),
        "min_weights": np.array([minimum_x, 1 - minimum_x]),
        "endpoint": np.array([endpoint_x, 1 - endpoint_x]),
    }


TWO_ASSET_CASES = [
    pytest.param([0.001, 0.002], [[0.0001, 0], [0, 0.0004]], 0.0001, 2.0, id="positive-interior"),
    pytest.param([0.001, 0.0005], [[0.0001, 0.00018], [0.00018, 0.0004]], 0.0001, 2.0, id="positive-at-gross-cap"),
    pytest.param([0.001, 0.0005], [[0.0001, 0.00018], [0.00018, 0.0004]], 0.0001, 3.0, id="positive-interior-with-short"),
    pytest.param([0.002, -0.001], [[0.0004, 0], [0, 0.0001]], 0.0, 2.0, id="mixed-excess"),
    pytest.param([0.001, 0.002], [[0.0001, 0], [0, 0.0004]], 0.01, 2.0, id="all-feasible-excess-negative"),
    pytest.param([-0.01, -0.10], [[0.04, 0], [0, 0.09]], 0.0, 2.0, id="negative-assets-positive-portfolio"),
    pytest.param([-0.1, -0.1], [[0.01, 0], [0, 1.0]], 0.0, 2.0, id="equal-negative-excess-max-risk-vertex"),
    pytest.param([-1.0, -3.0], [[0.04, 0], [0, 0.09]], 0.0, 2.0, id="maximum-feasible-excess-zero"),
    pytest.param([0.001, 0.001], [[0.0001, 0], [0, 0.0004]], 0.001, 2.0, id="all-excess-zero"),
    pytest.param([0.0002, 0.002], [[1e-10, 0], [0, 1e-4]], 0.0001, 2.0, id="heterogeneous-risk-scales"),
]


@pytest.mark.parametrize("means,covariance,risk_free,gross_limit", TWO_ASSET_CASES)
def test_short_two_asset_optimum_minimum_and_frontier_match_closed_form(means, covariance, risk_free, gross_limit):
    means, covariance = np.array(means), np.array(covariance)
    oracle = two_asset_oracle(means, covariance, risk_free, gross_limit)
    result = optimize_portfolio(means, covariance, risk_free, points=7, allow_short=True, gross_limit=gross_limit)

    assert result.allow_short is True
    assert result.gross_limit == gross_limit
    for weights in [result.weights, result.min_weights, *result.frontier_weights]:
        assert_short_feasible(weights, gross_limit)
    measured = direct_metrics(result.weights, means, covariance, risk_free)
    assert (result.max_return, result.max_risk, result.max_sharpe) == pytest.approx(measured, rel=3e-6, abs=1e-10)
    assert result.max_sharpe == pytest.approx(oracle["max_sharpe"], rel=3e-6, abs=1e-10)
    np.testing.assert_allclose(result.min_weights, oracle["min_weights"], rtol=3e-6, atol=1e-7)
    minimum = direct_metrics(oracle["min_weights"], means, covariance, risk_free)
    assert result.min_return == pytest.approx(minimum[0], abs=1e-9)
    assert result.min_risk == pytest.approx(minimum[1], rel=3e-6)

    np.testing.assert_allclose(result.frontier_returns, result.frontier_weights @ means, atol=1e-9)
    direct_risks = np.sqrt(np.einsum("ij,jk,ik->i", result.frontier_weights, covariance, result.frontier_weights))
    np.testing.assert_allclose(result.frontier_risks, direct_risks, rtol=3e-6)
    assert np.all(np.diff(result.frontier_returns) >= -1e-9)
    assert np.all(np.diff(result.frontier_risks) >= -1e-9)

    if means[0] == means[1]:
        assert result.frontier_weights.shape == (1, 2)
        np.testing.assert_allclose(result.frontier_weights[0], oracle["min_weights"], atol=1e-7)
    else:
        np.testing.assert_allclose(result.frontier_weights[-1], oracle["endpoint"], atol=1e-7)
        for target, weights, risk in zip(result.frontier_returns, result.frontier_weights, result.frontier_risks):
            # With two distinct means, sum(w)=1 and the target uniquely fix w.
            first = (target - means[1]) / (means[0] - means[1])
            expected_weights = np.array([first, 1 - first])
            np.testing.assert_allclose(weights, expected_weights, atol=1e-7)
            assert risk == pytest.approx(direct_metrics(expected_weights, means, covariance, risk_free)[1], rel=3e-6)


def test_gross_one_is_the_same_feasible_set_as_long_only():
    means = np.array([0.001, 0.002, -0.0003])
    covariance = np.array([[0.0004, 0.00003, 0], [0.00003, 0.0001, 0.00002], [0, 0.00002, 0.0009]])
    baseline = optimize_portfolio(means, covariance, 0.0001, points=5)
    short = optimize_portfolio(means, covariance, 0.0001, points=5, allow_short=True, gross_limit=1)
    assert short.allow_short is True
    assert short.gross_limit == 1
    for name in ("weights", "min_weights", "frontier_weights", "frontier_returns", "frontier_risks"):
        np.testing.assert_allclose(getattr(short, name), getattr(baseline, name), atol=1e-8)
    assert short.max_sharpe == pytest.approx(baseline.max_sharpe)
    for weights in [short.weights, *short.frontier_weights]:
        assert np.min(weights) >= -1e-8
        assert_short_feasible(weights, 1)


def test_default_long_only_ignores_irrelevant_gross_argument():
    means, covariance = np.array([0.001, 0.002]), np.diag([0.0001, 0.0004])
    baseline = optimize_portfolio(means, covariance, 0.0001, points=3)
    ignored = optimize_portfolio(means, covariance, 0.0001, points=3, allow_short=False, gross_limit="not-used")
    assert baseline.allow_short is False
    assert ignored.allow_short is False
    assert baseline.gross_limit == ignored.gross_limit == 1
    np.testing.assert_allclose(ignored.weights, baseline.weights, atol=1e-10)
    np.testing.assert_allclose(ignored.frontier_weights, baseline.frontier_weights, atol=1e-10)
    assert portfolio_metrics([0.3, 0.7], means, covariance, 0.0001, gross_limit=None) == pytest.approx(
        direct_metrics(np.array([0.3, 0.7]), means, covariance, 0.0001)
    )


def test_short_metrics_use_signed_weights_and_whole_gross_exposure():
    weights = np.array([1.2, 0.3, -0.5])
    means = np.array([0.002, 0.001, -0.0004])
    covariance = np.array([[0.0004, 0.00003, -0.00001], [0.00003, 0.0001, 0.00002], [-0.00001, 0.00002, 0.0009]])
    actual = portfolio_metrics(weights, means, covariance, 0.0001, allow_short=True, gross_limit=2)
    assert actual == pytest.approx(direct_metrics(weights, means, covariance, 0.0001), rel=1e-10)


@pytest.mark.parametrize("weights", [
    pytest.param([1, 1, -0.5, -0.5], id="gross-three-with-every-weight-inside-individual-bounds"),
    pytest.param([1.1, -0.2, 0], id="net-investment-not-one"),
])
def test_short_metrics_reject_joint_domain_violations(weights):
    # The first vector obeys even the tight component bounds [-0.5, 1.5]
    # implied by G=2, but its whole gross exposure is 3 rather than <=2.
    with pytest.raises(PortfolioError):
        portfolio_metrics(weights, np.full(len(weights), 0.001), np.eye(len(weights)), 0, allow_short=True, gross_limit=2)


def test_three_asset_gross_cap_and_endpoint_match_independent_grid():
    means, covariance = np.array([0.002, 0.001, -0.003]), np.eye(3) * 0.0001
    result = optimize_portfolio(means, covariance, 0, points=7, allow_short=True, gross_limit=2)
    grid = np.linspace(-0.5, 1.5, 401)
    first, second = np.meshgrid(grid, grid)
    weights = np.column_stack([first.ravel(), second.ravel(), (1 - first - second).ravel()])
    weights = weights[np.abs(weights).sum(axis=1) <= 2 + 1e-12]
    sharpes = (weights @ means) / np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))
    assert result.max_sharpe >= float(np.max(sharpes)) - 1e-8
    assert np.abs(result.weights).sum() == pytest.approx(2, abs=1e-6)
    np.testing.assert_allclose(result.min_weights, np.full(3, 1 / 3), atol=1e-7)
    np.testing.assert_allclose(result.frontier_weights[-1], [1.5, 0, -0.5], atol=1e-7)
    assert result.frontier_returns[-1] == pytest.approx(0.0045)
    assert result.frontier_returns[-1] > np.max(means)
    for candidate in [result.weights, result.min_weights, *result.frontier_weights]:
        assert_short_feasible(candidate, 2)


def test_tied_extreme_returns_endpoint_accounts_for_cross_covariance():
    means = np.array([0.002, 0.002, -0.001, -0.001])
    covariance = np.array([
        [0.0004, 0.00003, 0.00004, -0.00002],
        [0.00003, 0.0009, -0.00001, 0.00006],
        [0.00004, -0.00001, 0.000225, 0.00002],
        [-0.00002, 0.00006, 0.00002, 0.000625],
    ])
    # Maximum return fixes total long weight=1.5 on the two high-return
    # assets and short weight=-0.5 on the two low-return assets. Solve the
    # coupled equality-constrained variance problem, not each side separately.
    equality = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    inverse_times_equality = np.linalg.solve(covariance, equality.T)
    expected = inverse_times_equality @ np.linalg.solve(equality @ inverse_times_equality, [1.5, -0.5])
    assert np.min(expected[:2]) > 0 and np.max(expected[2:]) < 0
    result = optimize_portfolio(means, covariance, 0.0001, points=5, allow_short=True, gross_limit=2)
    np.testing.assert_allclose(result.frontier_weights[-1], expected, rtol=3e-6, atol=1e-7)
    assert result.frontier_returns[-1] == pytest.approx(0.0035)
    assert result.frontier_returns[-1] > np.max(means)
    assert result.frontier_risks[-1] == pytest.approx(np.sqrt(expected @ covariance @ expected), rel=3e-6)
    for candidate in [result.weights, result.min_weights, *result.frontier_weights]:
        assert_short_feasible(candidate, 2)


@pytest.mark.parametrize("gross_limit", [True, False, 0.99, -1, np.nan, np.inf, -np.inf, None, "bad"])
def test_invalid_short_gross_limit_is_rejected(gross_limit):
    with pytest.raises(PortfolioError):
        optimize_portfolio([0.001, 0.002], np.eye(2), 0, points=3, allow_short=True, gross_limit=gross_limit)
    with pytest.raises(PortfolioError):
        portfolio_metrics([0.5, 0.5], [0.001, 0.002], np.eye(2), 0, allow_short=True, gross_limit=gross_limit)


@pytest.mark.parametrize("gross_limit", [1.0, 2.0])
def test_one_asset_remains_fully_invested_with_short_option(gross_limit):
    result = optimize_portfolio([0.002], [[0.0004]], 0.0001, points=3, allow_short=True, gross_limit=gross_limit)
    np.testing.assert_array_equal(result.weights, [1])
    np.testing.assert_array_equal(result.min_weights, [1])
    np.testing.assert_array_equal(result.frontier_weights, [[1]])
    assert (result.max_return, result.max_risk, result.max_sharpe) == pytest.approx((0.002, 0.02, 0.095))
    assert result.allow_short is True
    assert result.gross_limit == gross_limit


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_short_return_unit_changes_preserve_allocations(factor):
    means = np.array([0.001, 0.0005])
    covariance = np.array([[0.0001, 0.00018], [0.00018, 0.0004]])
    base = optimize_portfolio(means, covariance, 0.0001, points=5, allow_short=True, gross_limit=3)
    scaled = optimize_portfolio(means * factor, covariance * factor**2, 0.0001 * factor, points=5, allow_short=True, gross_limit=3)
    np.testing.assert_allclose(scaled.weights, base.weights, atol=2e-6)
    np.testing.assert_allclose(scaled.frontier_weights, base.frontier_weights, atol=2e-6)
    assert scaled.max_sharpe == pytest.approx(base.max_sharpe, rel=3e-6)
