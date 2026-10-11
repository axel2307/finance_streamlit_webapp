"""Annual CAL geometry and text comparisons use the actual constrained solution."""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pytest

import portfolio_charts_v2 as charts
import portfolio_modes_v1 as portfolio


MEANS = np.array([0.252, 0.0])
COVARIANCE = np.array([[0.0001, 0.00012], [0.00012, 0.0003]]) * 252


def solve(risk_free=0.04, allow_short=False, means=MEANS):
    return portfolio.optimize_portfolio(
        means, COVARIANCE, risk_free, allow_short=allow_short,
        gross_limit=2.0 if allow_short else 1.0,
    )


@pytest.fixture
def rendered(monkeypatch):
    figures = []

    def capture(figure, **kwargs):
        figures.append((figure, kwargs))

    monkeypatch.setattr(charts, "display_figure", capture)
    yield figures
    for figure, _ in figures:
        plt.close(figure)


@pytest.mark.parametrize("allow_short", [False, True], ids=["long-only", "gross-constrained"])
@pytest.mark.parametrize("risk_free", [0.04, 0.0, -0.03], ids=["positive-rf", "zero-rf", "negative-rf"])
def test_cal_stops_at_constrained_maximum_sharpe_and_has_annual_slope(allow_short, risk_free):
    solution = solve(risk_free, allow_short)
    # Both fixtures bind a constraint: an asset reaches zero or gross reaches 2.
    expected_weights = [1.5, -0.5] if allow_short else [1.0, 0.0]
    np.testing.assert_allclose(solution.weights, expected_weights, atol=2e-6, rtol=0)
    xs, ys = charts.capital_allocation_segment(solution, risk_free)

    np.testing.assert_array_equal(xs, [0, solution.max_risk])
    np.testing.assert_array_equal(ys, [risk_free, solution.max_return])
    assert len(xs) == len(ys) == 2  # No leveraged extension beyond the risky portfolio.
    assert (ys[1] - ys[0]) / (xs[1] - xs[0]) == pytest.approx(solution.max_sharpe)
    assert max(xs) == solution.max_risk
    assert ys[0] == risk_free  # Neither calendar-day nor trading-day conversion.


@pytest.mark.parametrize("allow_short", [False, True], ids=["long-only", "short"])
@pytest.mark.parametrize("excess", [0.0, -0.1], ids=["zero-sharpe", "negative-sharpe"])
def test_nonpositive_maximum_sharpe_does_not_create_an_improving_cal(allow_short, excess):
    risk_free = 0.04
    solution = solve(risk_free, allow_short, means=np.full(2, risk_free + excess))
    assert solution.max_sharpe <= 0
    assert charts.capital_allocation_segment(solution, risk_free) is None
    comparison = charts.optimization_comparison(solution, risk_free)
    assert np.isfinite(comparison.select_dtypes(include="number").to_numpy()).all()
    assert (comparison["Sharpe ratio"] <= 1e-14).all()  # GMV dot products may leave roundoff at zero.


@pytest.mark.parametrize("allow_short", [False, True], ids=["long-only", "short"])
def test_comparison_matches_actual_weights_and_annual_moments(allow_short):
    risk_free = 0.04
    solution = solve(risk_free, allow_short)
    table = charts.optimization_comparison(solution, risk_free)
    assert table.columns.tolist() == ["Portfolio", "Annual return", "Annual volatility", "Sharpe ratio"]
    assert table["Portfolio"].tolist() == ["Maximum Sharpe", "Global minimum variance"]
    weights = np.stack([solution.weights, solution.min_weights])
    expected_returns = weights @ MEANS
    expected_risks = np.sqrt(np.einsum("ij,jk,ik->i", weights, COVARIANCE, weights))
    np.testing.assert_allclose(table["Annual return"], expected_returns, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(table["Annual volatility"], expected_risks, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(table["Sharpe ratio"], (expected_returns - risk_free) / expected_risks,
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("field,value", [
    ("max_risk", 0.0), ("max_risk", -0.1), ("max_risk", 1e-13),
    ("max_risk", 1e-12), ("max_risk", np.nan), ("max_risk", np.inf),
    ("min_risk", 0.0), ("min_risk", 1e-13), ("min_risk", np.nan),
    ("max_return", np.nan), ("min_return", np.inf), ("max_sharpe", np.nan),
    ("max_sharpe", 123.0),
])
def test_invalid_or_unmeasurably_small_visual_risk_is_controlled(field, value):
    invalid = replace(solve(), **{field: value})
    # Even under strict NumPy handling, report the domain error, not divide-by-zero.
    with np.errstate(all="raise"):
        for helper in (charts.capital_allocation_segment, charts.optimization_comparison):
            with pytest.raises(charts.OptimizationViewError):
                helper(invalid, 0.04)


@pytest.mark.parametrize("risk_free", [np.nan, np.inf, -np.inf])
def test_nonfinite_risk_free_is_rejected_before_building_the_view(risk_free):
    solution = solve()
    with np.errstate(all="raise"):
        for helper in (charts.capital_allocation_segment, charts.optimization_comparison):
            with pytest.raises(charts.OptimizationViewError):
                helper(solution, risk_free)


@pytest.mark.parametrize("allow_short", [False, True], ids=["long-only", "short"])
@pytest.mark.parametrize("show_cloud", [True, False], ids=["samples-visible", "samples-hidden"])
def test_chart_marks_rf_and_actual_optima_from_zero_with_optional_subdued_samples(
    rendered, monkeypatch, allow_short, show_cloud,
):
    solution = solve(allow_short=allow_short)
    if not show_cloud:
        def forbid_sampling(*args, **kwargs):
            pytest.fail("Hiding samples must skip generating a sample cloud")
        monkeypatch.setattr(np.random, "default_rng", forbid_sampling)
    charts.plot_efficient_frontier_and_max_sharpe(
        MEANS, COVARIANCE, 0.04, solution, show_cloud=show_cloud,
    )
    assert len(rendered) == 1
    figure, kwargs = rendered[0]
    axis = figure.axes[0]
    suffix = " (Short allowed)" if allow_short else ""
    collections = {item.get_label(): item for item in axis.collections}
    lines = {line.get_label(): line for line in axis.lines}
    assert axis.get_xlim()[0] == 0
    assert (axis.get_xlabel(), axis.get_ylabel()) == ("Annual Volatility", "Annual Mean Return")
    assert isinstance(axis.xaxis.get_major_formatter(), PercentFormatter)
    assert isinstance(axis.yaxis.get_major_formatter(), PercentFormatter)
    for label, expected in [
        ("Risk-free proxy", [[0, 0.04]]),
        (f"Max Sharpe Ratio Portfolio{suffix}", [[solution.max_risk, solution.max_return]]),
        (f"Minimum Variance Portfolio{suffix}", [[solution.min_risk, solution.min_return]]),
    ]:
        np.testing.assert_allclose(collections[label].get_offsets(), expected, rtol=0, atol=1e-12)
    cal = lines["CAL — Maximum Sharpe"]
    np.testing.assert_array_equal(cal.get_xdata(), [0, solution.max_risk])
    np.testing.assert_array_equal(cal.get_ydata(), [0.04, solution.max_return])
    marker_shapes = [collections[label].get_paths()[0].vertices for label in (
        "Risk-free proxy", f"Max Sharpe Ratio Portfolio{suffix}", f"Minimum Variance Portfolio{suffix}"
    )]
    assert all(not np.array_equal(first, second)
               for position, first in enumerate(marker_shapes)
               for second in marker_shapes[position + 1:])
    assert "alt" in kwargs and "risk-free" in kwargs["alt"].lower()
    assert not any("tangen" in text.lower() or "cml" in text.lower() for text in lines)
    cloud_label = f"Random Portfolios{suffix}"
    assert (cloud_label in collections) is show_cloud
    assert len(figure.axes) == (2 if show_cloud else 1)
    if show_cloud:
        assert len(collections[cloud_label].get_offsets()) == 1000
        assert collections[cloud_label].get_alpha() <= 0.25


@pytest.mark.parametrize("excess", [0.0, -0.1], ids=["zero-sharpe", "negative-sharpe"])
def test_nonpositive_chart_keeps_rf_and_actual_optima_without_cal(rendered, excess):
    risk_free = 0.04
    means = np.full(2, risk_free + excess)
    solution = solve(risk_free, True, means=means)
    charts.plot_efficient_frontier_and_max_sharpe(
        means, COVARIANCE, risk_free, solution, show_cloud=False,
    )
    axis = rendered[0][0].axes[0]
    assert not any("CAL" in line.get_label() for line in axis.lines)
    assert any(item.get_label() == "Risk-free proxy" for item in axis.collections)
    assert axis.get_xlim()[0] == 0
