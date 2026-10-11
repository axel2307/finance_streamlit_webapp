"""Offline UI evidence for consistent long-only and short-enabled portfolios."""

import numpy as np
import pandas as pd
import pytest
import streamlit as st

import portfolio_modes_v1 as portfolio
from test_app import (
    assert_no_exception,
    assert_no_portfolio_results,
    assert_valid_downloads_remain_available,
    downloaded_returns,
    offline_app,
    select_and_go,
)


def prices_with_moments(means=(0.001, 0.0), covariance=None):
    """25 prices whose 24 returns have controlled sample means/covariance."""
    if covariance is None:
        covariance = np.array([[0.0001, 0.00012], [0.00012, 0.0003]])
    count = 24
    phase = 2 * np.pi * np.arange(count) / count
    basis = np.column_stack((np.sin(phase), np.cos(phase))) * np.sqrt(2 * (count - 1) / count)
    returns = np.asarray(means) + basis @ np.linalg.cholesky(covariance).T
    baseline = np.array([100.0, 150.0])
    values = np.vstack((baseline, baseline * np.cumprod(1 + returns, axis=0)))
    columns = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("Close", "MSFT")], names=["Price", "Ticker"]
    )
    return pd.DataFrame(values, index=pd.date_range("2025-01-02", periods=count + 1, freq="B", name="Date"), columns=columns)


@pytest.fixture
def short_app(offline_app, monkeypatch):
    create_app, responses, calls = offline_app
    responses["prices"] = prices_with_moments()
    evidence = {"optimizations": [], "lines": []}
    original_optimizer = portfolio.optimize_portfolio

    def capture_optimizer(*args, **kwargs):
        result = original_optimizer(*args, **kwargs)
        evidence["optimizations"].append((kwargs.copy(), result))
        return result

    monkeypatch.setattr(portfolio, "optimize_portfolio", capture_optimizer)
    # The existing fixture seeds Dirichlet draws; also seed the short-leg size.
    monkeypatch.setattr(np.random, "uniform", np.random.default_rng(54321).uniform)
    original_pyplot = st.pyplot

    def capture_frontier(figure=None, *args, **kwargs):
        if figure is not None:
            for axis in figure.get_axes():
                for line in axis.lines:
                    evidence["lines"].append({
                        "label": line.get_label(),
                        "x": np.asarray(line.get_xdata()).copy(),
                        "y": np.asarray(line.get_ydata()).copy(),
                    })
        return original_pyplot(figure, *args, **kwargs)

    monkeypatch.setattr(st, "pyplot", capture_frontier)
    return create_app, responses, calls, evidence


def set_mode(app, mode, gross_percent=200):
    app.sidebar.radio(key="portfolio_mode").set_value(mode).run()
    assert_no_exception(app)
    if mode == "Short allowed":
        app.sidebar.number_input(key="gross_limit_percent").set_value(gross_percent)
    return app


def displayed_weights(app):
    tables = [
        element.value for element in list(app.dataframe) + list(app.table)
        if "Opt Weight" in element.value.columns
    ]
    assert len(tables) == 1
    return tables[0].set_index("Symbol")["Opt Weight"].reindex(["AAPL", "MSFT"]).to_numpy()


def displayed_metrics(app):
    tables = [table.value for table in app.table if "Sharpe Ratio" in table.value.columns]
    assert len(tables) == 1
    return tables[0].iloc[0]


def assert_feasible(weights, gross_limit, allow_short):
    values = np.atleast_2d(weights)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values.sum(axis=1), 1, rtol=0, atol=1e-7)
    assert (np.abs(values).sum(axis=1) <= gross_limit + 1e-7).all()
    if not allow_short:
        assert (values >= -1e-7).all()


def chart_collections(responses, label):
    return [
        collection["offsets"] for plot in responses["plots"] for collection in plot["collections"]
        if collection["label"] == label
    ]


def test_long_only_default_preserves_existing_controls(short_app):
    create_app, _, calls, _ = short_app
    app = create_app()

    assert [radio.label for radio in app.sidebar.radio] == [
        "Asset Class", "Exchange", "Serch by:", "Portfolio mode"
    ]
    mode = app.sidebar.radio(key="portfolio_mode")
    assert mode.options == ["Long-only", "Short allowed"]
    assert mode.value == "Long-only"
    assert not app.sidebar.number_input
    assert app.sidebar.selectbox[0].value == "1y"
    assert app.sidebar.selectbox[1].value == "1d"
    assert not calls


def test_short_control_explains_net_and_gross_without_an_arbitrary_upper_bound(short_app):
    create_app, _, calls, _ = short_app
    app = set_mode(create_app(), "Short allowed")
    limit = app.sidebar.number_input(key="gross_limit_percent")

    assert limit.min == 100
    assert limit.value == 200
    assert limit.step == 25
    assert limit.max is None
    caption = " ".join(element.value for element in app.sidebar.caption)
    for explanation in ["Negative weights", "100%", "sum of absolute weights", "200%", "150% long", "50% short", "Borrowing costs are not included"]:
        assert explanation in caption
    limit.set_value(10000).run()
    assert_no_exception(app)
    assert app.sidebar.number_input(key="gross_limit_percent").value == 10000
    assert not calls


@pytest.mark.parametrize(
    "mode,gross_percent,expected_weights",
    [
        ("Long-only", 100, [1.0, 0.0]),
        ("Short allowed", 100, [1.0, 0.0]),
        ("Short allowed", 150, [1.25, -0.25]),
        ("Short allowed", 200, [1.5, -0.5]),
    ],
)
def test_mode_weights_metrics_markers_frontier_and_cloud_are_consistent(short_app, mode, gross_percent, expected_weights):
    create_app, responses, _, evidence = short_app
    app = select_and_go(set_mode(create_app(), mode, gross_percent), ["AAPL", "MSFT"])
    assert_no_exception(app)
    assert not app.error

    allow_short = mode == "Short allowed"
    gross_limit = gross_percent / 100 if allow_short else 1.0
    kwargs, solution = evidence["optimizations"][-1]
    assert kwargs == {"allow_short": allow_short, "gross_limit": gross_limit}
    assert solution.allow_short == allow_short
    assert solution.gross_limit == gross_limit
    weights = displayed_weights(app)
    np.testing.assert_allclose(weights, expected_weights, rtol=0, atol=2e-6)
    np.testing.assert_allclose(weights, solution.weights, rtol=0, atol=1e-10)
    for allocation in [weights, solution.min_weights, solution.frontier_weights]:
        assert_feasible(allocation, gross_limit, allow_short)

    returns = downloaded_returns(app)
    means = returns.mean().to_numpy()
    covariance = returns.cov().to_numpy()
    risk_free = 0.042 / (360 - 91 * 0.042)
    expected_return = float(weights @ means)
    expected_risk = float(np.sqrt(weights @ covariance @ weights))
    metrics = displayed_metrics(app)
    assert metrics["Portfolio"] == f"{mode} Max Sharpe Ratio"
    assert metrics["Return"] == pytest.approx(expected_return, rel=1e-7, abs=1e-10)
    assert metrics["Std Dev"] == pytest.approx(expected_risk, rel=1e-7, abs=1e-10)
    assert metrics["Sharpe Ratio"] == pytest.approx((expected_return - risk_free) / expected_risk, rel=1e-7)
    assert metrics["Net Exposure (%)"] == pytest.approx(weights.sum() * 100)
    assert metrics["Gross Exposure (%)"] == pytest.approx(np.abs(weights).sum() * 100)

    suffix = " (Short allowed)" if allow_short else ""
    maximum = chart_collections(responses, f"Max Sharpe Ratio Portfolio{suffix}")
    minimum = chart_collections(responses, f"Minimum Variance Portfolio{suffix}")
    assert len(maximum) == len(minimum) == 1
    np.testing.assert_allclose(maximum[0], [[expected_risk, expected_return]], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(minimum[0], [[solution.min_risk, solution.min_return]], rtol=1e-7, atol=1e-10)
    frontier = [line for line in evidence["lines"] if line["label"] == f"Efficient Frontier{suffix}"]
    assert len(frontier) == 1
    np.testing.assert_allclose(frontier[0]["x"], solution.frontier_risks, rtol=0, atol=1e-10)
    np.testing.assert_allclose(frontier[0]["y"], solution.frontier_returns, rtol=0, atol=1e-10)

    cloud = chart_collections(responses, f"Random Portfolios{suffix}")
    assert len(cloud) == 1 and cloud[0].shape == (1000, 2)
    # With two distinct means the plotted return identifies both sample weights.
    first_weight = (cloud[0][:, 1] - means[1]) / (means[0] - means[1])
    cloud_weights = np.column_stack((first_weight, 1 - first_weight))
    assert_feasible(cloud_weights, gross_limit, allow_short)
    expected_cloud_risk = np.sqrt(np.einsum("ij,jk,ik->i", cloud_weights, covariance, cloud_weights))
    np.testing.assert_allclose(cloud[0][:, 0], expected_cloud_risk, rtol=1e-7, atol=1e-10)
    if allow_short and gross_limit > 1:
        assert np.min(cloud_weights) < -0.01


def test_switching_back_to_long_only_discards_the_hidden_short_budget(short_app):
    create_app, _, _, evidence = short_app
    app = set_mode(create_app(), "Short allowed", 300)
    app.run()
    app = set_mode(app, "Long-only")
    assert not app.sidebar.number_input
    app = select_and_go(app, ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    assert evidence["optimizations"][-1][0] == {"allow_short": False, "gross_limit": 1.0}
    np.testing.assert_allclose(displayed_weights(app), [1.0, 0.0], atol=2e-6)
    assert displayed_metrics(app)["Portfolio"] == "Long-only Max Sharpe Ratio"
    assert displayed_metrics(app)["Gross Exposure (%)"] == pytest.approx(100)


def test_realized_gross_is_not_the_available_short_budget(short_app):
    create_app, responses, _, _ = short_app
    responses["prices"] = prices_with_moments(means=(0.001, 0.001), covariance=np.diag([0.0001, 0.0004]))
    app = select_and_go(set_mode(create_app(), "Short allowed", 200), ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    weights = displayed_weights(app)
    np.testing.assert_allclose(weights, [0.8, 0.2], atol=2e-6)
    assert displayed_metrics(app)["Gross Exposure (%)"] == pytest.approx(100, abs=1e-5)
    assert displayed_metrics(app)["Gross Exposure (%)"] < 200


def test_negative_equal_means_show_actual_sharpe_marker_even_off_the_frontier(short_app):
    create_app, responses, _, evidence = short_app
    responses["prices"] = prices_with_moments(means=(-0.001, -0.001), covariance=np.diag([0.0001, 0.0004]))
    app = select_and_go(set_mode(create_app(), "Short allowed", 200), ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    metrics = displayed_metrics(app)
    weights = displayed_weights(app)
    np.testing.assert_allclose(weights, [-0.5, 1.5], atol=2e-6)
    assert metrics["Sharpe Ratio"] < 0
    assert any(
        "All feasible portfolios have nonpositive excess returns. Maximizing a negative Sharpe ratio can favor higher risk, so this portfolio may lie outside the efficient frontier."
        == element.value for element in app.caption
    )
    solution = evidence["optimizations"][-1][1]
    assert solution.max_risk > solution.min_risk
    marker = chart_collections(responses, "Max Sharpe Ratio Portfolio (Short allowed)")
    assert len(marker) == 1
    np.testing.assert_allclose(marker[0], [[metrics["Std Dev"], metrics["Return"]]], rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("mode", ["Long-only", "Short allowed"])
def test_failed_optimizer_preserves_valid_csvs_without_portfolio_results(short_app, monkeypatch, mode):
    create_app, responses, _, _ = short_app
    message = "The optimizer could not find a reliable portfolio."
    options = []

    def fail_optimization(*args, **kwargs):
        options.append(kwargs)
        raise portfolio.PortfolioError(message)

    monkeypatch.setattr(portfolio, "optimize_portfolio", fail_optimization)
    app = select_and_go(set_mode(create_app(), mode, 150), ["AAPL", "MSFT"])

    assert_no_portfolio_results(app)
    assert message in [element.value for element in app.error]
    assert options == [{"allow_short": mode == "Short allowed", "gross_limit": 1.5 if mode == "Short allowed" else 1.0}]
    assert_valid_downloads_remain_available(app, expected_return_rows=24)
    assert not responses["plots"]
    assert not any("Opt Weight" in element.value.columns for element in list(app.dataframe) + list(app.table))
