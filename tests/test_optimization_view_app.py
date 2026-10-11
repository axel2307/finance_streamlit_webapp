"""Optimization display controls never submit draft analysis parameters."""

from dataclasses import replace
import sys

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit import dataframe_util

import portfolio_charts_v1 as legacy_charts
from test_analysis_flow import (
    assert_pending, assert_same_results, assert_success, downloads,
    frozen_results, snapshot_or_none,
)
from test_app import assert_no_exception, offline_app, select_and_go
from test_irx_app import irx_app
from test_short_app import prices_with_moments, set_mode


TOGGLE_KEY = "optimization_show_cloud_v1"


@pytest.fixture
def optimization_app(irx_app, monkeypatch):
    create_app, responses, calls, evidence = irx_app
    evidence["optimization_charts"] = []
    original_pyplot = st.pyplot

    def capture_optimization(figure=None, *args, **kwargs):
        if figure is not None and figure.axes:
            axis = figure.axes[0]
            if axis.get_title().startswith("Efficient Frontier and Random Portfolios"):
                evidence["optimization_charts"].append({
                    "lines": {line.get_label(): (
                        np.asarray(line.get_xdata()).copy(), np.asarray(line.get_ydata()).copy()
                    ) for line in axis.lines},
                    "collections": {item.get_label(): np.asarray(item.get_offsets()).copy()
                                    for item in axis.collections},
                    "xlim": axis.get_xlim(),
                    "axes_count": len(figure.axes),
                })
        return original_pyplot(figure, *args, **kwargs)

    monkeypatch.setattr(st, "pyplot", capture_optimization)
    return create_app, responses, calls, evidence


def comparison_element(app):
    tables = [element for element in list(app.dataframe) + list(app.table)
              if "Sharpe ratio" in element.value.columns]
    assert len(tables) == 1
    return tables[0]


def assert_view_uses_snapshot(app, evidence, *, show_cloud):
    snapshot = snapshot_or_none(app)
    solution = snapshot["solution"]
    annual_rf = snapshot["risk_free_quote"].annual_simple_yield
    table = comparison_element(app)
    numeric = table.value
    assert numeric["Portfolio"].tolist() == ["Maximum Sharpe", "Global minimum variance"]
    np.testing.assert_allclose(numeric["Annual return"], [solution.max_return, solution.min_return])
    np.testing.assert_allclose(numeric["Annual volatility"], [solution.max_risk, solution.min_risk])
    np.testing.assert_allclose(numeric["Sharpe ratio"], [
        solution.max_sharpe, (solution.min_return - annual_rf) / solution.min_risk,
    ])
    rendered = dataframe_util.convert_arrow_bytes_to_pandas_df(table.proto.arrow_data.styler.display_values)
    for column in ("Annual return", "Annual volatility"):
        assert pd.api.types.is_numeric_dtype(numeric[column])
        assert rendered[column].tolist() == [f"{value:.2%}" for value in numeric[column]]
    assert rendered["Sharpe ratio"].tolist() == [f"{value:.3f}" for value in numeric["Sharpe ratio"]]
    chart = evidence["optimization_charts"][-1]
    suffix = " (Short allowed)" if solution.allow_short else ""
    np.testing.assert_allclose(chart["collections"][f"Max Sharpe Ratio Portfolio{suffix}"],
                               [[solution.max_risk, solution.max_return]])
    np.testing.assert_allclose(chart["collections"]["Risk-free proxy"], [[0, annual_rf]])
    assert chart["xlim"][0] == 0
    assert (f"Random Portfolios{suffix}" in chart["collections"]) is show_cloud
    assert chart["axes_count"] == (2 if show_cloud else 1)
    if solution.max_sharpe > 0:
        x, y = chart["lines"]["CAL — Maximum Sharpe"]
        np.testing.assert_allclose(x, [0, solution.max_risk])
        np.testing.assert_allclose(y, [annual_rf, solution.max_return])
    else:
        assert "CAL — Maximum Sharpe" not in chart["lines"]


@pytest.mark.parametrize("initial_mode", ["Long-only", "Short allowed"])
def test_sample_toggle_preserves_snapshot_csv_and_pending_parameters_until_submit(
    optimization_app, initial_mode,
):
    create_app, _, calls, evidence = optimization_app
    app = select_and_go(set_mode(create_app(), initial_mode), ["AAPL", "MSFT"])
    assert_success(app)
    assert app.toggle(key=TOGGLE_KEY).label == "Show sample portfolios"
    assert app.toggle(key=TOGGLE_KEY).value is True
    optimization_tab = next(tab for tab in app.tabs if tab.label == "Optimization")
    assert len(optimization_tab.toggle) == 1
    assert not app.sidebar.toggle
    baseline = frozen_results(app)
    snapshot = snapshot_or_none(app)
    original_signature = snapshot["signature"]
    original_completed_at = snapshot["completed_at"]
    provider_count = len(calls)
    optimizer_count = len(evidence["optimizations"])
    assert_view_uses_snapshot(app, evidence, show_cloud=True)

    app.toggle(key=TOGGLE_KEY).set_value(False).run()
    assert_same_results(app, baseline)
    assert not any("Pending changes" in warning.value for warning in app.warning)
    assert_view_uses_snapshot(app, evidence, show_cloud=False)

    # Draft values change both frequency and exposure; the toggle is not an analysis input.
    if initial_mode == "Long-only":
        app.sidebar.radio(key="portfolio_mode").set_value("Short allowed").run()
    app.sidebar.number_input(key="gross_limit_percent").set_value(300)
    app.sidebar.selectbox(key="return_frequency").set_value("1wk")
    app.sidebar.selectbox[0].set_value("2y").run()
    assert_pending(app)
    assert_same_results(app, baseline)
    for show_cloud in (True, False):
        app.toggle(key=TOGGLE_KEY).set_value(show_cloud).run()
        assert_pending(app)
        assert_same_results(app, baseline)
        assert snapshot_or_none(app)["signature"] == original_signature
        assert snapshot_or_none(app)["completed_at"] == original_completed_at
        assert_view_uses_snapshot(app, evidence, show_cloud=show_cloud)
        assert len(calls) == provider_count
        assert len(evidence["optimizations"]) == optimizer_count

    app.sidebar.button(key="analyze_portfolio").click().run()
    assert_success(app)
    current = snapshot_or_none(app)
    assert current["signature"] != original_signature
    assert current["params"]["interval"] == "1wk"
    assert current["params"]["allow_short"] is True
    assert current["params"]["gross_limit"] == 3
    assert app.toggle(key=TOGGLE_KEY).value is False
    assert not any("Pending changes" in warning.value for warning in app.warning)
    assert len(calls) == provider_count + 1  # Annual IRX is still cached.
    assert len(evidence["optimizations"]) == optimizer_count + 1
    assert_view_uses_snapshot(app, evidence, show_cloud=False)


def test_running_process_with_legacy_chart_module_uses_new_interface(optimization_app, monkeypatch):
    create_app, _, calls, evidence = optimization_app
    legacy_identity = sys.modules["portfolio_charts_v1"]

    def old_four_argument_interface(means, covariance, risk_free, solution):
        pytest.fail("The new cloud keyword must not call a cached old chart function")

    monkeypatch.setattr(legacy_charts, "plot_efficient_frontier_and_max_sharpe", old_four_argument_interface)
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_success(app)
    assert sys.modules["portfolio_charts_v1"] is legacy_identity
    assert "portfolio_charts_v2" in sys.modules
    assert_view_uses_snapshot(app, evidence, show_cloud=True)
    baseline = frozen_results(app)
    counts = len(calls), len(evidence["optimizations"])
    app.toggle(key=TOGGLE_KEY).set_value(False).run()
    assert_same_results(app, baseline)
    assert_view_uses_snapshot(app, evidence, show_cloud=False)
    assert (len(calls), len(evidence["optimizations"])) == counts


@pytest.mark.parametrize("mode", ["Long-only", "Short allowed"])
def test_negative_sharpe_explains_cash_dominance_and_keeps_valid_annual_comparison(optimization_app, mode):
    create_app, responses, _, evidence = optimization_app
    responses["prices"] = prices_with_moments(means=(-0.002, -0.003))
    app = select_and_go(set_mode(create_app(), mode), ["AAPL", "MSFT"])
    assert_success(app)
    assert snapshot_or_none(app)["solution"].max_sharpe < 0
    assert_view_uses_snapshot(app, evidence, show_cloud=True)
    warnings = " ".join(item.value for item in app.warning).lower()
    assert "risk-free" in warnings and "nonpositive" in warnings
    assert "at least as much expected return with zero modeled volatility" in warnings
    explanations = " ".join(item.value for item in list(app.caption) + list(app.info)).lower()
    assert "cash" in explanations
    assert "borrowing" in explanations


def test_invalid_near_zero_visual_risk_is_explained_without_losing_valid_downloads(optimization_app):
    create_app, _, calls, evidence = optimization_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    baseline_downloads = downloads(app)
    snapshot = snapshot_or_none(app)
    snapshot["solution"] = replace(snapshot["solution"], max_risk=1e-13)
    counts = len(calls), len(evidence["optimizations"])
    previous_charts = len(evidence["optimization_charts"])
    app.run()

    assert_no_exception(app)
    assert downloads(app) == baseline_downloads
    assert app.warning
    assert not any("Sharpe ratio" in table.value.columns for table in app.dataframe)
    assert len(evidence["optimization_charts"]) == previous_charts
    assert (len(calls), len(evidence["optimizations"])) == counts
