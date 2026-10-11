"""Analysis results persist independently of draft sidebar controls."""

import ftplib

import numpy as np
import pandas as pd
import pytest
import streamlit as st

import portfolio_modes_v1 as portfolio
from test_app import (
    assert_no_exception,
    downloaded_returns,
    offline_app,
    select_and_go,
)
from test_irx_app import irx_app
from test_short_app import displayed_weights, prices_with_moments, set_mode


SNAPSHOT_KEY = "analysis_snapshot_v1"


def snapshot_or_none(app):
    try:
        return app.session_state[SNAPSHOT_KEY]
    except KeyError:
        return None


def downloads(app):
    return tuple(element.value for element in app.markdown if "base64," in element.value)


def metric_cards(app):
    return {metric.label: metric.value for metric in app.metric}


def analyzed_settings(app):
    summaries = [element.value for element in app.caption
                 if element.value.startswith("Analyzed settings:")]
    assert len(summaries) == 1
    return summaries[0]


def price_calls(calls):
    return [call for call in calls if isinstance(call["tickers"], list)]


def assert_success(app):
    assert_no_exception(app)
    assert not app.error, [error.value for error in app.error]
    assert snapshot_or_none(app) is not None
    assert [tab.label for tab in app.tabs] == ["Overview", "Risk", "Optimization", "Data"]
    assert set(metric_cards(app)) == {"Annual return", "Annual volatility", "Sharpe ratio"}
    assert len(downloads(app)) == 3
    assert np.isfinite(displayed_weights(app)).all()


def frozen_results(app):
    return {
        "summary": analyzed_settings(app),
        "metrics": metric_cards(app),
        "downloads": downloads(app),
        "weights": displayed_weights(app).copy(),
    }


def assert_same_results(app, previous):
    assert_no_exception(app)
    assert analyzed_settings(app) == previous["summary"]
    assert metric_cards(app) == previous["metrics"]
    assert downloads(app) == previous["downloads"]
    np.testing.assert_array_equal(displayed_weights(app), previous["weights"])


def assert_pending(app):
    warnings = " ".join(element.value for element in app.warning)
    assert "Pending changes" in warnings
    assert "results and downloads below still use the analyzed settings" in warnings
    assert "Analyze portfolio" in warnings


def assert_no_optimization_results(app):
    assert_no_exception(app)
    assert not app.metric
    assert not any("Opt Weight" in element.value.columns
                   for element in list(app.dataframe) + list(app.table))
    assert not any("Sharpe Ratio" in element.value.columns for element in app.table)
    assert not any("Download Portfolio Weights" in link for link in downloads(app))


def test_empty_selection_does_not_download_or_create_results(irx_app):
    create_app, _, calls, evidence = irx_app
    app = create_app()
    assert app.sidebar.button[0].label == "Analyze portfolio"
    assert snapshot_or_none(app) is None
    app.sidebar.button[0].click().run()

    assert_no_optimization_results(app)
    assert snapshot_or_none(app) is None
    assert not downloads(app)
    assert not calls
    assert not evidence["optimizations"]
    assert any("Select at least one asset" in warning.value for warning in app.warning)


@pytest.mark.parametrize("draft", ["mode", "gross", "frequency", "period", "assets"])
def test_changing_draft_preserves_labeled_analysis_without_fetching_or_optimizing(irx_app, draft):
    create_app, _, calls, evidence = irx_app
    app = create_app()
    if draft == "gross":
        app = set_mode(app, "Short allowed", 200)
    app = select_and_go(app, ["AAPL", "MSFT"])
    assert_success(app)
    previous = frozen_results(app)
    provider_count = len(calls)
    optimizer_count = len(evidence["optimizations"])
    assert "AAPL, MSFT" in previous["summary"]
    assert "Period: 1y" in previous["summary"]
    assert "Frequency: Daily (252/year)" in previous["summary"]
    assert f"Mode: {'Short allowed' if draft == 'gross' else 'Long-only'}" in previous["summary"]
    assert f"Gross limit: {'200%' if draft == 'gross' else '100%'}" in previous["summary"]

    if draft == "mode":
        app.sidebar.radio(key="portfolio_mode").set_value("Short allowed").run()
    elif draft == "gross":
        app.sidebar.number_input(key="gross_limit_percent").set_value(300).run()
    elif draft == "frequency":
        app.sidebar.selectbox[1].set_value("1wk").run()
    elif draft == "period":
        app.sidebar.selectbox[0].set_value("2y").run()
    else:
        app.sidebar.multiselect[0].set_value(["AAPL"]).run()

    assert_same_results(app, previous)
    assert_pending(app)
    assert len(calls) == provider_count
    assert len(evidence["optimizations"]) == optimizer_count


def test_reruns_and_duplicate_submit_reuse_the_successful_analysis(irx_app):
    create_app, _, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_success(app)
    previous = frozen_results(app)
    provider_count = len(calls)
    app.run()
    assert_same_results(app, previous)
    app.sidebar.button[0].click().run()
    assert_same_results(app, previous)

    # Returning a draft to the analyzed setting removes its pending warning.
    app.sidebar.selectbox[0].set_value("2y").run()
    assert_pending(app)
    app.sidebar.selectbox[0].set_value("1y").run()
    assert_same_results(app, previous)
    assert not any("Pending changes" in element.value for element in app.warning)
    assert len(calls) == provider_count
    assert len(evidence["optimizations"]) == 1


@pytest.mark.parametrize("filter_index,value", [(0, "ETF"), (1, "NYSE")])
def test_filters_clear_invalid_selection_and_empty_submit_removes_old_analysis(irx_app, filter_index, value):
    create_app, _, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    previous = frozen_results(app)
    provider_count = len(calls)
    app.sidebar.radio[filter_index].set_value(value).run()

    assert_no_exception(app)
    assert app.sidebar.multiselect[0].value == []
    assert not {"AAPL", "MSFT"}.intersection(app.sidebar.multiselect[0].options)
    assert_same_results(app, previous)
    assert_pending(app)
    app.sidebar.button[0].click().run()

    assert_no_optimization_results(app)
    assert snapshot_or_none(app) is None
    assert not downloads(app)
    assert any("Select at least one asset" in warning.value for warning in app.warning)
    assert len(calls) == provider_count
    assert len(evidence["optimizations"]) == 1


def test_failed_new_optimization_replaces_old_metrics_preserves_new_csv_and_can_retry(irx_app, monkeypatch):
    create_app, responses, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    original_returns = downloaded_returns(app)
    frontier_render_count = len(evidence["clouds"])
    replacement_prices = prices_with_moments(means=(0.002, -0.0004))
    replacement_prices.index = replacement_prices.index + pd.DateOffset(years=1)
    responses["prices"] = replacement_prices
    successful_optimizer = portfolio.optimize_portfolio
    failed_attempts = []

    def fail_optimization(*args, **kwargs):
        failed_attempts.append(1)
        raise portfolio.PortfolioError("Offline regression: optimizer could not find a valid portfolio")

    monkeypatch.setattr(portfolio, "optimize_portfolio", fail_optimization)
    app.sidebar.selectbox[0].set_value("2y")
    app.sidebar.button[0].click().run()
    assert_no_optimization_results(app)
    assert app.error
    assert snapshot_or_none(app) is not None
    assert "Period: 2y" in analyzed_settings(app)
    assert len(downloads(app)) == 2
    new_returns = downloaded_returns(app)
    expected_returns = replacement_prices.xs("Close", axis=1, level=0).pct_change(fill_method=None).iloc[1:]
    pd.testing.assert_frame_equal(new_returns, expected_returns, check_freq=False, check_names=False,
                                  rtol=1e-10, atol=1e-14)
    assert not new_returns.index.equals(original_returns.index)
    assert len(failed_attempts) == 1
    assert len(price_calls(calls)) == 2
    assert len(evidence["clouds"]) == frontier_render_count

    app.run()
    assert_no_optimization_results(app)
    assert app.error
    assert len(failed_attempts) == 1
    assert len(price_calls(calls)) == 2
    assert len(evidence["clouds"]) == frontier_render_count
    pd.testing.assert_frame_equal(downloaded_returns(app), new_returns)

    monkeypatch.setattr(portfolio, "optimize_portfolio", successful_optimizer)
    app.sidebar.button[0].click().run()
    assert_success(app)
    # A retry may reuse its already valid input data or request one fresh copy.
    assert len(price_calls(calls)) in (2, 3)
    assert len(evidence["optimizations"]) == 2
    pd.testing.assert_frame_equal(downloaded_returns(app), new_returns)


def test_new_price_failure_does_not_present_previous_metrics_or_downloads(irx_app):
    create_app, responses, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_success(app)
    frontier_render_count = len(evidence["clouds"])
    responses["prices"] = TimeoutError("Offline price provider timeout")
    app.sidebar.selectbox[0].set_value("2y")
    app.sidebar.button[0].click().run()

    assert_no_optimization_results(app)
    assert app.error
    assert not downloads(app)
    assert "Period: 2y" in analyzed_settings(app)
    assert len(price_calls(calls)) == 2
    assert len(evidence["optimizations"]) == 1
    assert len(evidence["clouds"]) == frontier_render_count


def test_submitting_one_asset_replaces_a_previous_portfolio_without_stale_weights(irx_app):
    create_app, _, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_success(app)
    frontier_render_count = len(evidence["clouds"])
    treasury_calls = len([call for call in calls if call["tickers"] == "^IRX"])
    app = select_and_go(app, ["AAPL"])

    assert_no_optimization_results(app)
    assert not app.error
    assert snapshot_or_none(app) is not None
    assert len(downloads(app)) == 2
    assert downloaded_returns(app).columns.tolist() == ["AAPL"]
    assert len(price_calls(calls)) == 2
    assert price_calls(calls)[-1]["tickers"] == ["AAPL"]
    assert len([call for call in calls if call["tickers"] == "^IRX"]) == treasury_calls
    assert len(evidence["optimizations"]) == 1
    assert len(evidence["clouds"]) == frontier_render_count
    assert any("Select at least two assets to optimize a portfolio" in info.value for info in app.info)
    assert not any("Pending changes" in warning.value for warning in app.warning)


@pytest.mark.parametrize("symbol,mode", [("AAPL", "Long-only"), ("NA", "Short allowed")])
def test_single_asset_exports_prices_and_returns_without_rate_or_optimizer(irx_app, symbol, mode):
    create_app, responses, calls, evidence = irx_app
    responses["prices"] = None
    app = select_and_go(set_mode(create_app(), mode), [symbol])

    assert_no_optimization_results(app)
    assert not app.error
    assert snapshot_or_none(app) is not None
    assert len(downloads(app)) == 2
    returns = downloaded_returns(app)
    assert returns.columns.tolist() == [symbol]
    assert len(returns) == 23
    assert np.isfinite(returns.to_numpy()).all()
    assert any("Select at least two assets to optimize a portfolio" in info.value for info in app.info)
    assert calls == [{"tickers": [symbol], "period": "1y", "interval": "1d"}]
    assert not evidence["optimizations"]
    assert not evidence["clouds"]
    previous_downloads = downloads(app)
    app.sidebar.button[0].click().run()
    assert downloads(app) == previous_downloads
    assert len(calls) == 1


def test_refresh_uses_analyzed_parameters_while_preserving_a_different_draft(irx_app):
    create_app, responses, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    original_returns = downloaded_returns(app)
    app.sidebar.selectbox[0].set_value("2y").run()
    assert_pending(app)
    replacement_prices = prices_with_moments(means=(0.0018, 0.0003))
    replacement_prices.index = replacement_prices.index + pd.Timedelta(days=31)
    responses["prices"] = replacement_prices
    refresh_buttons = [button for button in app.button if button.label == "Refresh analyzed data"]
    assert len(refresh_buttons) == 1
    refresh_buttons[0].click().run()

    assert_success(app)
    assert_pending(app)
    assert app.sidebar.selectbox[0].value == "2y"
    assert "Period: 1y" in analyzed_settings(app)
    assert len(price_calls(calls)) == 2
    assert price_calls(calls)[-1]["period"] == "1y"
    assert price_calls(calls)[-1]["interval"] == "1d"
    assert len(evidence["optimizations"]) == 2
    assert not downloaded_returns(app).index.equals(original_returns.index)


def test_catalog_failure_on_rerun_preserves_valid_analysis(irx_app, monkeypatch):
    create_app, _, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    previous = frozen_results(app)
    provider_count = len(calls)

    def unavailable_catalog(*args, **kwargs):
        raise OSError("Offline symbol directory unavailable")

    monkeypatch.setattr(ftplib, "FTP", unavailable_catalog)
    st.cache_data.clear()
    app.run()

    assert_same_results(app, previous)
    assert snapshot_or_none(app) is not None
    assert len(calls) == provider_count
    assert len(evidence["optimizations"]) == 1
    messages = " ".join(element.value for element in list(app.error) + list(app.warning))
    assert "directory" in messages.lower() and "unavailable" in messages.lower()
