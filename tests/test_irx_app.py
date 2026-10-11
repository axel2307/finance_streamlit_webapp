"""Offline UI regressions for the dated, cached Treasury bill proxy."""

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
    treasury_yield,
)
from test_short_app import (
    displayed_metrics,
    displayed_weights,
    prices_with_moments,
    set_mode,
)


# Independent expectations: Yahoo's level is percent, the bill discount uses
# a 360-day basis, and investment returns use simple interest (not compounding).
DISCOUNT = 4.2 / 100
ANNUAL_SIMPLE = 365 * DISCOUNT / (360 - 91 * DISCOUNT)
INTERVAL_DAYS = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90}


@pytest.fixture
def irx_app(offline_app, monkeypatch):
    create_app, responses, calls = offline_app
    responses["prices"] = prices_with_moments()
    evidence = {"optimizations": [], "clouds": []}
    original_optimizer = portfolio.optimize_portfolio

    def capture_optimizer(means, covariance, risk_free, **kwargs):
        result = original_optimizer(means, covariance, risk_free, **kwargs)
        evidence["optimizations"].append({
            "means": np.asarray(means).copy(),
            "covariance": np.asarray(covariance).copy(),
            "risk_free": risk_free,
            "kwargs": kwargs.copy(),
            "result": result,
        })
        return result

    monkeypatch.setattr(portfolio, "optimize_portfolio", capture_optimizer)
    monkeypatch.setattr(np.random, "uniform", np.random.default_rng(54321).uniform)
    original_pyplot = st.pyplot

    def capture_cloud(figure=None, *args, **kwargs):
        if figure is not None:
            for axis in figure.get_axes():
                for collection in axis.collections:
                    if collection.get_label().startswith("Random Portfolios"):
                        evidence["clouds"].append({
                            "offsets": np.asarray(collection.get_offsets()).copy(),
                            "sharpes": np.asarray(collection.get_array()).copy(),
                        })
        return original_pyplot(figure, *args, **kwargs)

    monkeypatch.setattr(st, "pyplot", capture_cloud)
    return create_app, responses, calls, evidence


def irx_calls(calls):
    return [call for call in calls if call["tickers"] == "^IRX"]


def assert_proxy_request(calls, count=1):
    assert irx_calls(calls) == [
        {"tickers": "^IRX", "period": "5d", "interval": "1d"}
    ] * count
    assert not any(call["tickers"] == "^TNX" for call in calls)


def assert_proxy_labels(app, interval):
    captions = " ".join(element.value for element in app.caption)
    for detail in (
        "Yahoo Finance", "^IRX", "2025-02-04", "13-week", "91 days",
        "remaining maturity unknown", "Bank discount quote: 4.200%",
    ):
        assert detail in captions
    markdown = " ".join(element.value for element in app.markdown)
    assert (
        f"Risk-free proxy (annual simple investment yield): {ANNUAL_SIMPLE * 100:.3f}%"
        in markdown
    )
    expected_rf = ANNUAL_SIMPLE * INTERVAL_DAYS[interval] / 365
    assert f"Risk-free return for interval {interval}: {expected_rf * 100:.6f}%" in markdown


def assert_interval_calculations(app, evidence, interval, mode):
    assert_no_exception(app)
    assert not app.error
    expected_rf = ANNUAL_SIMPLE * INTERVAL_DAYS[interval] / 365
    captured = evidence["optimizations"][-1]
    assert captured["risk_free"] == pytest.approx(expected_rf, rel=1e-12, abs=1e-15)
    assert captured["kwargs"] == {
        "allow_short": mode == "Short allowed",
        "gross_limit": 2.0 if mode == "Short allowed" else 1.0,
    }

    returns = downloaded_returns(app, interval)
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert len(returns) == 24
    means = returns.mean().to_numpy()
    covariance = returns.cov().to_numpy()
    np.testing.assert_allclose(captured["means"], means, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(captured["covariance"], covariance, rtol=1e-10, atol=1e-14)
    weights = displayed_weights(app)
    expected_return = float(weights @ means)
    expected_risk = float(np.sqrt(weights @ covariance @ weights))
    metrics = displayed_metrics(app)
    assert metrics["Portfolio"] == f"{mode} Max Sharpe Ratio"
    assert metrics["Return"] == pytest.approx(expected_return, rel=1e-9, abs=1e-12)
    assert metrics["Std Dev"] == pytest.approx(expected_risk, rel=1e-9, abs=1e-12)
    assert metrics["Sharpe Ratio"] == pytest.approx(
        (expected_return - expected_rf) / expected_risk, rel=1e-9, abs=1e-12
    )
    np.testing.assert_allclose(weights, captured["result"].weights, rtol=0, atol=1e-12)
    cloud = evidence["clouds"][-1]
    assert cloud["offsets"].shape == (1000, 2)
    np.testing.assert_allclose(
        cloud["sharpes"],
        (cloud["offsets"][:, 1] - expected_rf) / cloud["offsets"][:, 0],
        rtol=1e-10, atol=1e-12,
    )
    assert_proxy_labels(app, interval)


@pytest.mark.parametrize("mode", ["Long-only", "Short allowed"])
@pytest.mark.parametrize("interval", ["1d", "5d", "1mo", "3mo"])
def test_irx_units_provenance_and_sharpe_for_every_interval_and_mode(irx_app, interval, mode):
    create_app, _, calls, evidence = irx_app
    app = set_mode(create_app(), mode)
    app.sidebar.selectbox[1].set_value(interval)
    app = select_and_go(app, ["AAPL", "MSFT"])

    assert_proxy_request(calls)
    assert_interval_calculations(app, evidence, interval, mode)


def test_valid_irx_quote_is_cached_across_go_modes_and_intervals(irx_app):
    create_app, responses, calls, evidence = irx_app
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_interval_calculations(app, evidence, "1d", "Long-only")
    assert_proxy_request(calls)

    # A new provider value must not replace the valid cached quote on every Go.
    responses["yield"] = treasury_yield(value=9.9)
    app.sidebar.button[0].click().run()
    assert_interval_calculations(app, evidence, "1d", "Long-only")

    app = set_mode(app, "Short allowed")
    app.sidebar.button[0].click().run()
    assert_interval_calculations(app, evidence, "1d", "Short allowed")

    app.sidebar.selectbox[1].set_value("5d")
    app.sidebar.button[0].click().run()
    assert_interval_calculations(app, evidence, "5d", "Short allowed")
    assert_proxy_request(calls)
    assert len(evidence["optimizations"]) == 4
    assert len([call for call in calls if isinstance(call["tickers"], list)]) == 4


@pytest.mark.parametrize("mode", ["Long-only", "Short allowed"])
@pytest.mark.parametrize("failure", ["download", "empty", "invalid_quote"])
def test_irx_failure_preserves_csv_and_recovers_without_caching_failure(irx_app, mode, failure):
    create_app, responses, calls, evidence = irx_app
    responses["yield"] = {
        "download": TimeoutError("offline Treasury download timeout"),
        "empty": pd.DataFrame(),
        "invalid_quote": treasury_yield(value=np.nan),
    }[failure]
    app = select_and_go(set_mode(create_app(), mode), ["AAPL", "MSFT"])

    assert_no_portfolio_results(app)
    assert "^IRX" in " ".join(error.value for error in app.error)
    assert_valid_downloads_remain_available(app, expected_return_rows=24)
    original_returns = downloaded_returns(app)
    assert not evidence["optimizations"]
    assert not evidence["clouds"]
    assert_proxy_request(calls)

    responses["yield"] = treasury_yield()
    app.sidebar.button[0].click().run()
    assert_interval_calculations(app, evidence, "1d", mode)
    assert_proxy_request(calls, count=2)
    pd.testing.assert_frame_equal(downloaded_returns(app), original_returns)
