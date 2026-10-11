"""Annual portfolio analytics keep native observations and numeric exports."""

import base64
from io import StringIO
import re
import sys

from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit import dataframe_util
import yfinance as yf

import finance_data
import risk_free_irx_v1
from test_app import assert_no_exception, downloaded_returns, offline_app, select_and_go
from test_irx_app import ANNUAL_SIMPLE, PERIODS_PER_YEAR, assert_proxy_request, irx_app
from test_short_app import displayed_metrics, displayed_weights, prices_with_moments, set_mode


def frequency_prices(interval):
    """Different price bars, dates and sample sizes for each Yahoo frequency."""
    count, cadence, means, scale = {
        "1d": (48, "B", (0.001, 0.0), 1.0),
        "1wk": (20, "W-FRI", (0.003, 0.001), 2.5),
        "1mo": (12, "ME", (0.012, 0.004), 4.0),
        "3mo": (8, "QE-DEC", (0.025, 0.008), 7.0),
    }[interval]
    phase = 2 * np.pi * np.arange(count) / count
    basis = np.column_stack((np.sin(phase), np.cos(phase))) * np.sqrt(2 * (count - 1) / count)
    covariance = np.array([[0.0001, 0.00012], [0.00012, 0.0003]]) * scale
    returns = np.asarray(means) + basis @ np.linalg.cholesky(covariance).T
    baseline = np.array([100.0, 150.0])
    values = np.vstack((baseline, baseline * np.cumprod(1 + returns, axis=0)))
    return pd.DataFrame(
        values,
        index=pd.date_range("2022-01-03", periods=count + 1, freq=cadence, name="Date"),
        columns=pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Close", "MSFT")]),
    )


@pytest.fixture
def annual_app(irx_app, monkeypatch):
    create_app, responses, calls, evidence = irx_app
    bars = {interval: frequency_prices(interval) for interval in PERIODS_PER_YEAR}
    evidence["requests"] = []
    evidence["annual_figures"] = []
    original_download = yf.download

    def frequency_download(**kwargs):
        evidence["requests"].append(kwargs.copy())
        if kwargs["tickers"] == "^IRX":
            return original_download(**kwargs)
        selected = [("Close", symbol) for symbol in kwargs["tickers"]]
        return bars[kwargs["interval"]].loc[:, selected].copy()

    monkeypatch.setattr(yf, "download", frequency_download)
    original_pyplot = st.pyplot

    def capture_annual_figure(figure=None, *args, **kwargs):
        if figure is not None:
            for axis in figure.get_axes():
                if axis.get_title().startswith("Efficient Frontier and Random Portfolios"):
                    evidence["annual_figures"].append({
                        "title": axis.get_title(),
                        "xlabel": axis.get_xlabel(),
                        "ylabel": axis.get_ylabel(),
                        "xformatter": axis.xaxis.get_major_formatter(),
                        "yformatter": axis.yaxis.get_major_formatter(),
                        "collections": {
                            item.get_label(): np.asarray(item.get_offsets()).copy()
                            for item in axis.collections
                        },
                        "lines": {
                            line.get_label(): (np.asarray(line.get_xdata()).copy(),
                                               np.asarray(line.get_ydata()).copy())
                            for line in axis.lines
                        },
                    })
        return original_pyplot(figure, *args, **kwargs)

    monkeypatch.setattr(st, "pyplot", capture_annual_figure)
    return create_app, responses, calls, evidence, bars


def downloaded_csv(app, label, index_col):
    links = [element.value for element in app.markdown if label in element.value]
    assert len(links) == 1
    match = re.search(r'base64,([^\"]+)"', links[0])
    assert match is not None
    text = base64.b64decode(match.group(1)).decode()
    return pd.read_csv(StringIO(text), index_col=index_col,
                       parse_dates=[index_col] if index_col == "Date" else None)


def table_with_column(app, column):
    elements = [element for element in list(app.dataframe) + list(app.table)
                if column in element.value.columns]
    assert len(elements) == 1
    return elements[0]


def assert_percent_display_keeps_numeric_data(element, columns):
    payload = element.proto.arrow_data.styler.display_values
    assert payload, "Percentage formatting must reach the rendered table"
    display = dataframe_util.convert_arrow_bytes_to_pandas_df(payload)
    for column in columns:
        assert pd.api.types.is_numeric_dtype(element.value[column])
        assert display[column].tolist() == [f"{value:.2%}" for value in element.value[column]]
    return display


@pytest.mark.parametrize("mode", ["Long-only", "Short allowed"])
def test_started_session_recalculates_each_frequency_with_annual_inputs_and_native_csv(annual_app, mode):
    create_app, _, calls, evidence, bars = annual_app
    retained_modules = {"finance_data": finance_data, "risk_free_irx_v1": risk_free_irx_v1}
    app = create_app()
    frequency = app.sidebar.selectbox(key="return_frequency")
    assert frequency.value == "1d"
    assert frequency.options == [
        "Daily (252/year)", "Weekly (52/year)", "Monthly (12/year)", "Quarterly (4/year)"
    ]
    frequency_expanders = [
        expander for expander in app.sidebar.expander
        if expander.label == "Advanced: return frequency"
    ]
    assert len(frequency_expanders) == 1
    assert len(frequency_expanders[0].selectbox) == 1
    assert not evidence["requests"]
    app = set_mode(app, mode)
    observed_rows = []
    annual_means = []

    for interval, periods in PERIODS_PER_YEAR.items():
        app.sidebar.selectbox(key="return_frequency").set_value(interval)
        app = select_and_go(app, ["AAPL", "MSFT"])
        assert_no_exception(app)
        assert not app.error
        assert all(sys.modules[name] is module for name, module in retained_modules.items())
        assert "annual_moments_v1" in sys.modules

        expected_prices = bars[interval].xs("Close", axis=1, level=0)
        expected_returns = expected_prices.pct_change(fill_method=None).iloc[1:]
        exported_returns = downloaded_returns(app, interval)
        pd.testing.assert_frame_equal(exported_returns, expected_returns, check_freq=False,
                                      check_names=False, rtol=1e-10, atol=1e-14)
        pd.testing.assert_frame_equal(
            downloaded_csv(app, "Download Time Series", "Date"), expected_prices,
            check_freq=False, check_names=False, rtol=1e-10, atol=1e-12,
        )
        observed_rows.append(len(exported_returns))
        captured = evidence["optimizations"][-1]
        expected_mean = expected_returns.mean().to_numpy() * periods
        expected_covariance = expected_returns.cov().to_numpy() * periods
        np.testing.assert_allclose(captured["means"], expected_mean, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(captured["covariance"], expected_covariance, rtol=1e-10, atol=1e-13)
        assert captured["risk_free"] == pytest.approx(ANNUAL_SIMPLE, rel=1e-12)
        annual_means.append(captured["means"])
        weights = displayed_weights(app)
        solution = captured["result"]
        expected_return = weights @ expected_mean
        expected_risk = np.sqrt(weights @ expected_covariance @ weights)
        metrics = displayed_metrics(app)
        assert metrics["Annual Return"] == pytest.approx(expected_return, abs=1e-12)
        assert metrics["Annual Volatility"] == pytest.approx(expected_risk, abs=1e-12)
        assert metrics["Sharpe Ratio"] == pytest.approx((expected_return - ANNUAL_SIMPLE) / expected_risk)

        weight_csv = downloaded_csv(app, "Download Portfolio Weights", "Symbol")
        assert weight_csv.columns.tolist() == ["Opt Weight"]
        np.testing.assert_allclose(weight_csv.reindex(["AAPL", "MSFT"])["Opt Weight"], weights,
                                   rtol=0, atol=1e-12)
        weight_table = table_with_column(app, "Opt Weight")
        np.testing.assert_allclose(weight_table.value["Annual Mean Return"], expected_mean,
                                   rtol=1e-10, atol=1e-13)
        assert_percent_display_keeps_numeric_data(weight_table, ["Opt Weight", "Annual Mean Return"])
        metrics_table = table_with_column(app, "Sharpe Ratio")
        display = assert_percent_display_keeps_numeric_data(metrics_table, ["Annual Return", "Annual Volatility"])
        assert display["Sharpe Ratio"].iloc[0] == f"{metrics['Sharpe Ratio']:.3f}"

        chart = evidence["annual_figures"][-1]
        assert "Annual" in chart["title"]
        assert (chart["xlabel"], chart["ylabel"]) == ("Annual Volatility", "Annual Mean Return")
        for formatter in (chart["xformatter"], chart["yformatter"]):
            assert isinstance(formatter, PercentFormatter)
            assert formatter.xmax == 1
        suffix = " (Short allowed)" if mode == "Short allowed" else ""
        np.testing.assert_allclose(chart["collections"][f"Max Sharpe Ratio Portfolio{suffix}"],
                                   [[expected_risk, expected_return]], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(chart["collections"][f"Minimum Variance Portfolio{suffix}"],
                                   [[solution.min_risk, solution.min_return]], rtol=1e-10, atol=1e-12)
        frontier_x, frontier_y = chart["lines"][f"Efficient Frontier{suffix}"]
        np.testing.assert_allclose(frontier_y, solution.frontier_weights @ expected_mean, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(frontier_x, np.sqrt(np.einsum(
            "ij,jk,ik->i", solution.frontier_weights, expected_covariance, solution.frontier_weights
        )), rtol=1e-9, atol=1e-12)
        captions = " ".join(element.value for element in app.caption)
        assert f"{periods} periods/year ({interval})" in captions
        assert "In-sample estimates" in captions and "not forecasts or realized gains" in captions

    assert observed_rows == [48, 20, 12, 8]
    assert all(not np.allclose(annual_means[0], other) for other in annual_means[1:])
    assert_proxy_request(calls)
    requests = [request for request in evidence["requests"] if isinstance(request["tickers"], list)]
    assert len(requests) == 4
    assert [request["interval"] for request in requests] == list(PERIODS_PER_YEAR)
    assert all(request["auto_adjust"] is True for request in requests)


@pytest.mark.parametrize("means", [(0.0, 0.0), (-0.002, -0.003)], ids=["zero", "negative"])
def test_zero_and_negative_annual_returns_are_numeric_and_format_as_percent(annual_app, means):
    create_app, _, _, evidence, bars = annual_app
    bars["1d"] = prices_with_moments(means=means)
    app = select_and_go(create_app(), ["AAPL", "MSFT"])
    assert_no_exception(app)
    assert not app.error
    metrics_table = table_with_column(app, "Sharpe Ratio")
    display = assert_percent_display_keeps_numeric_data(metrics_table, ["Annual Return", "Annual Volatility"])
    result = evidence["optimizations"][-1]["result"]
    assert displayed_metrics(app)["Annual Return"] == pytest.approx(result.max_return, abs=1e-12)
    assert displayed_metrics(app)["Sharpe Ratio"] < 0
    if means == (0.0, 0.0):
        assert abs(result.max_return) < 1e-12
        assert display["Annual Return"].iloc[0] in ("0.00%", "-0.00%")
    else:
        assert result.max_return < 0
        assert display["Annual Return"].iloc[0].startswith("-")
    weight_table = table_with_column(app, "Opt Weight")
    assert_percent_display_keeps_numeric_data(weight_table, ["Opt Weight", "Annual Mean Return"])
    assert np.isfinite(downloaded_returns(app).to_numpy()).all()


def test_single_asset_also_requests_adjusted_bars_without_a_treasury_quote(annual_app):
    create_app, _, calls, evidence, bars = annual_app
    app = create_app()
    app.sidebar.selectbox(key="return_frequency").set_value("1wk")
    app = select_and_go(app, ["AAPL"])
    assert_no_exception(app)
    assert not app.error
    assert evidence["requests"] == [{
        "tickers": ["AAPL"], "period": "1y", "interval": "1wk", "auto_adjust": True,
    }]
    assert not calls
    assert not evidence["optimizations"]
    assert any("Download Time Series" in element.value for element in app.markdown)
    expected_prices = bars["1wk"].xs("Close", axis=1, level=0)[["AAPL"]]
    # Preserve the original single-asset raw-price export schema, including
    # its provider column levels and Date column after reset_index.
    price_links = [element.value for element in app.markdown if "Download Time Series" in element.value]
    assert len(price_links) == 1
    price_payload = re.search(r'base64,([^\"]+)"', price_links[0])
    assert price_payload is not None
    raw_prices = bars["1wk"].loc[:, [("Close", "AAPL")]]
    assert base64.b64decode(price_payload.group(1)).decode() == raw_prices.reset_index(level=0).to_csv(index=True)
    pd.testing.assert_frame_equal(
        downloaded_returns(app, "1wk"), expected_prices.pct_change(fill_method=None).iloc[1:],
        check_freq=False, check_names=False, rtol=1e-10, atol=1e-14,
    )
    assert any("Select at least two assets to optimize a portfolio" in item.value for item in app.info)
