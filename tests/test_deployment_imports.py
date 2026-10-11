"""A retained pre-short-selling import must not break the current app.

The legacy module below is a real module loaded from a temporary file, with
the old public signature. It needs neither repository history nor saved local
evidence, so this regression runs identically in CI and on a clean checkout.
"""

import importlib
import importlib.util
import inspect
import sys

import numpy as np
import pytest

from test_app import assert_no_exception, downloaded_returns, offline_app, select_and_go
from test_short_app import chart_collections, displayed_metrics, displayed_weights, prices_with_moments, set_mode


LEGACY_SOURCE = '''
class PortfolioError(ValueError):
    pass

calls = []
legacy_instance_state = object()

def optimize_portfolio(mean_returns, covariance, risk_free_rate, points=50):
    calls.append((mean_returns, covariance, risk_free_rate, points))
    raise AssertionError("The current app must not invoke the retained legacy optimizer")
'''


@pytest.fixture
def retained_legacy_portfolio(monkeypatch, tmp_path):
    legacy_path = tmp_path / "previous_deployment" / "portfolio.py"
    legacy_path.parent.mkdir()
    legacy_path.write_text(LEGACY_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("portfolio", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "portfolio", legacy)
    spec.loader.exec_module(legacy)
    original_namespace = legacy.__dict__.copy()
    signature = "(mean_returns, covariance, risk_free_rate, points=50)"
    assert str(inspect.signature(legacy.optimize_portfolio)) == signature

    original_reload = importlib.reload

    def prohibit_legacy_reload(module):
        assert module is not legacy, "The current app must not reload a shared legacy module"
        return original_reload(module)

    monkeypatch.setattr(importlib, "reload", prohibit_legacy_reload)

    def assert_untouched():
        assert sys.modules.get("portfolio") is legacy, "The cached legacy module must not be evicted or replaced"
        assert legacy.__dict__.keys() == original_namespace.keys()
        assert all(legacy.__dict__[name] is value for name, value in original_namespace.items())
        assert not legacy.calls
        assert str(inspect.signature(legacy.optimize_portfolio)) == signature

    yield assert_untouched
    assert_untouched()


@pytest.mark.parametrize(
    "mode,budget,expected_weights",
    [("Long-only", 1.0, [1.0, 0.0]), ("Short allowed", 2.0, [1.5, -0.5])],
)
def test_current_app_uses_new_module_without_touching_retained_legacy(
    offline_app, retained_legacy_portfolio, monkeypatch, mode, budget, expected_weights
):
    import portfolio_modes_v1

    create_app, responses, _ = offline_app
    responses["prices"] = prices_with_moments()
    observed = []
    actual_optimizer = portfolio_modes_v1.optimize_portfolio

    def observe_current_optimizer(*args, **kwargs):
        solution = actual_optimizer(*args, **kwargs)
        observed.append((kwargs.copy(), solution))
        return solution

    monkeypatch.setattr(portfolio_modes_v1, "optimize_portfolio", observe_current_optimizer)
    app = create_app()
    retained_legacy_portfolio()
    assert app.sidebar.radio(key="portfolio_mode").value == "Long-only"
    app = set_mode(app, mode, int(budget * 100))
    retained_legacy_portfolio()
    app = select_and_go(app, ["AAPL", "MSFT"])

    assert_no_exception(app)
    assert not app.error
    retained_legacy_portfolio()
    assert len(observed) == 1
    options, solution = observed[0]
    assert options == {"allow_short": mode == "Short allowed", "gross_limit": budget}
    weights = displayed_weights(app)
    np.testing.assert_allclose(weights, expected_weights, rtol=0, atol=2e-6)
    np.testing.assert_allclose(weights, solution.weights, rtol=0, atol=1e-12)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1, abs=1e-7)
    assert np.abs(weights).sum() <= budget + 1e-7
    if mode == "Short allowed":
        assert weights[1] < -0.1, "The regression must exercise an actual negative position"

    returns = downloaded_returns(app)
    expected_return = float(weights @ (returns.mean().to_numpy() * 252))
    expected_risk = float(np.sqrt(weights @ (returns.cov().to_numpy() * 252) @ weights))
    risk_free = 365 * 0.042 / (360 - 91 * 0.042)
    metrics = displayed_metrics(app)
    assert metrics["Portfolio"] == f"{mode} Max Sharpe Ratio"
    assert metrics["Annual Return"] == pytest.approx(expected_return, rel=1e-7, abs=1e-10)
    assert metrics["Annual Volatility"] == pytest.approx(expected_risk, rel=1e-7, abs=1e-10)
    assert metrics["Sharpe Ratio"] == pytest.approx((expected_return - risk_free) / expected_risk, rel=1e-7)
    suffix = " (Short allowed)" if mode == "Short allowed" else ""
    markers = chart_collections(responses, f"Max Sharpe Ratio Portfolio{suffix}")
    assert len(markers) == 1
    np.testing.assert_allclose(markers[0], [[expected_risk, expected_return]], rtol=1e-7, atol=1e-10)


def test_compatibility_facade_reexports_the_same_public_api_objects():
    import portfolio
    import portfolio_modes_v1

    public_names = {"PortfolioError", "PortfolioResult", "optimize_portfolio", "portfolio_metrics"}
    assert set(portfolio.__all__) == public_names
    for name in public_names:
        assert getattr(portfolio, name) is getattr(portfolio_modes_v1, name)
