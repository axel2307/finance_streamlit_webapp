"""Validated, fully invested long-only portfolio analysis.

All returns, covariances and the risk-free rate must refer to the same interval.
This module deliberately does not choose an annualization or rate convention.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class PortfolioError(ValueError):
    """Inputs or an optimization result cannot support reliable analysis."""


@dataclass(frozen=True)
class PortfolioResult:
    weights: np.ndarray
    max_return: float
    max_risk: float
    max_sharpe: float
    min_weights: np.ndarray
    min_return: float
    min_risk: float
    frontier_returns: np.ndarray
    frontier_risks: np.ndarray
    frontier_weights: np.ndarray
    all_excess_returns_nonpositive: bool


_FEASIBILITY_TOL = 1e-7
_SOLVER_ERROR = (
    "The optimizer could not find a reliable portfolio. "
    "Try another asset selection or a longer period."
)


def _inputs(mean_returns, covariance, risk_free_rate):
    try:
        means = np.asarray(mean_returns, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        rf = float(risk_free_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError("Returns, covariance and the risk-free rate must be numeric.") from exc
    if means.ndim != 1 or not len(means):
        raise PortfolioError("Select at least one asset with a valid return history.")
    if cov.shape != (len(means), len(means)):
        raise PortfolioError("The covariance matrix must match the selected assets.")
    if not np.isfinite(means).all() or not np.isfinite(cov).all() or not np.isfinite(rf):
        raise PortfolioError("Prices must provide finite returns and covariance, and a valid risk-free rate.")
    scale = float(np.max(np.abs(cov)))
    if scale <= 0:
        raise PortfolioError("The selected assets have no measurable independent risk. Choose another selection or period.")
    scaled = cov / scale
    if not np.allclose(scaled, scaled.T, rtol=1e-10, atol=1e-12):
        raise PortfolioError("The covariance matrix is inconsistent. Choose another selection or period.")
    # Average only symmetry differences at floating-point precision; no shrinkage
    # or pseudo-inverse is used to turn an invalid covariance into a valid one.
    scaled = (scaled + scaled.T) / 2
    try:
        eigenvalues = np.linalg.eigvalsh(scaled)
        # A sample covariance may be mathematically rank deficient while
        # roundoff permits Cholesky. Reject rank below machine precision,
        # without an economic cutoff, ridge adjustment or pseudo-inverse.
        rank_tolerance = np.finfo(float).eps * len(means) * eigenvalues[-1]
        if eigenvalues[0] <= rank_tolerance:
            raise np.linalg.LinAlgError("Numerically rank-deficient covariance")
        np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError as exc:
        raise PortfolioError(
            "The selected assets do not provide independent risk information. "
            "Choose a longer period or remove duplicate or dependent assets."
        ) from exc
    with np.errstate(over="ignore", invalid="ignore"):
        excess = means - rf
    if not np.isfinite(excess).all():
        raise PortfolioError("The return and risk-free rate scales cannot be compared reliably.")
    return means, scaled, scale, rf, excess


def _weights(weights, count):
    try:
        values = np.asarray(weights, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError("Portfolio weights must be finite numbers.") from exc
    if values.shape != (count,) or not np.isfinite(values).all():
        raise PortfolioError("Portfolio weights must match the assets and be finite.")
    if (
        np.min(values) < -_FEASIBILITY_TOL
        or np.max(values) > 1 + _FEASIBILITY_TOL
        or abs(float(np.sum(values)) - 1) > _FEASIBILITY_TOL
    ):
        raise PortfolioError("Portfolio weights must be between 0 and 1 and sum to 1.")
    # Remove only solver roundoff after checking the original weights.
    values = np.clip(values, 0, 1)
    return values / np.sum(values)


def _metrics(weights, means, scaled_cov, cov_scale, excess):
    ret = float(weights @ means)
    variance = float(weights @ scaled_cov @ weights)
    if variance <= 0 or not np.isfinite(variance):
        raise PortfolioError("The portfolio risk cannot be estimated reliably for these assets.")
    risk = float(np.sqrt(variance) * np.sqrt(cov_scale))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sharpe = float((weights @ excess) / risk)
    if not np.isfinite([ret, risk, sharpe]).all() or risk <= 0:
        raise PortfolioError("The portfolio metrics are not finite. Choose another selection or period.")
    return ret, risk, sharpe


def portfolio_metrics(weights, mean_returns, covariance, risk_free_rate):
    """Return (mean return, standard deviation, Sharpe) per input interval."""
    means, scaled, scale, _, excess = _inputs(mean_returns, covariance, risk_free_rate)
    return _metrics(_weights(weights, len(means)), means, scaled, scale, excess)


def _quadratic_minimum(cov, equality, targets, initial, *, simplex=True):
    """Solve a scaled convex QP and reject unsuccessful or infeasible results."""
    equality = np.atleast_2d(np.asarray(equality, dtype=float))
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    initial = np.asarray(initial, dtype=float)
    count = len(initial)
    constraints = {
        "type": "eq",
        "fun": lambda x: equality @ x - targets,
        "jac": lambda x: equality,
    }
    try:
        result = minimize(
            lambda x: 0.5 * float(x @ cov @ x),
            initial,
            jac=lambda x: cov @ x,
            method="SLSQP",
            bounds=[(0, 1 if simplex else None)] * count,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    if not getattr(result, "success", False):
        raise PortfolioError(_SOLVER_ERROR)
    try:
        solution = np.asarray(result.x, dtype=float)
        objective = float(result.fun)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    if (
        solution.shape != (count,)
        or not np.isfinite(solution).all()
        or not np.isfinite(objective)
        or np.min(solution) < -_FEASIBILITY_TOL
        or (simplex and np.max(solution) > 1 + _FEASIBILITY_TOL)
    ):
        raise PortfolioError(_SOLVER_ERROR)
    residual = equality @ solution - targets
    if not np.isfinite(residual).all() or np.max(np.abs(residual)) > _FEASIBILITY_TOL:
        raise PortfolioError(_SOLVER_ERROR)
    solution = _weights(solution, count) if simplex else np.maximum(solution, 0)
    if np.max(np.abs(equality @ solution - targets)) > _FEASIBILITY_TOL:
        raise PortfolioError(_SOLVER_ERROR)
    if not np.isfinite(solution @ cov @ solution):
        raise PortfolioError(_SOLVER_ERROR)
    if len(targets) == 1:
        # For min-variance (sum(w)=1) and the transformed Sharpe QP
        # (a.y=1), the equality multiplier is x.T @ C @ x. Check the
        # convex problem's KKT conditions as well as the success flag.
        gradient = cov @ solution
        quadratic = float(solution @ gradient)
        reduced_gradient = gradient - quadratic * equality[0]
        gradient_scale = max(1.0, float(np.max(np.abs(gradient))),
                             abs(quadratic) * float(np.max(np.abs(equality[0]))))
        if (
            np.min(reduced_gradient) < -1e-6 * gradient_scale
            or np.max(np.abs(solution * reduced_gradient)) > 1e-6 * max(1.0, abs(quadratic))
        ):
            raise PortfolioError(_SOLVER_ERROR)
    return solution


def _minimum_variance(cov):
    count = len(cov)
    if count == 1:
        return np.ones(1)
    return _quadratic_minimum(cov, np.ones(count), [1], np.full(count, 1 / count))


def _maximum_sharpe(cov, excess):
    count = len(excess)
    if np.max(excess) <= 0:
        # With a_i=-excess_i >= 0 and m=min(a_i/sigma_i),
        # a.w >= m*sum(w_i*sigma_i) >= m*sqrt(w.T @ cov @ w).
        # Thus Sharpe(w) <= -m, attained by a vertex. This includes zero excess.
        weights = np.zeros(count)
        weights[int(np.argmax(excess / np.sqrt(np.diag(cov))))] = 1
        return weights
    # Positive Sharpe: y = w/(excess.w) turns the ratio problem into
    # min y.T @ cov @ y subject to excess.y=1, y>=0. All assets remain
    # available, including nonpositive-excess assets that provide hedging.
    normalized_excess = excess / np.max(excess)
    if not np.isfinite(normalized_excess).all():
        raise PortfolioError("The asset return scales cannot be optimized reliably.")
    initial = np.zeros(count)
    initial[int(np.argmax(excess))] = 1
    solution = _quadratic_minimum(
        cov, normalized_excess, [1], initial, simplex=False
    )
    # Scale before summing to avoid overflow for an otherwise finite solution.
    largest = float(np.max(solution))
    if largest <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    solution = solution / largest
    return _weights(solution / np.sum(solution), count)


def optimize_portfolio(mean_returns, covariance, risk_free_rate, points=50):
    """Optimize and plot-ready frontier under one shared long-only policy.

    Rejects singular/indefinite covariance rather than changing the model.
    Equal means produce the single minimum-variance efficient portfolio.
    """
    means, cov, scale, _, excess = _inputs(mean_returns, covariance, risk_free_rate)
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 2:
        raise PortfolioError("The frontier needs at least two requested points.")
    count = len(means)
    min_weights = _minimum_variance(cov)
    min_return, min_risk, _ = _metrics(min_weights, means, cov, scale, excess)
    weights = _maximum_sharpe(cov, excess) if count > 1 else np.ones(1)
    max_return, max_risk, max_sharpe = _metrics(weights, means, cov, scale, excess)

    # Do not use np.allclose: financially distinct small returns must retain
    # their frontier. Scaling the return constraint handles near-equal means.
    if np.all(means == means[0]):
        frontier_weights = np.array([min_weights])
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            relative_returns = (means - np.min(means)) / (np.max(means) - np.min(means))
        if not np.isfinite(relative_returns).all():
            raise PortfolioError("The return range cannot be optimized reliably.")
        maximum_assets = np.flatnonzero(means == np.max(means))
        endpoint = np.zeros(count)
        endpoint[maximum_assets] = _minimum_variance(cov[np.ix_(maximum_assets, maximum_assets)])
        start = float(relative_returns @ min_weights)
        if start >= 1 - np.finfo(float).eps:
            frontier_weights = np.array([min_weights])
        else:
            targets = np.linspace(start, 1, points)
            solutions = [min_weights]
            equality = np.vstack((np.ones(count), relative_returns))
            for fraction, target in zip(np.linspace(0, 1, points)[1:-1], targets[1:-1]):
                initial = (1 - fraction) * min_weights + fraction * endpoint
                solutions.append(_quadratic_minimum(cov, equality, [1, target], initial))
            solutions.append(endpoint)
            frontier_weights = np.array(solutions)
    frontier_metrics = [
        _metrics(w, means, cov, scale, excess) for w in frontier_weights
    ]
    return PortfolioResult(
        weights=weights,
        max_return=max_return,
        max_risk=max_risk,
        max_sharpe=max_sharpe,
        min_weights=min_weights,
        min_return=min_return,
        min_risk=min_risk,
        frontier_returns=np.array([row[0] for row in frontier_metrics]),
        frontier_risks=np.array([row[1] for row in frontier_metrics]),
        frontier_weights=frontier_weights,
        all_excess_returns_nonpositive=bool(np.max(excess) <= 0),
    )
