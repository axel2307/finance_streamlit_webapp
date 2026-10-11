"""Validated, fully invested portfolios with a shared exposure policy.

All returns, covariances and the risk-free rate must refer to the same interval.
This module deliberately does not choose an annualization or rate convention.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize


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
    allow_short: bool = False
    gross_limit: float = 1.0


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


def _policy(allow_short, gross_limit):
    if not isinstance(allow_short, (bool, np.bool_)):
        raise PortfolioError("The short-selling option must be true or false.")
    if not allow_short:
        return 1.0
    if isinstance(gross_limit, (bool, np.bool_)):
        raise PortfolioError("The gross exposure limit must be a finite number of at least 1 (100%).")
    try:
        gross = float(gross_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError("The gross exposure limit must be a finite number of at least 1 (100%).") from exc
    if not np.isfinite(gross) or gross < 1:
        raise PortfolioError("The gross exposure limit must be a finite number of at least 1 (100%).")
    return gross


def _weights(weights, count, *, allow_short=False, gross_limit=1.0):
    try:
        values = np.asarray(weights, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError("Portfolio weights must be finite numbers.") from exc
    if values.shape != (count,) or not np.isfinite(values).all():
        raise PortfolioError("Portfolio weights must match the assets and be finite.")
    if allow_short:
        if (
            abs(float(np.sum(values)) - 1) > _FEASIBILITY_TOL
            or float(np.sum(np.abs(values))) > gross_limit + _FEASIBILITY_TOL
        ):
            raise PortfolioError("Portfolio weights must sum to 1 and respect the gross exposure limit.")
        # Signed weights must never be clipped to long-only bounds.
        values = values / np.sum(values)
        if float(np.sum(np.abs(values))) > gross_limit + _FEASIBILITY_TOL:
            raise PortfolioError("Portfolio weights exceed the gross exposure limit.")
        return values
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


def portfolio_metrics(weights, mean_returns, covariance, risk_free_rate, *, allow_short=False, gross_limit=2.0):
    """Return (mean return, standard deviation, Sharpe) per input interval."""
    means, scaled, scale, _, excess = _inputs(mean_returns, covariance, risk_free_rate)
    gross = _policy(allow_short, gross_limit)
    return _metrics(
        _weights(weights, len(means), allow_short=allow_short, gross_limit=gross),
        means, scaled, scale, excess,
    )


def _unconstrained_minimum(cov, equality, targets):
    """Equality-constrained minimum in whitened coordinates; also a risk bound."""
    factor = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(factor, equality.T).T
    row_scales = np.linalg.norm(whitened, axis=1)
    if not np.isfinite(row_scales).all() or np.min(row_scales) <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    normalized = whitened / row_scales[:, None]
    minimum, _, rank, _ = np.linalg.lstsq(normalized, targets / row_scales, rcond=None)
    if rank != len(targets):
        raise PortfolioError(_SOLVER_ERROR)
    return np.linalg.solve(factor.T, minimum), float(np.linalg.norm(minimum))


def _certify_optimality(cov, solution, equality, targets):
    """Check relative primal/dual KKT errors for any number of equalities."""
    gradient = cov @ solution
    quadratic = float(solution @ gradient)
    _certify_nonnegative_qp(solution, equality, targets, gradient, quadratic)


def _certify_nonnegative_qp(solution, equality, targets, gradient, quadratic):
    """KKT certificate for a convex QP with equality and nonnegative variables.

    The short-selling lift is positive semidefinite, not positive definite.
    Its gradient and objective are evaluated through the original positions
    to avoid subtracting large, nearly equal blocks of an augmented Hessian.
    """
    gradient_scale = float(np.max(np.abs(gradient)))
    if not np.isfinite([quadratic, gradient_scale]).all() or min(quadratic, gradient_scale) <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    active = solution > np.finfo(float).eps * len(solution) * np.max(solution)
    tolerance = 2e-6

    def valid(multiplier):
        dual_gradient = equality.T @ multiplier
        reduced = gradient - dual_gradient
        scale = max(gradient_scale, float(np.max(np.abs(dual_gradient))))
        # No absolute floor: even a very small objective must have small
        # *relative* complementarity and primal/dual gap.
        return (
            np.isfinite(reduced).all()
            and np.min(reduced) >= -tolerance * scale
            and np.sum(np.abs(solution * reduced)) <= tolerance * quadratic
            and abs(quadratic - float(targets @ multiplier)) <= tolerance * quadratic
        )

    multiplier, _, rank, _ = np.linalg.lstsq(equality[:, active].T, gradient[active], rcond=None)
    if valid(multiplier):
        return
    if rank == len(targets):
        raise PortfolioError(_SOLVER_ERROR)
    # At a frontier vertex the active equality rows can be dependent; the
    # multiplier is then not unique. Seek a feasible dual certificate instead
    # of treating the arbitrary minimum-norm multiplier as the only one.
    normalized_gradient = gradient / gradient_scale
    dual = linprog(
        np.zeros(len(targets)),
        A_ub=np.vstack((equality.T, -equality[:, active].T)),
        b_ub=np.r_[normalized_gradient + 1e-9, -normalized_gradient[active] + 1e-9],
        bounds=[(None, None)] * len(targets),
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    if not dual.success or not np.isfinite(dual.x).all() or not valid(dual.x * gradient_scale):
        raise PortfolioError(_SOLVER_ERROR)


def _quadratic_minimum(cov, equality, targets, initial, *, simplex=True):
    """Solve in risk units, with an objective scale derived from this QP."""
    equality = np.atleast_2d(np.asarray(equality, dtype=float))
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    initial = np.asarray(initial, dtype=float)
    count = len(initial)
    deviations = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(deviations, deviations)
    risk_equality = equality / deviations
    try:
        unconstrained, _ = _unconstrained_minimum(correlation, risk_equality, targets)
    except np.linalg.LinAlgError as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    unconstrained_original = unconstrained / deviations
    if (
        np.min(unconstrained_original) >= 0
        and (not simplex or np.max(unconstrained_original) <= 1)
        and np.max(np.abs(equality @ unconstrained_original - targets)) <= _FEASIBILITY_TOL
    ):
        initial = unconstrained_original
    # x_i = (risk_scale / sigma_i) * z_i. Correlation preconditions each
    # variable. Scale to a feasible portfolio's risk after using the exact
    # unconstrained solution when it is long-only. Using the unconstrained
    # risk as the scale when it requires shorts can instead create enormous
    # objectives on a high-return, long-only frontier point.
    initial_risk_units = deviations * initial
    risk_scale = float(np.sqrt(initial_risk_units @ correlation @ initial_risk_units))
    if not np.isfinite(risk_scale) or risk_scale <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    transform = risk_scale / deviations
    transformed_equality = equality * transform
    row_scales = np.linalg.norm(transformed_equality, axis=1)
    transformed_equality = transformed_equality / row_scales[:, None]
    transformed_targets = targets / row_scales
    constraints = {
        "type": "eq",
        "fun": lambda z: transformed_equality @ z - transformed_targets,
        "jac": lambda z: transformed_equality,
    }
    try:
        result = minimize(
            lambda z: 0.5 * float(z @ correlation @ z),
            initial / transform,
            jac=lambda z: correlation @ z,
            method="SLSQP",
            bounds=[(0, 1 / value if simplex else None) for value in transform],
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    if not getattr(result, "success", False):
        raise PortfolioError(_SOLVER_ERROR)
    try:
        transformed_solution = np.asarray(result.x, dtype=float)
        objective = float(result.fun)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    if (
        transformed_solution.shape != (count,)
        or not np.isfinite(transformed_solution).all()
        or not np.isfinite(objective)
    ):
        raise PortfolioError(_SOLVER_ERROR)
    solution = transformed_solution * transform
    if (
        not np.isfinite(solution).all()
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
    final_transformed = solution / transform
    if np.max(np.abs(transformed_equality @ final_transformed - transformed_targets)) > _FEASIBILITY_TOL:
        raise PortfolioError(_SOLVER_ERROR)
    _certify_optimality(correlation, final_transformed, transformed_equality, transformed_targets)
    return solution


def _minimum_variance(cov):
    count = len(cov)
    if count == 1:
        return np.ones(1)
    initial = np.zeros(count)
    initial[int(np.argmin(np.diag(cov)))] = 1
    return _quadratic_minimum(cov, np.ones(count), [1], initial)


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
    best_vertex = int(np.argmax(excess / np.sqrt(np.diag(cov))))
    initial[best_vertex] = 1 / normalized_excess[best_vertex]
    solution = _quadratic_minimum(
        cov, normalized_excess, [1], initial, simplex=False
    )
    # Scale before summing to avoid overflow for an otherwise finite solution.
    largest = float(np.max(solution))
    if largest <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    solution = solution / largest
    return _weights(solution / np.sum(solution), count)


def _short_vertex(count, long_asset, short_asset, gross):
    weights = np.zeros(count)
    weights[long_asset] = gross / 2 + 0.5
    weights[short_asset] -= gross / 2 - 0.5
    return weights


def _short_lift(weights, gross, padding_asset):
    """Represent signed positions with fixed long/short budgets and overlap."""
    positive = np.maximum(weights, 0)
    negative = np.maximum(-weights, 0)
    padding = (gross / 2 - 0.5) * np.sum(weights) - np.sum(negative)
    if not np.isfinite(padding) or padding < -_FEASIBILITY_TOL:
        raise PortfolioError(_SOLVER_ERROR)
    positive[padding_asset] += max(0.0, padding)
    negative[padding_asset] += max(0.0, padding)
    return np.r_[positive, negative]


def _short_quadratic_minimum(cov, equality, targets, initial, gross, *, homogeneous=False):
    """Convex signed QP lifted to nonnegative long/short legs.

    For net-one portfolios, sum(long)=A and sum(short)=B, where A-B=1
    and A+B=gross. Overlapping legs represent unused gross capacity.
    The homogeneous Sharpe QP instead imposes B*sum(long)=A*sum(short).
    Only the original covariance is SPD; the lifted objective is PSD.
    """
    equality = np.atleast_2d(np.asarray(equality, dtype=float))
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    initial = np.asarray(initial, dtype=float)
    count = len(initial)
    deviations = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(deviations, deviations)
    try:
        exact, _ = _unconstrained_minimum(correlation, equality / deviations, targets)
    except np.linalg.LinAlgError as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    exact = exact / deviations
    exact_net = float(np.sum(exact))
    if (
        np.isfinite(exact).all()
        and exact_net > 0
        and np.sum(np.abs(exact)) <= gross * exact_net
        and np.max(np.abs(equality @ exact - targets)) <= _FEASIBILITY_TOL
    ):
        initial = exact
    initial_units = deviations * initial
    risk_scale = float(np.sqrt(initial_units @ correlation @ initial_units))
    if not np.isfinite(risk_scale) or risk_scale <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    transform = risk_scale / deviations
    repeated_transform = np.r_[transform, transform]
    if not np.isfinite(repeated_transform).all() or np.min(repeated_transform) <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    long_budget, short_budget = gross / 2 + 0.5, gross / 2 - 0.5
    if homogeneous:
        lifted_equality = np.vstack((
            np.c_[equality, -equality],
            np.r_[np.full(count, short_budget), np.full(count, -long_budget)],
        ))
        lifted_targets = np.r_[targets, 0.0]
    else:
        lifted_equality = np.vstack((
            np.r_[np.ones(count), np.zeros(count)],
            np.r_[np.zeros(count), np.ones(count)],
            np.c_[equality[1:], -equality[1:]],
        ))
        lifted_targets = np.r_[long_budget, short_budget, targets[1:]]
    transformed_equality = lifted_equality * repeated_transform
    row_scales = np.linalg.norm(transformed_equality, axis=1)
    if not np.isfinite(row_scales).all() or np.min(row_scales) <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    transformed_equality = transformed_equality / row_scales[:, None]
    transformed_targets = lifted_targets / row_scales
    lifted_initial = _short_lift(initial, gross, int(np.argmin(deviations))) / repeated_transform
    if not np.isfinite(lifted_initial).all() or not np.isfinite(transformed_targets).all():
        raise PortfolioError(_SOLVER_ERROR)

    def objective(z):
        positions = z[:count] - z[count:]
        return 0.5 * float(positions @ correlation @ positions)

    def gradient(z):
        partial = correlation @ (z[:count] - z[count:])
        return np.r_[partial, -partial]

    try:
        result = minimize(
            objective, lifted_initial, jac=gradient, method="SLSQP",
            bounds=[(0, None)] * (2 * count),
            constraints={
                "type": "eq",
                "fun": lambda z: transformed_equality @ z - transformed_targets,
                "jac": lambda z: transformed_equality,
            },
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    if not getattr(result, "success", False):
        raise PortfolioError(_SOLVER_ERROR)
    try:
        z = np.asarray(result.x, dtype=float)
        reported_objective = float(result.fun)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise PortfolioError(_SOLVER_ERROR) from exc
    if (
        z.shape != (2 * count,) or not np.isfinite(z).all()
        or not np.isfinite(reported_objective) or np.min(z) < -_FEASIBILITY_TOL
    ):
        raise PortfolioError(_SOLVER_ERROR)
    original_legs = z * repeated_transform
    if not np.isfinite(original_legs).all() or np.min(original_legs) < -_FEASIBILITY_TOL:
        raise PortfolioError(_SOLVER_ERROR)
    z = np.maximum(z, 0)
    legs = z * repeated_transform
    solution = legs[:count] - legs[count:]
    net = float(np.sum(solution))
    if (
        not np.isfinite(solution).all() or net <= 0
        or np.max(np.abs(equality @ solution - targets)) > _FEASIBILITY_TOL
        or np.max(np.abs(lifted_equality @ legs - lifted_targets)) > _FEASIBILITY_TOL
        or np.max(np.abs(transformed_equality @ z - transformed_targets)) > _FEASIBILITY_TOL
        or np.sum(np.abs(solution)) > gross * net + _FEASIBILITY_TOL
    ):
        raise PortfolioError(_SOLVER_ERROR)
    _certify_nonnegative_qp(z, transformed_equality, transformed_targets, gradient(z), 2 * objective(z))
    return solution


def _maximum_short_sharpe(cov, excess, gross, minimum):
    count = len(excess)
    best_return_vertex = _short_vertex(count, int(np.argmax(excess)), int(np.argmin(excess)), gross)
    maximum_excess = float(best_return_vertex @ excess)
    if maximum_excess <= 0:
        if np.all(excess == 0):
            return minimum.copy(), True
        # Every feasible point is a convex combination of the A*e_i-B*e_j
        # vertices. If all have nonpositive excess, the triangle inequality
        # for the covariance norm bounds its Sharpe by the best vertex's.
        # Individual negative excess alone is NOT enough for this branch.
        long_budget, short_budget = gross / 2 + 0.5, gross / 2 - 0.5
        variances = np.diag(cov)
        vertex_variance = (
            long_budget**2 * variances[:, None] + short_budget**2 * variances[None, :]
            - 2 * long_budget * short_budget * cov
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scores = (long_budget * excess[:, None] - short_budget * excess[None, :]) / np.sqrt(vertex_variance)
        np.fill_diagonal(scores, -np.inf)
        long_asset, short_asset = np.unravel_index(np.argmax(scores), scores.shape)
        if not np.isfinite(scores[long_asset, short_asset]):
            raise PortfolioError(_SOLVER_ERROR)
        return _short_vertex(count, long_asset, short_asset, gross), True
    normalized_excess = excess / np.max(np.abs(excess))
    initial_excess = float(best_return_vertex @ normalized_excess)
    if not np.isfinite(normalized_excess).all() or initial_excess <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    solution = _short_quadratic_minimum(
        cov, normalized_excess, [1], best_return_vertex / initial_excess, gross,
        homogeneous=True,
    )
    # The cone plus excess.y=1 implies sum(y)>0. Normalize without clipping.
    normalized = solution / np.max(np.abs(solution))
    net = float(np.sum(normalized))
    if not np.isfinite(net) or net <= 0:
        raise PortfolioError(_SOLVER_ERROR)
    return _weights(normalized / net, count, allow_short=True, gross_limit=gross), False


def _optimize_short(means, cov, scale, excess, points, gross):
    count = len(means)
    initial = np.zeros(count)
    initial[int(np.argmin(np.diag(cov)))] = 1
    min_weights = _weights(
        _short_quadratic_minimum(cov, np.ones(count), [1], initial, gross),
        count, allow_short=True, gross_limit=gross,
    )
    min_return, min_risk, _ = _metrics(min_weights, means, cov, scale, excess)
    weights, nonpositive = _maximum_short_sharpe(cov, excess, gross, min_weights)
    weights = _weights(weights, count, allow_short=True, gross_limit=gross)
    max_return, max_risk, max_sharpe = _metrics(weights, means, cov, scale, excess)
    if np.all(means == means[0]):
        frontier_weights = np.array([min_weights])
    else:
        relative_returns = (means - np.min(means)) / (np.max(means) - np.min(means))
        if not np.isfinite(relative_returns).all():
            raise PortfolioError("The return range cannot be optimized reliably.")
        maximum_target = gross / 2 + 0.5
        equality = np.vstack((np.ones(count), relative_returns))
        # This QP optimizes the entire maximum-return face, including all
        # long/short cross-covariances when either extreme has tied assets.
        endpoint = _short_quadratic_minimum(
            cov, equality, [1, maximum_target],
            _short_vertex(count, int(np.argmax(means)), int(np.argmin(means)), gross), gross,
        )
        start = float(relative_returns @ min_weights)
        if start >= maximum_target - np.finfo(float).eps * maximum_target:
            frontier_weights = np.array([min_weights])
        else:
            targets = np.linspace(start, maximum_target, points)
            solutions = [min_weights]
            for fraction, target in zip(np.linspace(0, 1, points)[1:-1], targets[1:-1]):
                initial = (1 - fraction) * min_weights + fraction * endpoint
                solutions.append(_short_quadratic_minimum(cov, equality, [1, target], initial, gross))
            solutions.append(endpoint)
            frontier_weights = np.array([
                _weights(w, count, allow_short=True, gross_limit=gross) for w in solutions
            ])
    frontier_metrics = [_metrics(w, means, cov, scale, excess) for w in frontier_weights]
    return PortfolioResult(
        weights, max_return, max_risk, max_sharpe, min_weights, min_return, min_risk,
        np.array([row[0] for row in frontier_metrics]),
        np.array([row[1] for row in frontier_metrics]), frontier_weights, nonpositive,
        allow_short=True, gross_limit=gross,
    )


def optimize_portfolio(mean_returns, covariance, risk_free_rate, points=50, *, allow_short=False, gross_limit=2.0):
    """Optimize all portfolios under one net-one, shared exposure policy.

    Rejects singular/indefinite covariance rather than changing the model.
    Equal means produce the single minimum-variance efficient portfolio.
    Short selling adds an explicit L1 gross cap; its default is 200%.
    """
    means, cov, scale, _, excess = _inputs(mean_returns, covariance, risk_free_rate)
    gross = _policy(allow_short, gross_limit)
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or points < 2:
        raise PortfolioError("The frontier needs at least two requested points.")
    count = len(means)
    if allow_short and gross > 1 and count > 1:
        try:
            return _optimize_short(means, cov, scale, excess, points, gross)
        except (OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise PortfolioError(_SOLVER_ERROR) from exc
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
        allow_short=bool(allow_short),
        gross_limit=gross,
    )
