"""Compact charts for validated market data and annual portfolio estimates.

Color is supplemented by line styles, marker shapes, outlines and text. These
source-level affordances support interpretation; they are not a WCAG audit of
the rendered app. The underlying numeric tables remain the text alternative.
"""

from textwrap import fill

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


_TEXT = "#172B4D"
_MUTED = "#475569"
_GRID = "#E2E8F0"
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#A66A00", "#3B748A", "#20252B")
_LINE_STYLES = ("-", "--", "-.", ":")

# A view-only limit for meaningful division/display, not a portfolio policy
# or a replacement for the optimizer's covariance validation.
_MIN_VIEW_VOLATILITY = 1e-12


class OptimizationViewError(ValueError):
    """Portfolio results cannot support a reliable comparison or chart."""


def _finite_value(value, name):
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise OptimizationViewError(f"The optimization view requires a finite real {name}.")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise OptimizationViewError(f"The optimization view requires a finite real {name}.") from exc
    if not np.isfinite(number):
        raise OptimizationViewError(f"The optimization view requires a finite real {name}.")
    return number


def _view_metrics(solution, risk_free_rate):
    risk_free = _finite_value(risk_free_rate, "annual risk-free proxy")
    names = ("max_return", "max_risk", "max_sharpe", "min_return", "min_risk")
    try:
        metrics = {name: _finite_value(getattr(solution, name), name) for name in names}
    except AttributeError as exc:
        raise OptimizationViewError("The optimization view requires complete portfolio results.") from exc
    if metrics["max_risk"] <= _MIN_VIEW_VOLATILITY or metrics["min_risk"] <= _MIN_VIEW_VOLATILITY:
        raise OptimizationViewError(
            "Annual volatility is too close to zero for a reliable optimization view "
            "(must be greater than 1e-12)."
        )
    max_excess = _finite_value(metrics["max_return"] - risk_free, "maximum-Sharpe excess return")
    min_excess = _finite_value(metrics["min_return"] - risk_free, "minimum-variance excess return")
    calculated_max_sharpe = _finite_value(max_excess / metrics["max_risk"], "maximum Sharpe ratio")
    min_sharpe = _finite_value(min_excess / metrics["min_risk"], "minimum-variance Sharpe ratio")
    if not np.isclose(metrics["max_sharpe"], calculated_max_sharpe, rtol=1e-7, atol=1e-9):
        raise OptimizationViewError("The reported Sharpe ratio does not match the annual return, volatility and risk-free proxy.")
    metrics.update(risk_free=risk_free, max_excess=max_excess, min_sharpe=min_sharpe)
    return metrics


def optimization_comparison(solution, rf):
    """Compare the two actual portfolios, keeping numeric annual values."""
    metrics = _view_metrics(solution, rf)
    return pd.DataFrame([
        {"Portfolio": "Maximum Sharpe", "Annual return": metrics["max_return"],
         "Annual volatility": metrics["max_risk"], "Sharpe ratio": metrics["max_sharpe"]},
        {"Portfolio": "Global minimum variance", "Annual return": metrics["min_return"],
         "Annual volatility": metrics["min_risk"], "Sharpe ratio": metrics["min_sharpe"]},
    ])


def capital_allocation_segment(solution, rf):
    """Return only the unlevered RF-to-portfolio segment for positive excess."""
    metrics = _view_metrics(solution, rf)
    if metrics["max_sharpe"] <= 0 or metrics["max_excess"] <= 0:
        return None
    return (
        np.array([0.0, metrics["max_risk"]]),
        np.array([metrics["risk_free"], metrics["max_return"]]),
    )


def display_figure(fig, *, alt=None):
    """Render at the available container width and always release the figure."""
    try:
        if alt is None:
            alt = "Financial chart."
            if fig.axes:
                axis = fig.axes[0]
                title = axis.get_title().replace("\n", ". ")
                alt = f"{title}." if title else alt
                if axis.get_xlabel():
                    alt += f" Horizontal axis: {axis.get_xlabel()}."
                if axis.get_ylabel():
                    alt += f" Vertical axis: {axis.get_ylabel()}."
        st.pyplot(fig, width="stretch", alt=alt)
    finally:
        plt.close(fig)


def _figure(title, xlabel, ylabel, *, height=4.8, wrap_title=True):
    fig, ax = plt.subplots(figsize=(6.4, height), layout="constrained")
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(fill(title, 40) if wrap_title else title, fontsize=14,
                 fontweight="bold", color=_TEXT, pad=14)
    ax.set_xlabel(xlabel, fontsize=12, color=_TEXT, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=12, color=_TEXT, labelpad=9)
    ax.tick_params(labelsize=11, colors=_MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#94A3B8")
    ax.set_axisbelow(True)
    ax.grid(visible=True, color=_GRID, linewidth=0.8)
    return fig, ax


def _legend_below(ax, *, columns=2, wrap_width=22, display_labels=None):
    # Wrap the displayed legend without changing the artists' stable labels,
    # which also identify their numeric evidence in the regression suite.
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if display_labels is not None:
        labels = [display_labels.get(label, label) for label in labels]
    ax.legend(
        handles, [fill(label, wrap_width, break_long_words=False) for label in labels],
        loc="upper center", bbox_to_anchor=(0.5, -0.21),
        ncol=min(columns, len(handles)), fontsize=10.5, labelcolor=_TEXT,
        frameon=False, borderaxespad=0, handlelength=2.1, columnspacing=1.3,
    )


def _dated_lines(ax, values):
    for position, column in enumerate(values.columns):
        ax.plot(
            values.index, values[column], label=str(column),
            color=_COLORS[position % len(_COLORS)],
            linestyle=_LINE_STYLES[position % len(_LINE_STYLES)], linewidth=2.1,
        )
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _legend_below(ax)


def plot_stock_prices(prices):
    fig, ax = _figure("Adjusted price history", "Date", "Adjusted Close Price (USD)")
    _dated_lines(ax, prices)
    display_figure(fig)


def plot_normalized_prices(prices):
    # Each series starts at its first observed value. Missing values stay missing.
    baseline = prices.apply(lambda column: column.dropna().iloc[0])
    fig, ax = _figure("Price performance (rebased to 1)", "Date", "Price / first observation")
    _dated_lines(ax, prices / baseline)
    display_figure(fig)


def plot_correlation_matrix(correlation):
    count = len(correlation.columns)
    height = max(4.8, min(8.8, 0.45 * count + 1.5))
    fig, ax = _figure("Asset correlations", "", "", height=height)
    ax.grid(False)
    sns.heatmap(
        correlation, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0,
        annot_kws={"fontsize": max(7, min(11, 75 / max(6, count)))},
        linewidths=1, linecolor="white", square=True, ax=ax,
        cbar_kws={"shrink": 0.8, "aspect": 25, "label": "Correlation", "ticks": [-1, 0, 1]},
    )
    ax.tick_params(axis="x", labelsize=11, rotation=30)
    ax.tick_params(axis="y", labelsize=11, rotation=0)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    # Choose annotation contrast from the actual cell fill, including pale
    # correlations for which a white annotation would become hard to read.
    cells = ax.collections[0]
    for annotation in ax.texts:
        column, row = (int(position) for position in annotation.get_position())
        rgb = np.asarray(cells.to_rgba(correlation.iloc[row, column])[:3])
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luminance = float(linear @ [0.2126, 0.7152, 0.0722])
        annotation.set_color("black" if luminance > 0.179 else "white")
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=10, colors=_MUTED)
    colorbar.set_label("Correlation", fontsize=11, color=_TEXT)
    display_figure(fig)


def plot_returns_kde_per_asset(returns, interval):
    fig, ax = _figure("Return distributions", f"Return ({interval})", "Density")
    has_density = False
    constant_values = []
    for position, column in enumerate(returns.columns):
        values = returns[column].dropna()
        if values.empty:
            continue
        color = _COLORS[position % len(_COLORS)]
        linestyle = _LINE_STYLES[position % len(_LINE_STYLES)]
        if values.nunique() == 1:
            # A constant observation has no KDE density; represent its actual
            # location explicitly instead of requesting a singular estimate.
            constant = float(values.iloc[0])
            constant_values.append(constant)
            ax.axvline(constant, color=color, linestyle=linestyle,
                       linewidth=2.1, label=f"{column} (constant)")
        else:
            has_density = True
            sns.kdeplot(data=values, label=str(column), fill=False, linewidth=2.1,
                        color=color, linestyle=linestyle, ax=ax)
    if not has_density:
        ax.set_ylabel("Constant-return marker", fontsize=12, color=_TEXT, labelpad=9)
        ax.set_yticks([])
        if constant_values:
            low, high = min(constant_values), max(constant_values)
            padding = max((high - low) * 0.15, max(abs(low), abs(high)) * 0.05, 0.0001)
            ax.set_xlim(low - padding, high + padding)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    _legend_below(ax)
    display_figure(fig)


def plot_efficient_frontier_and_max_sharpe(mean_returns, covariance, risk_free_rate, solution, *, show_cloud=True):
    """Compare annual portfolios and optionally show the repeatable cloud."""
    metrics = _view_metrics(solution, risk_free_rate)
    segment = capital_allocation_segment(solution, risk_free_rate)
    if not isinstance(show_cloud, (bool, np.bool_)):
        raise OptimizationViewError("Choose whether to show the sampled portfolios.")
    mode_suffix = " (Short allowed)" if solution.allow_short else ""
    if show_cloud:
        rng = np.random.default_rng(2026)
        random_weights = rng.dirichlet(np.ones(len(mean_returns)), 1000)
        if solution.allow_short:
            short_weights = rng.dirichlet(np.ones(len(mean_returns)), 1000)
            short_exposure = rng.uniform(0, (solution.gross_limit - 1) / 2, size=(1000, 1))
            # Unchanged formula: net exposure is one; overlap may reduce the
            # realized gross exposure below the available 1 + 2*s budget.
            random_weights = (1 + short_exposure) * random_weights - short_exposure * short_weights
        with np.errstate(over="ignore", invalid="ignore"):
            random_returns = random_weights @ mean_returns
            random_risks = np.sqrt(np.einsum("ij,jk,ik->i", random_weights, covariance, random_weights))
        if (not np.isfinite(random_returns).all() or not np.isfinite(random_risks).all()
                or (random_risks <= _MIN_VIEW_VOLATILITY).any()):
            raise OptimizationViewError("Sample portfolios require finite returns and positive annual volatility for this view.")
        with np.errstate(over="ignore", invalid="ignore"):
            random_sharpes = (random_returns - metrics["risk_free"]) / random_risks
        if not np.isfinite(random_sharpes).all():
            raise OptimizationViewError("Sample portfolio Sharpe ratios cannot be displayed reliably.")
    mode = "Short allowed" if solution.allow_short else "Long-only"
    subtitle = f"Annual · {mode}" + (" · samples hidden" if not show_cloud else "")
    fig, ax = _figure(
        f"Efficient Frontier and Random Portfolios\n{subtitle}",
        "Annual Volatility", "Annual Mean Return", height=6.1, wrap_title=False,
    )
    scatter = None
    if show_cloud:
        scatter = ax.scatter(
            random_risks, random_returns, c=random_sharpes, cmap="viridis", marker=".",
            s=15, alpha=0.18, linewidths=0, label=f"Random Portfolios{mode_suffix}", zorder=2,
        )
    ax.plot(
        solution.frontier_risks, solution.frontier_returns, color=_COLORS[0], linewidth=2.4,
        marker="." if len(solution.frontier_returns) == 1 else None,
        label=f"Efficient Frontier{mode_suffix}", zorder=3,
    )
    if segment is not None:
        ax.plot(*segment, color="#6D28D9", linestyle="--", linewidth=2,
                label="CAL — Maximum Sharpe", zorder=4)
    ax.scatter(
        0.0, metrics["risk_free"], marker="s", facecolor="#6D28D9",
        edgecolor="#111827", linewidths=1, s=65,
        label="Risk-free proxy", zorder=6, clip_on=False,
    )
    ax.annotate(
        f"RF {metrics['risk_free']:.2%}", xy=(0.0, metrics["risk_free"]),
        xytext=(9, 9), textcoords="offset points", ha="left", va="bottom",
        fontsize=10.5, color=_TEXT,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.95},
        zorder=7,
    )
    ax.scatter(
        solution.min_risk, solution.min_return, marker="D", facecolor="white",
        edgecolor=_COLORS[0], linewidths=1.8, s=75,
        label=f"Minimum Variance Portfolio{mode_suffix}", zorder=5,
    )
    ax.scatter(
        solution.max_risk, solution.max_return, marker="*", facecolor=_COLORS[1],
        edgecolor="#111827", linewidths=1.1, s=245,
        label=f"Max Sharpe Ratio Portfolio{mode_suffix}", zorder=6,
    )
    plotted_risk_max = max(float(np.max(solution.frontier_risks)), solution.max_risk)
    if show_cloud:
        plotted_risk_max = max(plotted_risk_max, float(np.max(random_risks)))
    label_near_right_edge = solution.max_risk >= 0.8 * plotted_risk_max
    ax.annotate(
        f"Max Sharpe {solution.max_sharpe:.2f}",
        xy=(solution.max_risk, solution.max_return), xytext=(-10 if label_near_right_edge else 0, 19),
        textcoords="offset points", ha="right" if label_near_right_edge else "center",
        va="bottom", fontsize=10.5,
        color=_TEXT, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.95},
        arrowprops={"arrowstyle": "-", "color": _MUTED, "linewidth": 0.8}, zorder=7,
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(x=0.09, y=0.18)
    ax.set_xlim(left=0)
    _legend_below(ax, display_labels={
        f"Random Portfolios{mode_suffix}": "Sample portfolios",
        f"Efficient Frontier{mode_suffix}": "Efficient frontier",
        f"Minimum Variance Portfolio{mode_suffix}": "Global minimum variance",
        f"Max Sharpe Ratio Portfolio{mode_suffix}": "Max Sharpe",
    })
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.035, shrink=0.8)
        colorbar.set_label("Sharpe Ratio", fontsize=11, color=_TEXT)
        colorbar.ax.tick_params(labelsize=10, colors=_MUTED)
        colorbar.locator = MaxNLocator(nbins=4)
        colorbar.update_ticks()
    allocation_description = (
        "The dashed CAL segment joins the risk-free proxy to the maximum Sharpe portfolio, without extension beyond it."
        if segment is not None else
        "The CAL segment is omitted because the maximum Sharpe ratio or excess return is nonpositive."
    )
    cloud_description = "Dots are 1,000 sampled feasible portfolios." if show_cloud else "Sample portfolios are hidden."
    display_figure(fig, alt=(
        f"Annual portfolio chart, {mode}. The star marks the maximum Sharpe portfolio: "
        f"annual mean return {solution.max_return:.2%}, annual volatility {solution.max_risk:.2%}, "
        f"Sharpe ratio {solution.max_sharpe:.3f}. The diamond marks global minimum variance. "
        f"The square marks the annual risk-free proxy at zero volatility and return {metrics['risk_free']:.2%}. "
        f"The solid line is the efficient frontier. {allocation_description} {cloud_description}"
    ))
