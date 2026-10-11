"""Compatibility facade for the validated portfolio API.

The Streamlit entry point imports the versioned implementation directly so that
an earlier cached ``portfolio`` module cannot supply the previous API.
"""

from portfolio_modes_v1 import (
    PortfolioError,
    PortfolioResult,
    optimize_portfolio,
    portfolio_metrics,
)

__all__ = [
    "PortfolioError",
    "PortfolioResult",
    "optimize_portfolio",
    "portfolio_metrics",
]
