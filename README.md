# finance_streamlit_webapp
This is a Streamlit web application created by Axel Rodrigo for educational purposes.

## Run locally

Tested with Python 3.12; dependencies also resolve for Python 3.11, as used by the
dev container. Create an isolated virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run finance_streamlit_webapp.py
```

The entry point is `finance_streamlit_webapp.py`. The app reads the public Nasdaq
symbol directory using anonymous FTP and market prices using yfinance. These
sources require network access; no API keys or Streamlit secrets are needed.
The symbol directory is cached for one hour.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests replace Nasdaq and Yahoo responses with deterministic fixtures. They do not
require network access or credentials.

## Streamlit Community Cloud

Use repository `axel2307/finance_streamlit_webapp` and main file
`finance_streamlit_webapp.py`, with Python 3.11 or 3.12 and the branch containing the fix.
After reviewing and publishing the fix to the deployed branch, let Community Cloud
rebuild from `requirements.txt`; reboot the app from its management panel if needed.
Check that the default All / All / Symbol filters load, that `NA` is selectable,
and that selecting two assets and pressing Go reaches Portfolio Metrics.

The app imports `portfolio_modes_v1`, a distinct module for the long/short API.
This prevents a running Cloud process from reusing the earlier long-only
`portfolio` import when it receives the updated entry point. The implementation
uses Python's normal import locking; it does not reload or replace shared modules.
`portfolio.py` remains a compatibility facade for the four public API symbols.
Future incompatible optimizer interfaces should receive a new module identity.

The pandas 3 startup fix preserves the valid ticker `NA` when reading CSV and
removes missing labels before sorting. The Treasury-bill proxy extracts a dated
scalar explicitly from Yahoo's plain or MultiIndex data.
Direct dependencies are pinned to the versions used for this verification.

## Validated portfolio behavior

The sidebar defaults to **Long-only**: every weight is between 0 and 1 and weights
sum to 1. **Short allowed** keeps net exposure at 100% and adds a configurable
gross exposure limit, initially 200%. Gross exposure is the sum of absolute
weights; 200% permits up to 150% long and 50% short. The limit is a maximum, not
a target. Negative weights are short positions. Borrowing costs are not modeled.
The weights table, metrics, minimum-risk portfolio, efficient frontier and random
portfolio chart all use the selected exposure policy. The table reports realized
net and gross exposures. A 100% gross limit is equivalent to long-only.

- Returns retain every valid observation and its date. Missing prices are never
  forward-filled; statistics use only returns observed for all selected assets.
  Partial histories are accepted when the common sample supports the analysis.
- Missing assets, invalid prices, insufficient common returns, and singular or
  numerically unusable covariance matrices produce an explanatory error instead
  of NaN results or an unhandled inverse-matrix exception. At least two common
  returns are necessary, but are not sufficient for an invertible covariance.
- Solver results must satisfy convergence, finiteness and portfolio constraints.
  Per-asset risk scaling and a feasible-portfolio objective scale prevent small
  variances from being hidden by absolute stopping tolerances. Relative
  optimality checks also cover frontier points with different expected returns.
  Equal expected returns produce a single minimum-variance frontier point.
  Nonpositive *feasible portfolio* excess returns are handled explicitly; shorts
  can yield a positive excess return even when every asset is below the rate.
- Provider failures are handled at the FTP and yfinance boundaries. The cached
  catalog is assembled in memory only after both transfers complete.
- Exchange filters preserve Nasdaq's source market codes. NYSE is code `N`;
  NYSE Arca, BATS, NYSE MKT and IEX are separate options when present. Unknown
  codes retain an explicit `Unknown (code)` label.
- The KDE is labeled as a distribution of **returns**, and return downloads
  name the selected interval. Single-asset downloads and the literal ticker
  `NA` remain supported.

The current `^IRX` proxy follows the simple-yield conversion below. Portfolio
optimization, returns, volatility and Sharpe use annual units as described below.
Historical-rate alignment and visual redesign remain separate product decisions.
The app is educational and its estimates do not guarantee future performance.

The regression suite covers the twelve reviewed defects: return retention and
CSV dates (BUG01/02), usable data and covariance (BUG03/04), optimality and a shared
long-only model (BUG05/06), solver failures and equal-mean frontiers (BUG07/08),
provider errors (BUG09/10), KDE semantics (BUG11) and exchange identity (BUG12).

## Risk-free proxy

Yahoo Finance / yfinance is the only rate source. The app requests the last five
daily observations of `^IRX` and uses the most recent dated, nonmissing `Close`
**level**, expressed in percent; it never takes percentage changes of the index.
The source, quote date and assumptions are displayed beside the results.
Validated quotes are cached for up to one hour across selections and modes.
Missing, ambiguous or invalid data stops portfolio analysis with an explanation
while preserving the price and return CSVs; failures are not cached and there
is no fallback rate.

Let `d = quote / 100`. With an **assumed 91-day** bill maturity, the estimated
annual simple investment yield is `y = 365*d / (360 - 91*d)`. For example, a
4% bank-discount quote gives approximately 4.097% annual simple yield.
This follows discount pricing on face value with a 360-day basis, then measures
the gain against the purchase price on a 365-day basis
([TreasuryDirect pricing](https://www.treasurydirect.gov/marketable-securities/understanding-pricing/)).
The 91 days are a proxy assumption, not the bill's known remaining maturity;
this is not an exact replication of DGS3MO or a historical rate series.

`y` is **not an effective annual yield**. The optimizer now compares it directly
with annualized arithmetic asset returns; it is not divided by 365 to compare
against returns per trading observation. No compounding exponent is applied.
The same annual rate is used by the optimizer, metrics and charts in both modes.
Zero or negative observed quotes remain valid when the implied bill price is
positive and the conversion is finite; missing quotes are never replaced by zero.

## Annual portfolio estimates

Asset downloads explicitly use yfinance `auto_adjust=True`. Simple returns are
computed from adjusted closes without filling missing prices. The default is
daily data; the sidebar's advanced return-frequency selector also offers weekly,
monthly and quarterly data. Changing frequency downloads different bars and
re-estimates the sample, rather than changing the display of the same result.

For a frequency with `N` observations per year, the optimizer receives
`mean_annual = mean_period * N` and `covariance_annual = covariance_period * N`.
The covariance uses the sample estimator (`ddof=1`). The conventions are:

| Return interval | N per year |
| --- | ---: |
| Daily (`1d`) | 252 |
| Weekly (`1wk`) | 52 |
| Monthly (`1mo`) | 12 |
| Quarterly (`3mo`) | 4 |

The old `5d` **return interval** is replaced by `1wk`: this model has an explicit
52-week convention and does not assign an arbitrary annual factor to 5-day bars.
`5d` remains available as a historical lookback period and for the IRX request.

Portfolio volatility follows the square-root-of-time convention, `sigma * sqrt(N)`.
Annual Sharpe is `(annual arithmetic return - annual IRX proxy) / annual volatility`;
it receives no second annualization. This convention assumes stable moments and
does not account for serial correlation
([Sharpe, Time Dependence](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)).
Weights, per-asset means, portfolio metrics, frontier, optimal markers and random
portfolios share these annual inputs and exposure constraints. Return, volatility
and weights are formatted as percentages; Sharpe is dimensionless. Covariance
remains in annual squared-return units, and the KDE shows periodic simple returns.

These are **in-sample estimates of annual arithmetic returns**, not forecasts,
realized gains or compounded returns. No CAGR is shown because the app does not
define a rebalanced portfolio wealth path. The current IRX proxy is not a
historical rate series, and short borrowing costs remain outside this model.

Price and return CSVs preserve the selected-frequency observations and dates;
their rows are never annualized. The portfolio-weights CSV contains the actual
signed fractional weights, matching the displayed percentages. Scaling mean,
covariance and RF by the same positive factor preserves optimal weights; changing
from the earlier calendar-day RF convention to annual IRX can legitimately change
weights because it changes the relative expected excess returns.

## Short-selling optimization

For gross limit `G >= 1`, the domain is `sum(w)=1, sum(abs(w))<=G`. Let
`A=(G+1)/2` and `B=(G-1)/2`. Nonnegative long/short legs with totals `A` and `B`
represent this domain exactly: overlapping legs represent unused gross capacity.
The convex minimum-variance QPs use that lift. Only the original covariance must
be positive definite; the lifted objective is positive semidefinite by construction.
Risk scaling, original-domain feasibility and relative primal/dual certificates
are checked before returning results.

When a positive excess return is attainable, maximum Sharpe is equivalent to
minimizing `y' covariance y` with `excess' y=1` and
`sum(abs(y)) <= G*sum(y)`. The constraints imply `sum(y)>0`; normalize by that sum.
The nonnegative lift fixes the long/short total ratio rather than their magnitude.

If every feasible excess is nonpositive, compare the vertices
`A*unit(i)-B*unit(j)`, `i != j`. This is globally sufficient: each feasible
portfolio is a convex combination of vertices, and the covariance norm's triangle
inequality bounds its nonpositive Sharpe by the best vertex. Check the attainable
maximum `A*max(excess)-B*min(excess)`, not only the asset excesses.
With negative excess returns, the maximum-Sharpe portfolio can have higher risk
than the minimum-risk portfolio and can lie outside the efficient frontier.

The frontier extends to `A*max(mean)-B*min(mean)`. Ties at either extreme are
optimized jointly on the maximum-return face, retaining long/short cross-covariances.
Equal means yield one minimum-variance frontier point. A single asset retains
weight 1 in either mode. Numerically unreliable results produce a controlled
error; the model does not silently relax exposure limits or regularize covariance.
