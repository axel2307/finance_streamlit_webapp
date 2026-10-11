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

The pandas 3 startup fix preserves the valid ticker `NA` when reading CSV and
removes missing labels before sorting. The Treasury-yield calculation extracts a
scalar explicitly, which is required for pandas 3 and yfinance's MultiIndex data.
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

This repair does not change the existing current-`^TNX` proxy, days/365 rate
conversion, or per-interval return/volatility/Sharpe convention. Historical-rate
alignment, annualization and visual redesign remain separate product decisions.
The app is educational and its estimates do not guarantee future performance.

The regression suite covers the twelve reviewed defects: return retention and
CSV dates (BUG01/02), usable data and covariance (BUG03/04), optimality and a shared
long-only model (BUG05/06), solver failures and equal-mean frontiers (BUG07/08),
provider errors (BUG09/10), KDE semantics (BUG11) and exchange identity (BUG12).

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
