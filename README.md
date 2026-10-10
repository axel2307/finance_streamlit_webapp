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

The analysis uses one **long-only** portfolio model: every weight is between 0
and 1 and weights sum to 1. The weights table, metrics and maximum-Sharpe marker
share the same validated solution. The previous unrestricted short-selling table
has been removed because it used a different model from the chart and its
normalized formula could return a portfolio that did not maximize Sharpe.

- Returns retain every valid observation and its date. Missing prices are never
  forward-filled; statistics use only returns observed for all selected assets.
  Partial histories are accepted when the common sample supports the analysis.
- Missing assets, invalid prices, insufficient common returns, and singular or
  numerically unusable covariance matrices produce an explanatory error instead
  of NaN results or an unhandled inverse-matrix exception. At least two common
  returns are necessary, but are not sufficient for an invertible covariance.
- Solver results must satisfy convergence, finiteness and portfolio constraints.
  Equal expected returns produce a single minimum-variance frontier point.
  All nonpositive excess returns are handled explicitly rather than applying the
  positive-excess tangency formula with an invalid normalization.
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
