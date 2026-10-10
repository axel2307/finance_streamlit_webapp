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
