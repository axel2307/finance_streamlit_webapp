"""Streamlit portfolio workspace with explicit analysis and persistent results."""
import base64
from datetime import datetime, timezone
from ftplib import FTP, all_errors as ftp_errors
from html import escape
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from finance_data import DataError, extract_close_prices, parse_catalog, prepare_returns
import portfolio_modes_v1 as portfolio
from risk_free_irx_v1 import read_irx_quote
from annual_moments_v1 import FREQUENCY_LABELS, PERIODS_PER_YEAR, annualize_returns
from portfolio_charts_v2 import (
    plot_stock_prices, plot_normalized_prices, plot_correlation_matrix,
    plot_returns_kde_per_asset, plot_efficient_frontier_and_max_sharpe,
    OptimizationViewError, optimization_comparison, capital_allocation_segment,
)

SNAPSHOT_KEY = 'analysis_snapshot_v1'
REFRESH_KEY = 'refresh_analysis_v1'


@st.cache_data(ttl=3600)
def load_data():
    """Read a complete catalog in memory; failed transfers are never cached."""
    files = {}
    with FTP('ftp.nasdaqtrader.com', timeout=30) as ftp:
        ftp.login()
        ftp.cwd('symboldirectory')
        for name in ('nasdaqlisted.txt', 'otherlisted.txt'):
            buffer = BytesIO()
            ftp.retrbinary(f'RETR {name}', buffer.write)
            files[name] = buffer.getvalue().decode('utf-8')
    return parse_catalog(files['nasdaqlisted.txt'], files['otherlisted.txt'])


def download_data(label, **kwargs):
    try:
        result = yf.download(**kwargs)
    except Exception as exc:
        raise DataError(f'{label} could not be downloaded. Please try again later.') from exc
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise DataError(f'{label} is unavailable for this selection. Try another period or try again later.')
    return result


@st.cache_data(ttl=3600)
def load_risk_free_proxy():
    """Cache only a validated latest quote; failed downloads are never cached."""
    quotes = download_data('The Treasury bill proxy (^IRX)', tickers='^IRX', period='5d', interval='1d')
    return read_irx_quote(quotes)


def filedownload(df, text='Download CSV File', filename='Download.csv'):
    """A static download does not rerun the app or consult draft controls."""
    encoded = base64.b64encode(df.to_csv(index=True).encode()).decode()
    return (f'<a href="data:text/csv;base64,{encoded}" download="{escape(filename, quote=True)}">'
            f'{escape(text)}</a>')


def analysis_signature(params):
    return (tuple(params['tickers']), params['period'], params['interval'],
            params['allow_short'], params['gross_limit'])


def analyze(params, selected_assets):
    """Publish only artifacts from this attempt, including on controlled failure."""
    snapshot = {
        'signature': analysis_signature(params), 'params': params.copy(),
        'selected_assets': selected_assets.copy(deep=True), 'prices': None, 'price_export': None,
        'returns': None, 'annual': None, 'risk_free_quote': None, 'solution': None,
        'error': None, 'completed_at': None,
    }
    # Invalidate the previous attempt before external work can be interrupted.
    st.session_state[SNAPSHOT_KEY] = snapshot
    try:
        raw = download_data('Price data', tickers=list(params['tickers']),
                            period=params['period'], interval=params['interval'], auto_adjust=True)
        snapshot['prices'] = extract_close_prices(raw, params['tickers'])
        # Preserve the existing single-asset OHLC/volume export after validating
        # its closes and dates; multi-asset exports retain the close-price table.
        snapshot['price_export'] = (raw.reset_index(level=0).copy(deep=True)
                                    if len(params['tickers']) == 1 else snapshot['prices'])
        snapshot['returns'] = prepare_returns(snapshot['prices'])
        if len(params['tickers']) > 1:
            snapshot['annual'] = annualize_returns(snapshot['returns'], params['interval'])
            snapshot['risk_free_quote'] = load_risk_free_proxy()
            annual = snapshot['annual']
            snapshot['solution'] = portfolio.optimize_portfolio(
                annual.mean_returns, annual.covariance,
                snapshot['risk_free_quote'].annual_simple_yield,
                allow_short=params['allow_short'], gross_limit=params['gross_limit'])
    except (DataError, portfolio.PortfolioError) as exc:
        snapshot['error'] = str(exc)
    snapshot['completed_at'] = datetime.now(timezone.utc)
    st.session_state[SNAPSHOT_KEY] = snapshot


def request_refresh():
    snapshot = st.session_state.get(SNAPSHOT_KEY)
    if snapshot is not None:
        # Capture analyzed settings, even if the sidebar contains a new draft.
        st.session_state[REFRESH_KEY] = (
            snapshot['params'].copy(), snapshot['selected_assets'].copy(deep=True))


def result_weights(snapshot):
    return pd.DataFrame({
        'Symbol': snapshot['returns'].columns,
        'Opt Weight': snapshot['solution'].weights,
        'Annual Mean Return': snapshot['annual'].mean_returns,
    })


def render_overview(snapshot):
    solution = snapshot['solution']
    if solution is None:
        if len(snapshot['params']['tickers']) == 1:
            if snapshot['prices'] is not None and snapshot['returns'] is not None:
                st.info('One asset selected: prices and returns are available in Data. '
                        'Select at least two assets to optimize a portfolio.')
            else:
                st.info('One asset selected. Any valid downloads from this attempt are in Data. '
                        'Select at least two assets to optimize a portfolio.')
        else:
            st.info('Portfolio analysis is unavailable for this attempt. '
                    'Any valid price and return downloads are available in Data.')
        return
    cards = st.columns(3)
    cards[0].metric('Annual return', f'{solution.max_return:.2%}',
                    help='Estimated arithmetic return of the maximum-Sharpe portfolio.')
    cards[1].metric('Annual volatility', f'{solution.max_risk:.2%}',
                    help='Estimated standard deviation using annual sample covariance.')
    cards[2].metric('Sharpe ratio', f'{solution.max_sharpe:.3f}',
                    help='Annual excess return divided by annual volatility; dimensionless.')
    st.markdown('### Optimal composition')
    result = result_weights(snapshot)
    result_mode = 'Short allowed' if solution.allow_short else 'Long-only'
    if solution.allow_short:
        st.write(f'Short allowed: weights sum to 1 and gross exposure is at most {solution.gross_limit:.0%}.')
    else:
        st.write('Long-only: weights are between 0 and 1 and sum to 1.')
    st.dataframe(result.style.format({'Opt Weight': '{:.2%}', 'Annual Mean Return': '{:.2%}'}),
                 hide_index=True, width='stretch')
    with st.expander('Portfolio Metrics and exposures'):
        st.table(pd.DataFrame([{
            'Portfolio': f'{result_mode} Max Sharpe Ratio',
            'Annual Return': solution.max_return, 'Annual Volatility': solution.max_risk,
            'Sharpe Ratio': solution.max_sharpe,
            'Net Exposure (%)': float(np.sum(solution.weights)) * 100,
            'Gross Exposure (%)': float(np.sum(np.abs(solution.weights))) * 100,
        }]).style.format({
            'Annual Return': '{:.2%}', 'Annual Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.3f}', 'Net Exposure (%)': '{:.2f}%', 'Gross Exposure (%)': '{:.2f}%'}))
    annual = snapshot['annual']
    interval = snapshot['params']['interval']
    st.caption(f'Annual estimates use {annual.periods_per_year} periods/year ({interval}): '
               'arithmetic mean × N, covariance × N, volatility × √N. '
               'In-sample estimates, not forecasts or realized gains. CSV returns remain per observation.')
    quote = snapshot['risk_free_quote']
    st.caption(f'Yahoo Finance ^IRX · as of {quote.as_of:%Y-%m-%d} · '
               '13-week Treasury bill proxy · assumed 91 days (remaining maturity unknown). '
               f'Bank discount quote: {quote.discount_quote_percent:.3f}%.')
    st.write(f'Risk-free proxy (annual simple investment yield): {quote.annual_simple_yield * 100:.3f}%')
    if solution.allow_short and solution.all_excess_returns_nonpositive and solution.max_sharpe < 0:
        st.caption('All feasible portfolios have nonpositive excess returns. Maximizing a negative '
                   'Sharpe ratio can favor higher risk, so this portfolio may lie outside the efficient frontier.')


def render_risk(snapshot):
    returns = snapshot['returns']
    if returns is None or snapshot['solution'] is None:
        st.info('Risk comparisons require a successful analysis of at least two assets.')
        return
    st.markdown('### Correlation Matrix')
    correlation = returns.corr().round(2)
    plot_correlation_matrix(correlation)
    with st.expander('Correlation values'):
        st.dataframe(correlation, width='stretch')
    st.markdown('### Distribution of Returns by Asset (KDE)')
    plot_returns_kde_per_asset(returns, snapshot['params']['interval'])
    st.caption('Distributions use returns per observation, at the analyzed frequency.')
    with st.expander('Annual Variance - Covariance Matrix'):
        st.dataframe(pd.DataFrame(snapshot['annual'].covariance, index=returns.columns,
                                  columns=returns.columns), width='stretch')


def render_optimization(snapshot):
    solution = snapshot['solution']
    if solution is None:
        st.info('Select at least two assets and complete an analysis to compare portfolios.')
        return
    st.markdown('### Efficient Frontier and Portfolio Optimization')
    annual = snapshot['annual']
    annual_rf = snapshot['risk_free_quote'].annual_simple_yield
    show_cloud = st.toggle('Show sample portfolios', value=True, key='optimization_show_cloud_v1',
                           help='Change only the chart. Analyzed settings, results and downloads stay the same.')
    try:
        comparison = optimization_comparison(solution, annual_rf)
        cal_segment = capital_allocation_segment(solution, annual_rf)
        plot_efficient_frontier_and_max_sharpe(
            annual.mean_returns, annual.covariance, annual_rf, solution, show_cloud=show_cloud)
    except OptimizationViewError as exc:
        st.warning(str(exc))
        return
    st.caption('The square marks the risk-free proxy; the star marks maximum Sharpe; '
               'the diamond marks global minimum variance (GMV). '
               'The frontier and the comparison use the analyzed exposure constraints.')
    st.dataframe(comparison.style.format({
        'Annual return': '{:.2%}', 'Annual volatility': '{:.2%}', 'Sharpe ratio': '{:.3f}'}),
        hide_index=True, width='stretch')
    if cal_segment is None:
        st.warning('The maximum Sharpe ratio is nonpositive. In this model, the risk-free proxy '
                   'offers at least as much expected return with zero modeled volatility. '
                   'The CAL is omitted; the risky portfolio is not a better cash alternative.')
    else:
        st.caption('CAL — Maximum Sharpe shows theoretical combinations of the risk-free proxy '
                   'and this risky portfolio, from 0% to 100% in the risky portfolio. '
                   'Its slope is the annual Sharpe ratio; the segment ends at the risky portfolio.')
    st.caption('This reference does not change portfolio weights or implement a cash allocation, '
               'borrowing at the proxy rate, or a leveraged extension. '
               'Active portfolio constraints do not imply a smooth tangency to the frontier.')


def render_data(snapshot):
    params = snapshot['params']
    st.markdown('### Analyzed assets and downloads')
    st.dataframe(snapshot['selected_assets'], hide_index=True, width='stretch')
    st.caption('Downloads belong to the analyzed settings shown above. '
               'Prices and returns retain their observation dates and frequency.')
    prices, returns = snapshot['prices'], snapshot['returns']
    filename = f"{'-'.join(params['tickers'])}_{params['period']}_{params['interval']}"
    if prices is not None:
        st.markdown(filedownload(snapshot['price_export'], 'Download Time Series',
                                 f'{filename}_prices.csv'), unsafe_allow_html=True)
        if len(params['tickers']) == 1:
            st.caption('The single-asset Time Series CSV retains the provider fields (including available '
                       'adjusted OHLC and volume); the price table shows adjusted closes.')
    if returns is not None:
        st.markdown(filedownload(returns, f"Download Returns ({params['interval']})",
                                 f'{filename}_returns.csv'), unsafe_allow_html=True)
    if snapshot['solution'] is not None:
        weights = result_weights(snapshot)[['Symbol', 'Opt Weight']].set_index('Symbol')
        mode = 'short' if params['allow_short'] else 'long'
        st.markdown(filedownload(weights, 'Download Portfolio Weights',
                                 f'{filename}_{mode}_gross{params["gross_limit"]:g}_weights.csv'), unsafe_allow_html=True)
    st.button('Refresh analyzed data', on_click=request_refresh,
              help='Download fresh prices for the analyzed settings. Pending sidebar changes are not applied.',
              key='refresh_analyzed_data')
    st.caption('Analysis persists during this session. Refresh analyzed data to request new prices; '
               'the validated IRX quote may remain cached for up to one hour.')
    if prices is None:
        return
    with st.expander('Adjusted close prices'):
        st.dataframe(prices, width='stretch')
    if returns is not None:
        with st.expander('Returns per observation'):
            st.dataframe(returns, width='stretch')
    if snapshot['error'] is None:
        st.markdown('### Stock Price Timelines')
        plot_stock_prices(prices)
        st.markdown('### Normalized Stock Price Timelines')
        plot_normalized_prices(prices)


def render_snapshot(snapshot, draft_signature=None):
    if draft_signature is not None and draft_signature != snapshot['signature']:
        st.warning('Pending changes: the results and downloads below still use the analyzed settings. '
                   'Select Analyze portfolio to apply your changes.')
    params = snapshot['params']
    mode = 'Short allowed' if params['allow_short'] else 'Long-only'
    timestamp = snapshot['completed_at']
    time_label = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if timestamp else 'Attempt interrupted'
    st.caption(f"Analyzed settings: {', '.join(params['tickers'])} | Period: {params['period']} | "
               f"Frequency: {FREQUENCY_LABELS[params['interval']]} | Mode: {mode} | "
               f"Gross limit: {params['gross_limit']:.0%} | {time_label}")
    if snapshot['error']:
        st.error(snapshot['error'])
    elif timestamp is None:
        st.warning('The previous analysis was interrupted. Select Analyze portfolio to try again.')
    prices, returns = snapshot['prices'], snapshot['returns']
    if prices is not None and returns is not None and prices.isna().any().any():
        st.info(f'Partial price history: using {len(returns)} common return observations without filling missing prices.')
    overview, risk, optimization, data = st.tabs(['Overview', 'Risk', 'Optimization', 'Data'])
    with overview:
        render_overview(snapshot)
    with risk:
        render_risk(snapshot)
    with optimization:
        render_optimization(snapshot)
    with data:
        render_data(snapshot)


st.set_page_config(page_title='Portfolio Analytics', page_icon='📈', layout='wide')
st.title('Portfolio Analytics')
st.caption('Explore U.S. stocks and ETFs. Compare annual portfolio estimates and download market data.')

refresh = st.session_state.pop(REFRESH_KEY, None)
if refresh is not None:
    with st.spinner('Refreshing analyzed data…'):
        analyze(*refresh)

try:
    catalog = load_data()
except ftp_errors + (UnicodeError, DataError, pd.errors.ParserError):
    catalog = None
    st.error('The Nasdaq symbol directory is unavailable. Please try again later.')

draft_signature = None
if catalog is not None:
    st.sidebar.header('Build your portfolio')
    with st.sidebar.expander('Asset filters'):
        asset_class = st.radio('Asset Class', options=('All', 'Stock', 'ETF'), horizontal=True)
        preferred_exchanges = ['NASDAQ', 'NYSE', 'NYSE Arca', 'BATS', 'NYSE MKT', 'IEX']
        exchanges = catalog['Exchange'].dropna().unique().tolist()
        exchange_options = [name for name in preferred_exchanges if name in exchanges]
        exchange_options += sorted(set(exchanges) - set(preferred_exchanges))
        exchange = st.radio('Exchange', options=['All'] + exchange_options)
    filtered = catalog
    if asset_class != 'All':
        filtered = filtered[filtered['ETF'] == ('N' if asset_class == 'Stock' else 'Y')]
    if exchange != 'All':
        filtered = filtered[filtered['Exchange'] == exchange]
    search_by = st.sidebar.radio('Search by', options=('Symbol', 'Security Name'), horizontal=True)
    options = sorted(filtered[search_by].dropna().astype(str).unique())
    selection_key = 'selected_assets_v1'
    previous = st.session_state.get(selection_key, [])
    retained = [value for value in previous if value in options]
    if retained != previous:
        st.session_state[selection_key] = retained
        st.sidebar.info('Selection updated: assets outside the current filters were removed.')
    selected = st.sidebar.multiselect('Select assets', options, key=selection_key,
                                      placeholder='Search and select assets',
                                      help='Choose one asset for data or at least two for portfolio optimization.')
    with st.sidebar.expander(f'Available assets ({len(filtered):,})'):
        st.caption('This catalog follows your asset filters. Search and select assets above.')
        st.dataframe(filtered, hide_index=True, width='stretch', height=300)
    period = st.sidebar.selectbox('Period', options=('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'), index=5)
    with st.sidebar.expander('Advanced: return frequency'):
        interval = st.selectbox('Return frequency', options=tuple(PERIODS_PER_YEAR),
                                format_func=lambda value: FREQUENCY_LABELS[value], key='return_frequency')
        st.caption('Daily by default. Analyze portfolio applies the frequency and re-estimates from new bars.')
    mode = st.sidebar.radio('Portfolio mode', options=('Long-only', 'Short allowed'), key='portfolio_mode')
    allow_short = mode == 'Short allowed'
    gross_limit = 1.0
    if allow_short:
        gross_limit = st.sidebar.number_input('Gross exposure limit (%)', min_value=100,
                                              value=200, step=25, key='gross_limit_percent') / 100
        st.sidebar.caption('Negative weights are short positions. Net exposure stays at 100%. '
                           'Gross exposure is the sum of absolute weights. '
                           'For example, a 200% gross limit allows up to 150% long and 50% short exposure. '
                           'Borrowing costs are not included.')
    selected_assets = filtered[filtered[search_by].isin(selected)].copy()
    tickers = tuple(dict.fromkeys(selected_assets.Symbol.tolist()))
    params = {'tickers': tickers, 'period': period, 'interval': interval,
              'allow_short': allow_short, 'gross_limit': gross_limit}
    draft_signature = analysis_signature(params)
    submitted = st.sidebar.button('Analyze portfolio', type='primary', width='stretch', key='analyze_portfolio')
    st.sidebar.caption('Change settings, then analyze. Results remain tied to their analyzed settings.')
    if submitted:
        previous_snapshot = st.session_state.get(SNAPSHOT_KEY)
        if not tickers:
            st.session_state.pop(SNAPSHOT_KEY, None)
            st.warning('Select at least one asset, then choose Analyze portfolio.')
        elif (previous_snapshot is None or previous_snapshot['signature'] != draft_signature
              or previous_snapshot['error'] is not None or previous_snapshot['completed_at'] is None):
            with st.spinner('Analyzing your portfolio…'):
                analyze(params, selected_assets)
        else:
            st.info('These settings are already analyzed. Use Refresh analyzed data in Data to request new prices.')
else:
    st.sidebar.info('Asset selection is unavailable until the symbol directory recovers.')

snapshot = st.session_state.get(SNAPSHOT_KEY)
if snapshot is not None:
    render_snapshot(snapshot, draft_signature)
elif catalog is not None:
    st.info('Select assets in the sidebar, then choose Analyze portfolio. '
            'One asset provides data; two or more enable portfolio optimization.')

st.caption('Created by Axel Rodrigo for educational purposes. Contact: axel.23@hotmail.com.')
