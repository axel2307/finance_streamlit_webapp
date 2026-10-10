"""Streamlit interface for public market data and validated long-only portfolios."""
import base64
from ftplib import FTP, all_errors as ftp_errors
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf

from finance_data import DataError, extract_close_prices, parse_catalog, prepare_returns
import portfolio


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
    # Catch only at the external library boundary, not around our calculations.
    try:
        result = yf.download(**kwargs)
    except Exception as exc:
        raise DataError(f'{label} could not be downloaded. Please try again later.') from exc
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise DataError(f'{label} is unavailable for this selection. Try another period or try again later.')
    return result


def read_treasury_yield(data):
    if 'Close' not in data.columns:
        raise DataError('The Treasury yield (^TNX) is unavailable. Please try again later.')
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        if '^TNX' not in close.columns:
            raise DataError('The Treasury yield (^TNX) is unavailable. Please try again later.')
        close = close['^TNX']
    valid = pd.to_numeric(close, errors='coerce').dropna()
    if valid.empty or not np.isfinite(valid.iloc[-1]) or valid.iloc[-1] <= -100:
        raise DataError('The Treasury yield (^TNX) is unavailable. Please try again later.')
    return float(valid.iloc[-1])


def filedownload(df, text='Download CSV File'):
    encoded = base64.b64encode(df.to_csv(index=True).encode()).decode()
    return f'<a href="data:file/csv;base64,{encoded}" download="Download.csv">{text}</a>'


def display_figure(fig):
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


def plot_stock_prices(prices):
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=prices, linewidth=2.5, ax=ax)
    ax.set_title('Stock Price Timelines', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Close Price (USD)', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Tickers', fontsize=12, title_fontsize=14, loc='upper left')
    ax.grid(visible=True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    display_figure(fig)


def plot_normalized_prices(prices):
    # Each series starts at its first available observation; returns are never filled.
    baseline = prices.apply(lambda column: column.dropna().iloc[0])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=prices / baseline, linewidth=2.5, ax=ax)
    ax.set_title('Normalized Stock Price Timelines', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Normalized Price', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Tickers', fontsize=12, title_fontsize=14, loc='upper left')
    ax.grid(visible=True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    display_figure(fig)


def plot_correlation_matrix(correlation):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='vlag', linewidths=1,
                linecolor='white', cbar_kws={'shrink': 0.8, 'aspect': 30}, square=True, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=12, rotation=0)
    fig.tight_layout()
    display_figure(fig)


def plot_returns_kde_per_asset(returns, interval):
    fig, ax = plt.subplots(figsize=(12, 8))
    for column in returns.columns:
        sns.kdeplot(data=returns[column], label=column, fill=False, linewidth=2, ax=ax)
    ax.set_title('Distribution of Returns by Asset (KDE)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'Return ({interval})', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(title='Assets', fontsize=12, title_fontsize=14)
    fig.tight_layout()
    display_figure(fig)


def plot_efficient_frontier_and_max_sharpe(mean_returns, covariance, risk_free_rate, solution):
    """Render the same validated solution used by the weights and metrics tables."""
    random_weights = np.random.dirichlet(np.ones(len(mean_returns)), 1000)
    random_returns = random_weights @ mean_returns
    random_risks = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, covariance, random_weights))
    random_sharpes = (random_returns - risk_free_rate) / random_risks
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(random_risks, random_returns, c=random_sharpes, cmap='viridis',
                         marker='.', alpha=0.7, label='Random Portfolios')
    ax.plot(solution.frontier_risks, solution.frontier_returns, color='blue', linewidth=2,
            marker='.' if len(solution.frontier_returns) == 1 else None, label='Efficient Frontier')
    ax.scatter(solution.min_risk, solution.min_return, marker='o', color='green', s=100,
               label='Minimum Variance Portfolio')
    ax.scatter(solution.max_risk, solution.max_return, marker='*', color='red', s=200,
               label='Max Sharpe Ratio Portfolio')
    ax.set_title('Efficient Frontier and Random Portfolios', fontsize=16)
    ax.set_xlabel('Volatility (Risk)', fontsize=14)
    ax.set_ylabel('Return', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    fig.tight_layout()
    display_figure(fig)


filtro_etf = st.sidebar.radio('Asset Class', options=('All', 'Stock', 'ETF'))
try:
    import_df = load_data()
except ftp_errors + (UnicodeError, DataError, pd.errors.ParserError) as exc:
    st.error('The Nasdaq symbol directory is unavailable. Please try again later.')
    st.stop()

preferred_exchanges = ['NASDAQ', 'NYSE', 'NYSE Arca', 'BATS', 'NYSE MKT', 'IEX']
exchanges = import_df['Exchange'].dropna().unique().tolist()
exchange_options = [name for name in preferred_exchanges if name in exchanges]
exchange_options += sorted(set(exchanges) - set(preferred_exchanges))
filtro_category = st.sidebar.radio('Exchange', options=['All'] + exchange_options)
df_filtrada = import_df
if filtro_etf != 'All':
    df_filtrada = df_filtrada[df_filtrada['ETF'] == ('N' if filtro_etf == 'Stock' else 'Y')]
if filtro_category != 'All':
    df_filtrada = df_filtrada[df_filtrada['Exchange'] == filtro_category]
serch_option = st.sidebar.radio('Serch by:', options=('Symbol', 'Security Name'))
sorted_stock_unique = sorted(df_filtrada[serch_option].dropna().astype(str).unique())
selected_stock = st.sidebar.multiselect('Select Assets', sorted_stock_unique)
time_delta = st.sidebar.selectbox('Period', options=('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'), index=5)
intervalo = st.sidebar.selectbox('Interval', options=('1d', '5d', '1mo', '3mo'))

st.title('Portfolio Optimization and Financial Analytics Tool')
st.subheader('Analyze stocks and ETFs from NASDAQ and other U.S. exchanges, download time series, compute returns, variance-covariance and correlation matrices, and optimize portfolios using the Sharpe Ratio with data powered by yFinance.')
st.markdown('### Select the assets on the sidebar and click on go button')
st.markdown('List of available assets:')
st.write(df_filtrada)

if st.sidebar.button('Go'):
    df_selected_stock = df_filtrada[df_filtrada[serch_option].isin(selected_stock)]
    ticker_list = list(dict.fromkeys(df_selected_stock.Symbol.tolist()))
    if not ticker_list:
        st.warning('You must select an asset first')
    else:
        st.markdown('### Selected Assets')
        st.table(df_selected_stock)
        try:
            data_stocks = download_data('Price data', tickers=ticker_list, period=time_delta, interval=intervalo)
            time_series_close = extract_close_prices(data_stocks, ticker_list)
        except DataError as exc:
            st.error(str(exc))
            st.stop()
        if len(ticker_list) == 1:
            st.markdown(filedownload(data_stocks.reset_index(level=0), text='Download Time Series'), unsafe_allow_html=True)
        else:
            st.write('Filtered DataFrame with Close Prices:')
            st.write(time_series_close)
            st.markdown(filedownload(time_series_close, text='Download Time Series'), unsafe_allow_html=True)
            try:
                df_stocks = prepare_returns(time_series_close)
            except DataError as exc:
                st.error(str(exc))
                st.stop()
            st.markdown(filedownload(df_stocks, text=f'Download Returns ({intervalo})'), unsafe_allow_html=True)
            if time_series_close.isna().any().any():
                st.info(f'Partial price history: using {len(df_stocks)} common return observations without filling missing prices.')
            try:
                mean_ret = df_stocks.mean(axis=0).to_numpy()
                varcov = df_stocks.cov().to_numpy()
                ten_year_yield = download_data('The Treasury yield (^TNX)', tickers='^TNX', period='1d', interval='1d')
                annual_yield = read_treasury_yield(ten_year_yield)
                # Preserve the existing rate convention; methodology is a separate decision.
                days_mapping = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90}
                daily_yield = (1 + annual_yield / 100) ** (days_mapping[intervalo] / 365) - 1
                solution = portfolio.optimize_portfolio(mean_ret, varcov, daily_yield)
            except (DataError, portfolio.PortfolioError) as exc:
                st.error(str(exc))
                st.stop()

            sns.set_theme(style='darkgrid', palette='tab10')
            st.write('### Stock Price Timelines')
            plot_stock_prices(time_series_close)
            st.write('### Normalized Stock Price Timelines')
            plot_normalized_prices(time_series_close)
            st.write('### Variance - Covariance Matrix')
            st.write(df_stocks.cov())
            correlation = df_stocks.corr().round(2)
            st.write('### Correlation Matrix')
            st.write(correlation)
            plot_correlation_matrix(correlation)
            st.write('### Distribution of Returns by Asset (KDE)')
            plot_returns_kde_per_asset(df_stocks, intervalo)

            result = pd.DataFrame({'Symbol': df_stocks.columns, 'Opt Weight': solution.weights, 'Avg Return': mean_ret})
            st.write('### Portfolio Optimization: Optimal Weights for Selected Assets')
            st.write('Long-only: weights are between 0 and 1 and sum to 1.')
            st.write(result)
            st.write('I used the ^TNX index, representing the 10-Year U.S. Treasury yield, as a proxy for the risk-free rate:')
            st.write(f'Risk Free Rate: {annual_yield:.2f}%')
            st.write(f'Risk Free Yield for the interval {intervalo}: {daily_yield * 100:.2f}%')
            st.write('### Portfolio Metrics')
            st.table(pd.DataFrame([{'Portfolio': 'Long-only Max Sharpe Ratio', 'Return': solution.max_return,
                                   'Std Dev': solution.max_risk, 'Sharpe Ratio': solution.max_sharpe}]))
            st.write('### Efficient Frontier and Portfolio Optimization')
            plot_efficient_frontier_and_max_sharpe(mean_ret, varcov, daily_yield, solution)

st.write('This web app was created by Axel Rodrigo for educational purposes. For questions and comments, please contact axel.23@hotmail.com.')
