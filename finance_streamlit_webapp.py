import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import html5lib
import lxml
from bs4 import BeautifulSoup
#import pandas_datareader.data as web
from datetime import datetime, timedelta
import pickle
from ftplib import FTP
#yf.pdr_override()
directory = 'symboldirectory'
filenames = ('otherlisted.txt', 'nasdaqlisted.txt')
 

filtro_etf = st.sidebar.radio("Asset Class", options= ("All" , "Stock", "ETF"))
filtro_category = st.sidebar.radio("Exchange", options= ("All", "NASDAQ", "NYSE"))

st.cache_resource()
def load_data():

    ftp = FTP('ftp.nasdaqtrader.com')
    ftp.login()
    ftp.cwd(directory)

    for item in filenames:
        ftp.retrbinary('RETR {0}'.format(item), open(item, 'wb').write)

    ftp.quit()

    nasdaq_exchange_info = pd.read_csv('nasdaqlisted.txt',  sep='|')
    other_exchange_info = pd.read_csv('otherlisted.txt',  sep='|')
    nasdaq_exchange_info = nasdaq_exchange_info[nasdaq_exchange_info['Financial Status']=='N']
    nasdaq_exchange_info = nasdaq_exchange_info[nasdaq_exchange_info['Test Issue']=='N']
    nasdaq_exchange_info = nasdaq_exchange_info.drop(columns=['Test Issue', 'Financial Status', "Round Lot Size", "NextShares", "Market Category"])
    nasdaq_exchange_info["Exchange"] = "Nasdaq"


    other_exchange_info = other_exchange_info[other_exchange_info['Test Issue']=='N']
    other_exchange_info = other_exchange_info.drop(columns=['Test Issue', "Round Lot Size", "ACT Symbol", "Exchange", "CQS Symbol"])
    other_exchange_info = other_exchange_info.rename(columns={"NASDAQ Symbol": "Symbol"})
    other_exchange_info = other_exchange_info[["Symbol", "Security Name", "ETF"]]
    other_exchange_info["Exchange"] = "Nyse"
    import_df = pd.concat([nasdaq_exchange_info, other_exchange_info])
    return import_df

import_df = load_data()

df_filtrada = import_df

if filtro_etf != 'All':
   
    if filtro_etf == 'Stock':
        df_filtrada = df_filtrada[df_filtrada.ETF == "N"]
    elif filtro_etf == 'ETF':
        df_filtrada = df_filtrada[df_filtrada.ETF == "Y"]

    

 
if filtro_category != "All":
    if filtro_category == "NASDAQ":
        df_filtrada = df_filtrada[df_filtrada["Exchange"] == "Nasdaq"]
    elif filtro_category == "NYSE":
        df_filtrada = df_filtrada[df_filtrada["Exchange"] == "Nyse"]


serch_option = st.sidebar.radio("Serch by:", options=("Symbol", "Security Name"))

# Sidebar - Stock selection
sorted_stock_unique = sorted(df_filtrada[serch_option].astype(str).unique())
selected_stock = st.sidebar.multiselect('Select Assets', sorted_stock_unique)


time_delta = st.sidebar.selectbox("Period", options=  ("1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"), index=5)
intervalo = st.sidebar.selectbox("Interval", options= ("1d","5d","1mo","3mo"))

# Filtering data

st.title("Portfolio Optimization and Financial Analytics Tool")
st.subheader("Analyze stocks and ETFs from NASDAQ and NYSE, download time series, compute returns, variance-covariance and correlation matrices, and optimize portfolios using the Sharpe Ratio with data powered by yFinance.")
st.markdown("### Select the assets on the sidebar and click on go button")
st.markdown("List of available assets:")
st.write(df_filtrada)

close_string = 'Close'

def filedownload(df, text="Download CSV File"):
    """
    Generates a download link for a pandas DataFrame.

    Parameters:
        df: pandas.DataFrame
            The DataFrame to download.
        text: str
            The text displayed for the download link.

    Returns:
        str: HTML for the download link.
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode CSV to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="Download.csv">{text}</a>'
    return href




if st.sidebar.button("Go") :
    df_selected_stock = df_filtrada[(df_filtrada[serch_option].isin(selected_stock))]
    ticker_list = df_selected_stock.Symbol.tolist()

    if not ticker_list:
        st.warning("""You must select an asset first""")
    else:
        
        st.markdown("### Selected Assets")
        st.table(df_selected_stock)
        data_stocks = yf.download(tickers = ticker_list , period = time_delta, interval= intervalo)
        

        
        if len(ticker_list) == 1:
            df_stocks = data_stocks.reset_index(level=0)
            st.markdown(filedownload(df_stocks, text="Download Time Series"), unsafe_allow_html=True)
        else:

            # Procesamiento inicial
            order = [1, 0]
            df_stocks = data_stocks.reset_index(level=0).reorder_levels(order, axis=1)
            time_series = df_stocks

            # Extraer la columna 'Date' del MultiIndex y configurarla como índice
            if '' in time_series.columns.get_level_values('Ticker'):
                time_series['Date'] = time_series[('', 'Date')]  # Accede a la columna vacía con 'Date'
                time_series = time_series.drop(columns=[('', 'Date')])  # Elimina la columna del MultiIndex
                time_series = time_series.set_index('Date')  # Configura 'Date' como índice

            # Filtrar solo las columnas del nivel 'Close'
            time_series_close = time_series.xs(key="Close", axis=1, level="Price")

            # Mostrar el DataFrame resultante
            st.write("Filtered DataFrame with Close Prices:")
            st.write(time_series_close)



            st.markdown(filedownload(time_series_close, text="Download Time Series"), unsafe_allow_html=True)
            df_stocks = df_stocks.xs("Close", axis=1, level=1, drop_level=True).pct_change().dropna().iloc[1:]
            st.markdown(filedownload(df_stocks, text="Download Daily Returns"), unsafe_allow_html=True)

            # Configurar estilo de seaborn
            sns.set_theme(style="darkgrid", palette="tab10")

            # Gráfico de precios de cierre
            def plot_stock_prices(time_series_close):
                plt.figure(figsize=(14, 8))
                sns.lineplot(data=time_series_close, linewidth=2.5)
                plt.title("Stock Price Timelines", fontsize=18, fontweight="bold", pad=15)
                plt.xlabel("Date", fontsize=14)
                plt.ylabel("Close Price (USD)", fontsize=14)
                plt.xticks(rotation=45, fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(title="Tickers", fontsize=12, title_fontsize=14, loc="upper left")
                plt.grid(visible=True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(plt)

            # Gráfico de precios normalizados
            def plot_normalized_prices(time_series_close):
                # Normalizar precios
                normalized_data = time_series_close / time_series_close.iloc[0]

                plt.figure(figsize=(14, 8))
                sns.lineplot(data=normalized_data, linewidth=2.5)
                plt.title("Normalized Stock Price Timelines", fontsize=18, fontweight="bold", pad=15)
                plt.xlabel("Date", fontsize=14)
                plt.ylabel("Normalized Price", fontsize=14)
                plt.xticks(rotation=45, fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(title="Tickers", fontsize=12, title_fontsize=14, loc="upper left")
                plt.grid(visible=True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(plt)

            # Mostrar gráficos en Streamlit
            st.write("### Stock Price Timelines")
            plot_stock_prices(time_series_close)

            st.write("### Normalized Stock Price Timelines")
            plot_normalized_prices(time_series_close)

            varcov = df_stocks.cov()
            st.write("### Variance - Covariance Matrix")
            st.write(varcov)
            
            # Calcular la matriz de correlación
            correlation_matrix = df_stocks.corr()

            # Redondear a dos decimales
            correlation_matrix_rounded = correlation_matrix.round(2)

            # Función para graficar la matriz de correlación
            def plot_correlation_matrix(corr_matrix):
                plt.figure(figsize=(10, 8))  # Tamaño del gráfico
                sns.heatmap(
                    corr_matrix,
                    annot=True,  # Mostrar los valores en el gráfico
                    fmt=".2f",  # Formato de los valores (dos decimales)
                    cmap="vlag",  # Escala de colores moderna
                    linewidths=1,  # Líneas entre celdas
                    linecolor="white",  # Color de las líneas
                    cbar_kws={"shrink": 0.8, "aspect": 30},  # Barra de color más estilizada
                    square=True  # Celdas cuadradas para un diseño uniforme
                )
                plt.title("Correlation Matrix", fontsize=16, fontweight="bold", pad=20)  # Título mejorado
                plt.xticks(fontsize=12, rotation=45)  # Etiquetas en el eje X
                plt.yticks(fontsize=12, rotation=0)  # Etiquetas en el eje Y
                plt.tight_layout()  # Ajusta el diseño para evitar solapamientos
                st.pyplot(plt)

            # Mostrar la matriz de correlación en Streamlit
            st.write("### Correlation Matrix")
            st.write(correlation_matrix_rounded)  # Mostrar la tabla como DataFrame
            plot_correlation_matrix(correlation_matrix_rounded)  # Mostrar el gráfico

            # Función para graficar el KDE por activo
            def plot_volatility_kde_per_asset(df_stocks):
                plt.figure(figsize=(12, 8))  # Tamaño del gráfico
                for column in df_stocks.columns:
                    sns.kdeplot(
                        data=df_stocks[column],
                        label=column,  # Nombre del activo
                        fill=False,  # Desactiva el relleno bajo la curva
                        linewidth=2  # Grosor de las líneas
                    )
                plt.title("Distribution of Volatility by Asset (KDE)", fontsize=16, fontweight="bold", pad=20)
                plt.xlabel("Volatility", fontsize=14)
                plt.ylabel("Density", fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(title="Assets", fontsize=12, title_fontsize=14)  # Leyenda con el nombre de los activos
                plt.tight_layout()
                st.pyplot(plt)

            # Mostrar el gráfico en Streamlit
            st.write("### Distribution of Volatility by Asset (KDE)")
            plot_volatility_kde_per_asset(df_stocks)

            ten_year = "^TNX"
            ten_year_yield = yf.download(tickers = ten_year , period = "1d", interval= "1d")

            

            # Extraer el valor de 'Close' y conservar todos los decimales
            if "Close" in ten_year_yield.columns and not ten_year_yield.empty:
                annual_yield = float(ten_year_yield["Close"].iloc[0])  # Usar iloc[0] y convertir a float
                
            else:
                st.error("Error: La columna 'Close' no está disponible o el DataFrame está vacío.")

            # Mapear intervalos a días
            days_mapping = {
                "1d": 1,
                "5d": 5,
                "1mo": 30,
                "3mo": 90
            }

            # Obtener días del intervalo seleccionado
            days = days_mapping[intervalo]

           # Calcular el rendimiento ajustado para el intervalo
            daily_yield = (1 + annual_yield / 100) ** (days / 365) - 1

            # Calcular estadísticas iniciales
            mean_ret = df_stocks.mean(axis=0).values  # Rendimientos medios
            varcov = df_stocks.cov()  # Matriz de covarianza
            ticker_list2 = df_stocks.columns.tolist()

            # Pesos optimizados (Short Allowed)
            excess_return = mean_ret - daily_yield
            weights = np.linalg.inv(varcov) @ excess_return / np.sum(np.linalg.inv(varcov) @ excess_return)

            # Crear DataFrame con los pesos
            result = pd.DataFrame({"Symbol": ticker_list2, "Opt Weight": weights})
            result["Avg Return"] = mean_ret

            # Mostrar resultados iniciales
            st.write("### Portfolio Otimization: Optimal Weights for Selected Assets")
            st.write(result)
            st.write("I used the ^TNX index, representing the 10-Year U.S. Treasury yield, as a proxy for the risk-free rate:")
            st.write(f"Risk Free Rate: {annual_yield:.2f}%")
            st.write(f"Risk Free Yield for the interval {intervalo}: {daily_yield * 100:.2f}%")

            # Función para calcular métricas del portafolio
            def calculate_portfolio_metrics(weights, mean_ret, varcov, risk_free_rate):
                portfolio_return = np.dot(weights, mean_ret)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(varcov, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                return portfolio_return, portfolio_volatility, sharpe_ratio

            # Calcular métricas del portafolio optimizado
            ret, risk, sharpe = calculate_portfolio_metrics(weights, mean_ret, varcov, daily_yield)

            # Mostrar métricas del portafolio
            portfolio_results_df = pd.DataFrame([{
                "Portfolio": "Frontier Eficient Max Sharpe Ratio",
                "Return": ret,
                "Std Dev": risk,
                "Sharpe Ratio": sharpe,
            }])

            st.write("### Portfolio Metrics")
            st.table(portfolio_results_df)
            
            def plot_efficient_frontier_and_max_sharpe(mean_ret, varcov, risk_free_rate=daily_yield, points=50):
                """
                Plots the efficient frontier, maximum Sharpe ratio portfolio, and random portfolios.

                Parameters:
                - mean_ret: Array of mean returns for the assets.
                - varcov: Covariance matrix of asset returns.
                - risk_free_rate: Risk-free rate for Sharpe ratio calculation.
                - points: Number of points to calculate for the efficient frontier.
                """
                n_assets = len(mean_ret)
                n_samples = 1000  # Number of random portfolios to generate

                # Function to calculate portfolio performance
                def portfolio_performance(weights):
                    ret = np.dot(weights, mean_ret)
                    risk = np.sqrt(np.dot(weights.T, np.dot(varcov, weights)))
                    sharpe = (ret - risk_free_rate) / risk if risk > 0 else 0
                    return ret, risk, sharpe

                # Optimize for the minimum variance portfolio
                def min_variance_portfolio():
                    def risk(weights):
                        _, risk, _ = portfolio_performance(weights)
                        return risk

                    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights must sum to 1
                    bounds = [(0, 1) for _ in range(n_assets)]
                    initial_weights = np.ones(n_assets) / n_assets

                    result = minimize(risk, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
                    return result.x

                min_var_weights = min_variance_portfolio()
                min_var_ret, min_var_risk, _ = portfolio_performance(min_var_weights)

                # Generate points for the efficient frontier above the minimum variance portfolio
                target_returns = np.linspace(min_var_ret, mean_ret.max(), points)
                efficient_risks = []
                for target in target_returns:
                    def risk(weights):
                        _, risk, _ = portfolio_performance(weights)
                        return risk

                    constraints = [
                        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Weights must sum to 1
                        {"type": "eq", "fun": lambda w: np.dot(w, mean_ret) - target}  # Target return
                    ]
                    bounds = [(0, 1) for _ in range(n_assets)]
                    initial_weights = np.ones(n_assets) / n_assets

                    result = minimize(risk, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
                    weights = result.x
                    _, risk, _ = portfolio_performance(weights)
                    efficient_risks.append(risk)

                # Optimize for maximum Sharpe ratio
                def max_sharpe_ratio():
                    def neg_sharpe(weights):
                        _, _, sharpe = portfolio_performance(weights)
                        return -sharpe  # Negative because we want to maximize

                    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
                    bounds = [(0, 1) for _ in range(n_assets)]
                    initial_weights = np.ones(n_assets) / n_assets

                    result = minimize(neg_sharpe, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
                    return result.x

                max_sharpe_weights = max_sharpe_ratio()
                max_ret, max_risk, _ = portfolio_performance(max_sharpe_weights)

                # Generate random portfolios
                random_weights = np.random.dirichlet(np.ones(n_assets), n_samples)
                random_returns = []
                random_risks = []
                random_sharpes = []

                for w in random_weights:
                    ret, risk, sharpe = portfolio_performance(w)
                    random_returns.append(ret)
                    random_risks.append(risk)
                    random_sharpes.append(sharpe)

                # Plot efficient frontier
                plt.figure(figsize=(12, 8))
                plt.scatter(random_risks, random_returns, c=random_sharpes, cmap="viridis", marker=".", alpha=0.7, label="Random Portfolios")
                plt.plot(efficient_risks, target_returns, color="blue", linewidth=2, label="Efficient Frontier")
                plt.scatter(min_var_risk, min_var_ret, marker="o", color="green", s=100, label="Minimum Variance Portfolio")
                plt.scatter(max_risk, max_ret, marker="*", color="red", s=200, label="Max Sharpe Ratio Portfolio")

                # Add labels and title
                plt.title("Efficient Frontier and Random Portfolios", fontsize=16)
                plt.xlabel("Volatility (Risk)", fontsize=14)
                plt.ylabel("Return", fontsize=14)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.colorbar(label="Sharpe Ratio")
                plt.tight_layout()

                # Display the plot in Streamlit
                st.pyplot(plt)

            # Usar tus variables
            st.write("### Efficient Frontier and Portfolio Optimization")
            plot_efficient_frontier_and_max_sharpe(mean_ret, varcov, daily_yield)

st.write("This web app was created by Axel Rodrigo for educational purposes. For questions and comments, please contact axel.23@hotmail.com."







)



                                    
            
                    




    



