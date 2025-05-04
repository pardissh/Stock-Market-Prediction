import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf as compute_acf, pacf as compute_pacf
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
import logging
from multiprocessing import Pool
import os
import uuid

# Initialize session state
if 'grid_search_result' not in st.session_state:
    st.session_state.grid_search_result = None
if 'grid_search_status' not in st.session_state:
    st.session_state.grid_search_status = "not_started"
if 'period_cache' not in st.session_state:
    st.session_state.period_cache = {}

# Configure logging
logging.basicConfig(level=logging.INFO, filename='stock_analysis.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# StockMarketAnalysis class
class StockMarketAnalysis:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_data()
        logging.info(f"Initialized analysis for {symbol} from {start_date} to {end_date}")

    def load_data(self):
        start_date = pd.Timestamp(self.start_date).tz_localize('UTC')
        end_date = pd.Timestamp(self.end_date).tz_localize('UTC')
        historical_file = f"{self.symbol.lower()}_historical.csv"

        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Looking for CSV file at: {os.path.abspath(historical_file)}")

        if not os.path.exists(historical_file):
            raise FileNotFoundError(f"CSV file {historical_file} not found.")

        try:
            historical_data = pd.read_csv(historical_file, index_col="Date", parse_dates=True)
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data.index = pd.to_datetime(historical_data.index)
            if historical_data.index.tz is None:
                historical_data.index = historical_data.index.tz_localize('UTC')
            else:
                historical_data.index = historical_data.index.tz_convert('UTC')

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            logging.info(f"CSV columns: {historical_data.columns.tolist()}")
            logging.info(f"CSV date range: {historical_data.index.min()} to {historical_data.index.max()}")

            if not all(col in historical_data.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in historical_data.columns]
                raise ValueError(f"CSV file {historical_file} missing columns: {missing_cols}")
            if historical_data.empty:
                raise ValueError(f"CSV file {historical_file} is empty.")

            if 'Adj Close' not in historical_data.columns:
                historical_data['Adj Close'] = historical_data['Close']
                logging.info(f"Added 'Adj Close' column for {self.symbol} by copying 'Close'")

            historical_start = historical_data.index.min()
            historical_end = historical_data.index.max()

            adjusted_start = max(start_date, historical_start)
            adjusted_end = min(end_date, historical_end)
            if adjusted_start != start_date or adjusted_end != end_date:
                logging.info(
                    f"Adjusted date range from {start_date.date()} to {end_date.date()} "
                    f"to {adjusted_start.date()} to {adjusted_end.date()} to match CSV data."
                )
                st.warning(
                    f"Requested date range {start_date.date()} to {end_date.date()} "
                    f"adjusted to {adjusted_start.date()} to {adjusted_end.date()} "
                    f"to match available CSV data."
                )

            filtered_data = historical_data[
                (historical_data.index >= adjusted_start) & (historical_data.index <= adjusted_end)
            ]
            if filtered_data.empty:
                raise ValueError(
                    f"No data in CSV for {self.symbol} from {adjusted_start.date()} to {adjusted_end.date()}."
                )

            logging.info(f"Using CSV data for {self.symbol} from {adjusted_start.date()} to {adjusted_end.date()}")
            return filtered_data

        except Exception as e:
            logging.error(f"Failed to load CSV {historical_file}: {str(e)}")
            raise ValueError(f"Failed to load CSV for {self.symbol}: {str(e)}")

    def calculate_yearly_trading_days(self):
        if not isinstance(self.data.index, pd.DatetimeIndex):
            st.error("Data index is not a DatetimeIndex. Cannot calculate trading days.")
            logging.error("Invalid index type for calculating trading days")
            return None
        key = f"{self.symbol}_{self.start_date}_{self.end_date}"
        if key in st.session_state.period_cache:
            return st.session_state.period_cache[key]
        yearly_trading_days = {}
        start_year = self.data.index.min().year
        end_year = self.data.index.max().year
        tz = self.data.index.tz
        for year in range(start_year, end_year + 1):
            start_date = pd.Timestamp(f'{year}-01-01', tz=tz)
            end_date = pd.Timestamp(f'{year}-12-31', tz=tz)
            year_data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
            yearly_trading_days[year] = len(year_data)
        st.session_state.period_cache[key] = yearly_trading_days
        logging.info(f"Calculated trading days for {self.symbol}: {yearly_trading_days}")
        return yearly_trading_days

    def determine_sarima_period(self, data_length, default_period):
        # Prefer quarterly (63) or monthly (21) periods unless yearly is strongly justified
        fallback_periods = [
            ('quarterly', 63 if data_length >= 2 * 63 else None),
            ('monthly', 21 if data_length >= 2 * 21 else None),
            ('yearly', default_period if data_length >= 2 * default_period else None)
        ]
        for period_type, period in fallback_periods:
            if period is not None:
                logging.info(f"Selected {period_type} period {period} for SARIMA (data length: {data_length})")
                return period
        logging.warning(f"Data length {data_length} too short for SARIMA with minimum period {fallback_periods[1][1]}")
        return None

    def decompose_time_series(self, model='multiplicative', period=None, plot=True):
        if period is None:
            st.warning("Period not specified for decomposition. Using default period of 252 trading days.")
            logging.warning("No period specified for decomposition, defaulting to 252")
            period = 252

        data_length = len(self.data['Close'])
        # Define fallback periods: quarterly (~63 trading days), monthly (~21 trading days)
        fallback_periods = [
            ('quarterly', max(63, period // 4)),
            ('monthly', max(21, period // 12))
        ]

        # Check if data is sufficient for the provided period
        if data_length < 2 * period:
            for period_type, fallback_period in fallback_periods:
                if data_length >= 2 * fallback_period:
                    st.warning(
                        f"Insufficient data for decomposition with period {period} "
                        f"(need at least {2 * period} observations, got {data_length}). "
                        f"Using {period_type} period of {fallback_period} trading days."
                    )
                    logging.info(
                        f"Switched to {period_type} period {fallback_period} for decomposition "
                        f"(data length: {data_length}, required: {2 * period})"
                    )
                    period = fallback_period
                    break
            else:
                st.warning(
                    f"Insufficient data for decomposition even with monthly period "
                    f"(need at least {2 * fallback_periods[-1][1]} observations, got {data_length}). "
                    f"Skipping decomposition."
                )
                logging.warning(
                    f"Data too short for decomposition: {data_length} observations, "
                    f"minimum required: {2 * fallback_periods[-1][1]}"
                )
                return None, None

        try:
            result = seasonal_decompose(self.data['Close'], model=model, period=period)
            if plot:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
                ax1.plot(result.observed)
                ax1.set_title('Observed')
                ax2.plot(result.trend)
                ax2.set_title('Trend')
                ax3.plot(result.seasonal)
                ax3.set_title('Seasonal')
                ax4.plot(result.resid)
                ax4.set_title('Residual')
                plt.tight_layout()
                logging.info(f"Generated decomposition plot for {self.symbol} with period {period}")
                return result, fig
            return result, None
        except Exception as e:
            st.warning(f"Decomposition failed with period {period}: {e}. Skipping decomposition.")
            logging.error(f"Decomposition failed for {self.symbol} with period {period}: {str(e)}")
            return None, None

    def generate_decomposition_interpretation(self, decomposition, period):
        if decomposition is None:
            return "Unable to generate interpretation due to decomposition failure or insufficient data."
        try:
            trend = decomposition.trend.dropna()
            if len(trend) > 1:
                trend_change = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100
                trend_mid = trend.iloc[len(trend)//2]
                mid_change = (trend_mid - trend.iloc[0]) / trend.iloc[0] * 100
                end_mid_change = (trend.iloc[-1] - trend_mid) / trend_mid * 100
                if trend_change > 5:
                    if mid_change < -5 and end_mid_change > 5:
                        trend_desc = "an overall upward movement with a mid-period decline, reflecting long-term growth despite fluctuations"
                    else:
                        trend_desc = "a consistent upward movement, reflecting long-term growth"
                else:
                    trend_desc = "a relatively flat trend, indicating stable long-term behavior"
            else:
                trend_desc = "insufficient data to determine trend direction"

            seasonal = decomposition.seasonal.dropna()
            seasonal_amplitude = seasonal.max() - seasonal.min()
            seasonal_std = self.data['Close'].std()
            period_desc = "yearly" if period >= 200 else "quarterly" if period >= 50 else "monthly"
            if seasonal_amplitude / seasonal_std > 0.1:
                seasonal_desc = f"significant {period_desc} patterns with a period of {period} trading days"
            else:
                seasonal_desc = f"subtle {period_desc} patterns with a period of {period} trading days, suggesting seasonality is not dominant"

            resid = decomposition.resid.dropna()
            resid_std = resid.std()
            if resid_std / self.data['Close'].std() > 0.5:
                resid_desc = "highly noisy residuals, indicating significant random fluctuations or external events"
            else:
                resid_desc = "moderately noisy residuals, suggesting minor random fluctuations"

            interpretation = (
                f"The trend component shows {trend_desc} (change: {trend_change:.2f}%). "
                f"The seasonal component exhibits {seasonal_desc} (amplitude: {seasonal_amplitude:.2f}). "
                f"The residuals are {resid_desc} (std: {resid_std:.2f})."
            )
            logging.info(f"Decomposition interpretation for {self.symbol}: {interpretation}")
            return interpretation
        except Exception as e:
            logging.error(f"Failed to generate decomposition interpretation for {self.symbol}: {str(e)}")
            return f"Error generating interpretation: {e}"

    def summarize_data(self):
        return self.data['Close'].describe()

    def plot_time_series(self):
        if not isinstance(self.data.index, pd.DatetimeIndex):
            st.error("Data index is not a DatetimeIndex. Cannot plot time series.")
            logging.error("Invalid index type for time series plot")
            return None
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.data.index, self.data['Close'], label='Closing Price')
        ax.set_title(f'{self.symbol} Closing Price Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        logging.info(f"Generated time series plot for {self.symbol}")
        return fig

    def plot_density(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.data['Close'].hist(bins=30, edgecolor='black', alpha=0.7, ax=ax1)
        ax1.set_title('Histogram of Closing Price')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Frequency')
        self.data['Close'].plot(kind='kde', ax=ax2)
        ax2.set_title('Kernel Density Estimate of Closing Price')
        ax2.set_xlabel('Price')
        ax2.set_ylabel('Density')
        plt.tight_layout()
        logging.info(f"Generated density plot for {self.symbol}")
        return fig

    def plot_yearly_boxplot(self):
        if not self.data.empty:
            grouped = self.data.groupby(self.data.index.year)['Close']
            years_data = {year: group.values for year, group in grouped}
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(years_data.values(), labels=years_data.keys())
            ax.set_title(f'Yearly Box Plot of {self.symbol} Closing Price')
            ax.set_xlabel('Year')
            ax.set_ylabel('Price')
            ax.grid(True)
            logging.info(f"Generated yearly boxplot for {self.symbol}")
            return fig
        logging.error(f"No data available for yearly boxplot for {self.symbol}")
        return None

    def plot_volume(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.data.index, self.data['Volume'], label='Volume', color='orange')
        ax.set_title(f'{self.symbol} Trading Volume Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.legend()
        ax.grid(True)
        logging.info(f"Generated volume plot for {self.symbol}")
        return fig

    def plot_returns(self):
        self.data['Returns'] = self.data['Close'].pct_change()
        fig, ax = plt.subplots(figsize=(10, 5))
        self.data['Returns'].plot(label='Daily Returns', ax=ax)
        ax.set_title(f'{self.symbol} Daily Percentage Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Percentage Return')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.legend()
        ax.grid(True)
        logging.info(f"Generated returns plot for {self.symbol}")
        return fig

    def plot_rolling_mean_std(self, window=60):
        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.data.index, self.data['Close'], label='Closing Price')
        ax.plot(self.data.index, rolling_mean, label=f'Rolling Mean (Window={window})', color='red')
        ax.plot(self.data.index, rolling_mean + 2 * rolling_std, label='+2 Rolling Std', color='green', linestyle='--')
        ax.plot(self.data.index, rolling_mean - 2 * rolling_std, label='-2 Rolling Std', color='green', linestyle='--')
        ax.set_title(f'{self.symbol} Closing Price with Rolling Mean and Std (Window={window})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        logging.info(f"Generated rolling mean/std plot for {self.symbol}")
        return fig

    def describe_time_series(self):
        result = adfuller(self.data['Close'])
        original_adf = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
        diff_series = self.create_differenced_series(self.data['Close'])
        result_diff = adfuller(diff_series.dropna())
        differenced_adf = {
            'adf_statistic': result_diff[0],
            'p_value': result_diff[1],
            'critical_values': result_diff[4]
        }
        logging.info(f"Generated time series description for {self.symbol}")
        return {
            'describe': self.data.describe(),
            'original_adf': original_adf,
            'differenced_adf': differenced_adf
        }

    def check_stationarity(self, series, max_diff=3):
        diff_series = series.copy()
        d = 0
        result = adfuller(diff_series)
        if result[1] < 0.05:
            return d
        for i in range(1, max_diff + 1):
            diff_series = self.create_differenced_series(diff_series)
            d = i
            result = adfuller(diff_series.dropna())
            if result[1] < 0.05:
                return d
        logging.warning(f"Stationarity not achieved for {self.symbol} after {max_diff} differencings")
        return None

    def create_differenced_series(self, data, interval=1):
        diff = [data[i] - data[i - interval] for i in range(interval, len(data))]
        return pd.Series(diff, index=data.index[interval:])

    def plot_acf_pacf(self, series, lags=20):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_acf(series, lags=lags, ax=plt.gca())
        plt.title('ACF')
        plt.subplot(1, 2, 2)
        plot_pacf(series, lags=lags, ax=plt.gca())
        plt.title('PACF')
        plt.tight_layout()
        logging.info(f"Generated ACF/PACF plot for {self.symbol}")
        return plt.gcf()

    def analyze_stationarity(self):
        d_level = self.check_stationarity(self.data['Close'])
        if d_level is not None:
            differenced_close = self.data['Close']
            for _ in range(d_level):
                differenced_close = self.create_differenced_series(differenced_close)
            logging.info(f"Stationarity achieved for {self.symbol} with d={d_level}")
            return d_level, differenced_close
        logging.error(f"Failed to achieve stationarity for {self.symbol}")
        return None, None

    def generate_stationarity_interpretation(self, d_level, diff_series):
        try:
            if d_level is None:
                return "Stationarity not achieved with differencing, making ARIMA parameter estimation challenging."
            acf_vals = compute_acf(diff_series.dropna(), nlags=20, fft=False)
            pacf_vals = compute_pacf(diff_series.dropna(), nlags=20)
            significant_lags_acf = sum(abs(acf_vals[1:]) > 0.1)
            significant_lags_pacf = sum(abs(pacf_vals[1:]) > 0.1)
            q_suggestion = "q=0" if significant_lags_acf <= 1 else f"q={significant_lags_acf}"
            p_suggestion = "p=0" if significant_lags_pacf <= 1 else f"p={significant_lags_pacf}"
            model_desc = "aligning with a random walk model" if p_suggestion == "p=0" and q_suggestion == "q=0" else "suggesting a more complex ARIMA model"
            interpretation = (
                f"The ACF shows {significant_lags_acf} significant lag(s), suggesting {q_suggestion}. "
                f"The PACF shows {significant_lags_pacf} significant lag(s), suggesting {p_suggestion}, "
                f"{model_desc}."
            )
            logging.info(f"Stationarity interpretation for {self.symbol}: {interpretation}")
            return interpretation
        except Exception as e:
            logging.error(f"Failed to generate stationarity interpretation for {self.symbol}: {str(e)}")
            return f"Error generating stationarity interpretation: {e}"

    @st.cache_data
    def evaluate_arima_model(_self, series, arima_order):
        # Ensure no missing values
        series = series.interpolate().ffill().bfill()
        logging.info(f"Evaluating ARIMA{arima_order} for {_self.symbol}, data length: {len(series)}")
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        try:
            model = ARIMA(train, order=arima_order)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            rmse = sqrt(mean_squared_error(test, predictions))
            logging.info(f"ARIMA{arima_order} RMSE={rmse:.6f} for {_self.symbol}, train_size={train_size}, test_size={len(test)}")
            return rmse
        except Exception as e:
            logging.error(f"ARIMA{arima_order} evaluation failed for {_self.symbol}: {str(e)}")
            return float("inf")

    @st.cache_data
    def evaluate_sarima_model(_self, series, order, seasonal_order, _period):
        # Ensure no missing values
        series = series.interpolate().ffill().bfill()
        data_length = len(series)
        # Define fallback periods for reference in error message
        fallback_periods = [
            ('quarterly', 63),
            ('monthly', 21)
        ]
        # Determine appropriate seasonal period
        sarima_period = _self.determine_sarima_period(data_length, _period)
        if sarima_period is None:
            st.warning(
                f"Insufficient data for SARIMA with period {_period} "
                f"(need at least {2 * fallback_periods[-1][1]} observations, got {data_length}). "
                f"Skipping SARIMA evaluation."
            )
            logging.warning(
                f"Data too short for SARIMA: {data_length} observations, "
                f"minimum required: {2 * fallback_periods[-1][1]}"
            )
            return float("inf")
        seasonal_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], sarima_period)
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        if len(test) == 0:
            logging.error(f"Empty test set for SARIMA{order}{seasonal_order} for {_self.symbol}")
            return float("inf")
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                maxiter=50,
                method='lbfgs'
            )
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=True)
            rmse = sqrt(mean_squared_error(test, predictions))
            logging.info(f"SARIMA{order}{seasonal_order} RMSE={rmse:.6f} for {_self.symbol}")
            return rmse
        except Exception as e:
            logging.error(f"SARIMA{order}{seasonal_order} evaluation failed for {_self.symbol}: {str(e)}")
            return float("inf")

    def evaluate_tslm(self, series, seasonal_period):
        # Use a quarterly period (63) or monthly (21) based on data length
        data_length = len(series)
        tslm_period = 63 if data_length >= 2 * 63 else 21 if data_length >= 2 * 21 else None
        if tslm_period is None:
            logging.warning(f"Data length {data_length} too short for TSLM with minimum period 21")
            return float("inf"), None
        train_size = int(min(len(series) * 0.8, 500))
        train, test = series[:train_size], series[train_size:]
        train_df = pd.DataFrame({'Close': train})
        train_df['t'] = range(1, len(train) + 1)
        train_df['season'] = pd.Series(train.index).apply(lambda x: (x.dayofyear % tslm_period) + 1).astype(str)
        train_df = pd.get_dummies(train_df, columns=['season'], drop_first=True)
        predictors = ['t'] + [col for col in train_df.columns if col.startswith('season_')]
        try:
            model = OLS(train_df['Close'], train_df[predictors])
            model_fit = model.fit()
            test_df = pd.DataFrame({'Close': test})
            test_df['t'] = range(len(train) + 1, len(train) + len(test) + 1)
            test_df['season'] = pd.Series(test.index).apply(lambda x: (x.dayofyear % tslm_period) + 1).astype(str)
            test_df = pd.get_dummies(test_df, columns=['season'], drop_first=True)
            for col in predictors:
                if col not in test_df.columns:
                    test_df[col] = 0
            predictions = model_fit.predict(test_df[predictors])
            rmse = sqrt(mean_squared_error(test, predictions))
            logging.info(f"TSLM RMSE={rmse:.6f} for {self.symbol}")
            return rmse, model_fit
        except:
            logging.error(f"TSLM evaluation failed for {self.symbol}")
            return float("inf"), None

    def grid_search_arima(self, p_values, d_values, q_values):
        logging.info(f"Starting ARIMA grid search for {self.symbol} with p={p_values}, d={d_values}, q={q_values}")
        best_score, best_order = float("inf"), None
        results = []
        total_iterations = len(p_values) * len(d_values) * len(q_values)
        progress = st.progress(0)
        iteration = 0

        def evaluate_order(order):
            try:
                rmse = self.evaluate_arima_model(self.data['Close'], order)
                logging.info(f"Evaluated ARIMA{order} with RMSE={rmse:.6f} for {self.symbol}")
                return {'order': order, 'p': order[0], 'd': order[1], 'q': order[2], 'rmse': rmse}
            except Exception as e:
                logging.error(f"Failed to evaluate ARIMA{order} for {self.symbol}: {str(e)}")
                return {'order': order, 'p': order[0], 'd': order[1], 'q': order[2], 'rmse': 'Failed'}

        orders = [(p, d, q) for p in p_values for d in d_values for q in q_values]
        if total_iterations > 4:
            with Pool() as pool:
                results = pool.map(evaluate_order, orders)
        else:
            results = [evaluate_order(order) for order in orders]

        for result in results:
            if isinstance(result['rmse'], float) and result['rmse'] < best_score:
                best_score, best_order = result['rmse'], result['order']
            st.write(f"ARIMA({result['p']},{result['d']},{result['q']}) RMSE={result['rmse']}")
            iteration += 1
            progress.progress(iteration / total_iterations)

        progress.empty()
        logging.info(f"Best ARIMA order for {self.symbol}: {best_order} with RMSE={best_score:.6f}")
        return best_order, best_score, results

    def plot_model_residuals(self, order):
        try:
            train_size = int(len(self.data['Close']) * 0.8)
            train = self.data['Close'][:train_size]
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            residuals = pd.DataFrame(model_fit.resid)
            if residuals.isna().any().any():
                st.warning(f"Residuals contain {residuals.isna().sum().sum()} NaN values, which may affect analysis.")
                logging.warning(f"Residuals for ARIMA{order} contain NaN values for {self.symbol}")
            fig = plt.figure(figsize=(12, 15))
            plt.subplot(5, 1, 1)
            residuals.hist(bins=30, edgecolor='black', ax=plt.gca())
            plt.title('Histogram of Residuals')
            plt.subplot(5, 1, 2)
            residuals.plot(kind='kde', ax=plt.gca())
            plt.title('Density of Residuals')
            plt.subplot(5, 1, 3)
            residuals.plot(ax=plt.gca())
            plt.title('Time Series Plot of Residuals')
            plt.xlabel('Time')
            plt.ylabel('Residual Value')
            plt.subplot(5, 1, 4)
            plot_acf(residuals, lags=20, ax=plt.gca())
            plt.title('ACF of Residuals')
            plt.subplot(5, 1, 5)
            plot_pacf(residuals, lags=20, ax=plt.gca())
            plt.title('PACF of Residuals')
            plt.tight_layout()
            logging.info(f"Generated residual plots for ARIMA{order} for {self.symbol}")
            return fig, residuals
        except Exception as e:
            st.error(f"Error plotting residuals for order {order}: {e}")
            logging.error(f"Failed to plot residuals for ARIMA{order} for {self.symbol}: {str(e)}")
            return None, None

    def generate_residual_interpretation(self, residuals):
        try:
            if residuals is None or residuals.empty:
                return "Unable to generate residual interpretation due to model failure."
            residuals_series = residuals.iloc[:, 0] if isinstance(residuals, pd.DataFrame) else residuals
            residuals_series = residuals_series.dropna()
            if residuals_series.empty:
                return "Unable to generate residual interpretation due to insufficient data after removing NaN values."
            mean_residual = residuals_series.mean()
            centering_desc = "centered around zero" if abs(mean_residual) < 0.1 * residuals_series.std() else "not centered around zero, suggesting potential model bias"
            resid_std = residuals_series.std()
            acf_vals = compute_acf(residuals_series, nlags=20, fft=False)
            significant_lags = sum(abs(acf_vals[1:]) > 0.1)
            pattern_desc = "random with no clear patterns" if significant_lags <= 1 else "some patterns, indicating possible unmodeled structure"
            acf_desc = "insignificant lags, indicating the model captures the data’s structure well" if significant_lags <= 1 else f"{significant_lags} significant lags, suggesting some residual autocorrelation"
            interpretation = (
                f"The histogram and KDE show residuals are {centering_desc}. "
                f"The time series plot of residuals appears {pattern_desc}. "
                f"ACF/PACF plots show {acf_desc}."
            )
            logging.info(f"Residual interpretation for {self.symbol}: {interpretation}")
            return interpretation
        except Exception as e:
            logging.error(f"Failed to generate residual interpretation for {self.symbol}: {str(e)}")
            return f"Error generating residual interpretation: {e}"

    def predict_future(self, order, steps=30):
        try:
            data_filled = self.data.copy()
            missing_count = data_filled['Close'].isna().sum()
            if missing_count > 0:
                st.warning(f"Found {missing_count} missing values in 'Close'. Interpolating missing data.")
                logging.warning(f"Found {missing_count} missing values in Close for {self.symbol}")
            data_filled['Close'] = data_filled['Close'].interpolate().ffill().bfill()
            if data_filled.index.tz is None:
                data_filled.index = data_filled.index.tz_localize('UTC')
            else:
                data_filled.index = data_filled.index.tz_convert('UTC')
            model = ARIMA(data_filled['Close'], order=order, trend='t')
            model_fit = model.fit()
            forecast_obj = model_fit.get_forecast(steps=steps)
            forecast = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            last_date = data_filled.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B', tz='UTC')
            from pandas.tseries.holiday import USFederalHolidayCalendar
            cal = USFederalHolidayCalendar()
            holidays = cal.holidays(start=forecast_dates[0], end=forecast_dates[-1] + pd.Timedelta(days=1))
            forecast_dates = forecast_dates[~forecast_dates.isin(holidays)]
            max_iterations = 100
            iteration = 0
            while len(forecast_dates) < steps and iteration < max_iterations:
                last_date = forecast_dates[-1]
                extra_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps - len(forecast_dates) + 1, freq='B', tz='UTC')
                extra_dates = extra_dates[~extra_dates.isin(holidays)]
                forecast_dates = forecast_dates.union(extra_dates)
                iteration += 1
            if len(forecast_dates) < steps:
                st.warning(f"Could only generate {len(forecast_dates)} forecast dates instead of {steps} due to holiday constraints.")
                logging.warning(f"Generated only {len(forecast_dates)} forecast dates for {self.symbol}")
            forecast_dates = forecast_dates[:steps]
            forecast_series = pd.Series(forecast.values, index=forecast_dates)
            lower_series = pd.Series(conf_int['lower Close'].values, index=forecast_dates)
            upper_series = pd.Series(conf_int['upper Close'].values, index=forecast_dates)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data_filled.index[-100:], data_filled['Close'][-100:], label='Historical', color='blue')
            ax.plot(forecast_series.index, forecast_series, label='Forecast', color='red', linestyle='--')
            ax.fill_between(forecast_dates, lower_series, upper_series, color='red', alpha=0.2, label='95% Confidence Interval')
            ax.set_title(f'{self.symbol} Price Forecast (ARIMA{order}, Steps={steps})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            logging.info(f"Generated forecast plot for ARIMA{order} with {steps} steps for {self.symbol}")
            return fig, forecast_series
        except Exception as e:
            st.error(f"Error making future predictions: {e}")
            logging.error(f"Forecast failed for ARIMA{order} for {self.symbol}: {str(e)}")
            return None, None

    def validate_model(self, train_data, validation_data, order, title):
        try:
            if train_data.index.tz is None:
                train_data.index = train_data.index.tz_localize('UTC')
            else:
                train_data.index = train_data.index.tz_convert('UTC')
            if validation_data.index.tz is None:
                validation_data.index = validation_data.index.tz_localize('UTC')
            else:
                validation_data.index = validation_data.index.tz_convert('UTC')
            model = ARIMA(train_data, order=order, trend='t')
            model_fit = model.fit()
            steps = len(validation_data)
            forecast_obj = model_fit.get_forecast(steps=steps)
            predictions = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            validation_dates = validation_data.index
            predicted_series = pd.Series(predictions.values, index=validation_dates)
            lower_series = pd.Series(conf_int['lower Close'].values, index=validation_dates)
            upper_series = pd.Series(conf_int['upper Close'].values, index=validation_dates)
            rmse = sqrt(mean_squared_error(validation_data, predictions))
            comparison_df = pd.DataFrame({
                'Date': validation_dates,
                'Predicted': predictions.values,
                'Expected': validation_data.values
            })
            comparison_df['Predicted'] = comparison_df['Predicted'].round(3)
            comparison_df['Expected'] = comparison_df['Expected'].round(3)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(validation_data.index, validation_data, label='Expected', color='blue')
            ax.plot(predicted_series.index, predicted_series, label='Predicted', color='red')
            ax.fill_between(validation_dates, lower_series, upper_series, color='red', alpha=0.2, label='95% Confidence Interval')
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            logging.info(f"Generated validation plot for ARIMA{order} for {self.symbol}")
            return fig, predictions, rmse, comparison_df
        except Exception as e:
            st.error(f"Error validating model: {e}")
            logging.error(f"Validation failed for ARIMA{order} for {self.symbol}: {str(e)}")
            return None, None, None, None

    def generate_boxplot_interpretation(self):
        try:
            grouped = self.data.groupby(self.data.index.year)['Close']
            years_data = {year: group.values for year, group in grouped}
            medians = [np.median(data) for data in years_data.values()]
            spreads = [np.std(data) for data in years_data.values()]
            if len(medians) > 1:
                median_change = (medians[-1] - medians[0]) / medians[0] * 100
                median_desc = "increases" if median_change > 5 else "decreases" if median_change < -5 else "remains stable"
            else:
                median_desc = "cannot be assessed (insufficient years)"
            spread_change = (spreads[-1] - spreads[0]) / spreads[0] * 100 if spreads[0] != 0 else 0
            spread_desc = "widening spread indicating growing volatility" if spread_change > 10 else "narrowing spread indicating decreasing volatility" if spread_change < -10 else "stable spread indicating consistent volatility"
            interpretation = f"The median price {median_desc} yearly, with {spread_desc}."
            logging.info(f"Boxplot interpretation for {self.symbol}: {interpretation}")
            return interpretation
        except Exception as e:
            logging.error(f"Failed to generate boxplot interpretation for {self.symbol}: {str(e)}")
            return f"Error generating boxplot interpretation: {e}"

# Streamlit app
st.title("Stock Market Analysis App")
st.write("Analyze stock prices with time series techniques (by Pardis Sh)")

# Debug: List files in directory
st.write("Files in directory:", os.listdir('.'))

# User inputs
stock_symbols = ['AAPL', 'META', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA']
symbol = st.selectbox("Select Stock Symbol", stock_symbols)

start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-30"))
forecast_steps = st.slider("Forecast Steps (Days)", min_value=5, max_value=60, value=30, step=5)

if st.button("Run Analysis"):
    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)
    min_date = pd.Timestamp("2019-01-01")
    max_date = pd.Timestamp("2025-04-30")

    if start_date_ts >= end_date_ts:
        st.error("Start date must be before end date.")
        logging.error("Invalid date range: start_date >= end_date")
    elif start_date_ts < min_date:
        st.error("Start date cannot be before 2019-01-01 due to historical data limitations.")
        logging.error("Start date before 2019-01-01")
    elif end_date_ts > max_date:
        st.error(f"End date cannot be after 2025-04-30 due to CSV data limitations.")
        logging.error(f"End date {end_date_ts.date()} exceeds CSV max date {max_date.date()}")
    else:
        try:
            analysis = StockMarketAnalysis(symbol, start_date, end_date)
            yearly_trading_days = analysis.calculate_yearly_trading_days()
            first_year = analysis.data.index.min().year
            period = yearly_trading_days.get(first_year, 252)

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Decomposition", "Visualizations", "Time Series Description", "Models", "Predictions"])

            with tab1:
                st.subheader("Time Series Decomposition")
                st.write("Multiplicative model used due to exponential growth in stock prices.")
                decomposition, fig = analysis.decompose_time_series(period=period)
                if fig:
                    st.pyplot(fig)
                    interpretation = analysis.generate_decomposition_interpretation(decomposition, period)
                    st.write(f"**Interpretation**: {interpretation}")
                else:
                    st.write("Decomposition not available due to insufficient data or computation error.")

            with tab2:
                st.subheader("Visualizations")
                st.write("**Summary Statistics**")
                st.write(analysis.summarize_data())

                st.write("**Closing Price Over Time**")
                fig = analysis.plot_time_series()
                if fig:
                    st.pyplot(fig)

                st.write("**Histogram and Kernel Density**")
                fig = analysis.plot_density()
                if fig:
                    st.pyplot(fig)

                st.write("**Yearly Box Plot**")
                fig = analysis.plot_yearly_boxplot()
                if fig:
                    st.pyplot(fig)
                    st.write(f"**Interpretation**: {analysis.generate_boxplot_interpretation()}")

                st.write("**Trading Volume**")
                fig = analysis.plot_volume()
                if fig:
                    st.pyplot(fig)

                st.write("**Daily Returns**")
                fig = analysis.plot_returns()
                if fig:
                    st.pyplot(fig)

                st.write("**Rolling Mean and Std (Window=60)**")
                fig = analysis.plot_rolling_mean_std(window=60)
                if fig:
                    st.pyplot(fig)

            with tab3:
                st.subheader("Time Series Description")
                desc = analysis.describe_time_series()
                st.write("**Summary Statistics**")
                st.write(desc['describe'])

                st.write("There is a noticeable trend in the data.")
                st.write("There is seasonality in the data.")
                st.write("The ACF/PACF plots of the differenced series guide my choice of p and q values (see below).")

                st.write("**Stationarity Analysis**")
                st.write("**Original Series ADF Test**")
                st.write(f"ADF Statistic: {desc['original_adf']['adf_statistic']:.3f}")
                st.write(f"p-value: {desc['original_adf']['p_value']:.3f}")
                st.write("**Critical Values**")
                for key, value in desc['original_adf']['critical_values'].items():
                    st.write(f"{key}: {value:.3f}")
                if desc['original_adf']['p_value'] <= 0.05:
                    st.write("The series is likely stationary.")
                else:
                    st.write("The series is likely non-stationary.")

                st.write("**1st-Order Differenced Series ADF Test**")
                st.write(f"ADF Statistic: {desc['differenced_adf']['adf_statistic']:.3f}")
                st.write(f"p-value: {desc['differenced_adf']['p_value']:.3f}")
                st.write("**Critical Values**")
                for key, value in desc['differenced_adf']['critical_values'].items():
                    st.write(f"{key}: {value:.3f}")
                if desc['differenced_adf']['p_value'] <= 0.05:
                    st.write("The 1st-order differenced series is likely stationary.")
                else:
                    st.write("The 1st-order differenced series is likely non-stationary.")

                st.write("**Stationarity Analysis**")
                d_level, diff_series = analysis.analyze_stationarity()
                if d_level is not None:
                    st.write(f"Recommended differencing level (d): {d_level}")
                    st.write("**ACF/PACF of Differenced Series**")
                    st.write(f"**Interpretation**: {analysis.generate_stationarity_interpretation(d_level, diff_series)}")
                    fig = analysis.plot_acf_pacf(diff_series.dropna())
                    if fig:
                        st.pyplot(fig)
                else:
                    st.write("Stationarity not achieved with differencing.")

            with tab4:
                st.subheader("Time Series Models")

                st.write("**ARIMA Model (Default: ARIMA(0,1,0))**")
                try:
                    # Preprocess data consistently
                    data_close = analysis.data['Close'].interpolate().ffill().bfill()
                    logging.info(f"Default ARIMA(0,1,0) data length: {len(data_close)}, missing values: {data_close.isna().sum()}")
                    arima_rmse = analysis.evaluate_arima_model(data_close, arima_order=(0, 1, 0))
                    st.write(f"ARIMA(0,1,0) RMSE: {arima_rmse:.6f}")
                    st.write("**Interpretation**: ARIMA(0,1,0) models stock prices as a random walk, assuming price changes are unpredictable. This aligns with the efficient market hypothesis for financial time series.")
                    st.write(f"Debug: Data length = {len(data_close)}, Train size = {int(len(data_close) * 0.8)}")
                except Exception as e:
                    st.error(f"ARIMA(0,1,0) failed: {e}")
                    logging.error(f"ARIMA(0,1,0) failed for {symbol}: {str(e)}")

                if st.session_state.grid_search_status == "not_started":
                    st.session_state.grid_search_status = "running"
                    st.session_state.grid_search_result = None
                    p_values = [0, 1]
                    d_values = [1]
                    q_values = [0, 1]
                    st.write(f"Running ARIMA grid search (p={p_values}, d={d_values}, q={q_values})...")
                    with st.spinner("Performing ARIMA grid search..."):
                        try:
                            best_order, best_score, results = analysis.grid_search_arima(p_values, d_values, q_values)
                            st.session_state.grid_search_result = {'order': best_order, 'score': best_score, 'results': results}
                            st.session_state.grid_search_status = "completed"
                        except Exception as e:
                            st.session_state.grid_search_status = "error"
                            st.session_state.grid_search_result = {'error': str(e)}
                            logging.error(f"Grid search failed for {symbol}: {str(e)}")

                if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result and 'order' in st.session_state.grid_search_result:
                    st.write("**Grid Search Results**")
                    results_df = pd.DataFrame([
                        {
                            'Order': f'ARIMA({r["p"]},{r["d"]},{r["q"]})',
                            'p': r['p'],
                            'd': r['d'],
                            'q': r['q'],
                            'RMSE': f"{r['rmse']:.6f}" if isinstance(r['rmse'], float) else r['rmse']
                        }
                        for r in st.session_state.grid_search_result['results']
                    ])
                    st.table(results_df)
                    st.write(f"**Best ARIMA order**: {st.session_state.grid_search_result['order']}")
                    st.write(f"**Best RMSE**: {st.session_state.grid_search_result['score']:.6f}")
                    st.write("**Model Selection Discussion**:")
                    best_order = st.session_state.grid_search_result['order']
                    best_rmse = st.session_state.grid_search_result['score']
                    other_rmses = [r['rmse'] for r in st.session_state.grid_search_result['results'] if isinstance(r['rmse'], float) and r['rmse'] != best_rmse]
                    other_rmse_example = min(other_rmses, default=best_rmse) if other_rmses else best_rmse
                    st.write(f"The best model is ARIMA{best_order}, with an RMSE of {best_rmse:.6f}. This model suggests stock prices follow a pattern best captured by these parameters, potentially indicating a random walk if p=0, q=0.")
                    st.write(f"The RMSE values are close (e.g., {best_rmse:.6f} vs. {other_rmse_example:.6f} for other orders), so ARIMA{best_order} is selected for its balance of fit and simplicity, adhering to the principle of parsimony.")
                    st.write("ARIMA was chosen because it is well-established for time series forecasting, and the decomposition showed weak seasonality. Machine learning models like Prophet could be explored in future work.")

                st.write("**SARIMA Model**")
                with st.spinner("Evaluating SARIMA model (this may take a moment)..."):
                    try:
                        best_order = st.session_state.grid_search_result['order'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else (0, 1, 0)
                        sarima_rmse = analysis.evaluate_sarima_model(
                            analysis.data['Close'],
                            order=best_order,
                            seasonal_order=(1, 0, 1, 63),
                            _period=252
                        )
                        if np.isinf(sarima_rmse):
                            st.error(f"SARIMA{best_order}(1,0,1)[63] failed to fit the data.")
                            logging.error(f"SARIMA{best_order}(1,0,1)[63] returned inf RMSE for {symbol}")
                        else:
                            st.write(f"SARIMA{best_order}(1,0,1)[63] RMSE: {sarima_rmse:.6f}")
                            st.write(f"**Interpretation**: SARIMA extends ARIMA with seasonal components. A quarterly period (63 trading days) captures potential seasonal patterns. A lower RMSE suggests a good fit.")
                    except Exception as e:
                        st.error(f"SARIMA failed: {e}")
                        logging.error(f"SARIMA failed for {symbol}: {str(e)}")

                st.write("**TSLM Model**")
                with st.spinner("Evaluating TSLM model..."):
                    try:
                        tslm_rmse, _ = analysis.evaluate_tslm(analysis.data['Close'], seasonal_period=63)
                        st.write(f"TSLM (trend + 63 seasonal dummies) RMSE: {tslm_rmse:.6f}")
                        st.write(f"**Interpretation**: TSLM uses a linear model with trend and seasonal dummies, capturing long-term growth and quarterly cycles (63 trading days). It’s simpler but may miss complex dynamics.")
                    except Exception as e:
                        st.error(f"TSLM failed: {e}")
                        logging.error(f"TSLM failed for {symbol}: {str(e)}")

                st.write("**Residual Diagnostics**")
                try:
                    best_order = st.session_state.grid_search_result['order'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else (0, 1, 0)
                    st.write(f"Diagnostics for ARIMA{best_order}")
                    fig, residuals = analysis.plot_model_residuals(order=best_order)
                    if fig:
                        st.pyplot(fig)
                        residuals_clean = residuals.dropna()
                        if residuals_clean.empty:
                            st.error("Residuals are empty after removing NaN values, cannot proceed with diagnostics.")
                            logging.error(f"Empty residuals after cleaning for {symbol}")
                        else:
                            st.write("**Residual Statistics**")
                            st.write(residuals_clean.describe())
                            st.write(f"**Interpretation**: {analysis.generate_residual_interpretation(residuals_clean)}")
                            lb_test = acorr_ljungbox(residuals_clean, lags=[20], return_df=True)
                            st.write("**Ljung-Box Test for Residual Autocorrelation (lag=20)**")
                            st.write(lb_test)
                            st.write("**Interpretation**: A high p-value (>0.05) in the Ljung-Box test suggests no significant autocorrelation in residuals, confirming a good model fit.")
                    else:
                        st.error("Failed to generate residual plots.")
                except Exception as e:
                    st.error(f"Residual diagnostics failed: {e}")
                    logging.error(f"Residual diagnostics failed for {symbol}: {str(e)}")

                st.write("**Model Comparison**")
                try:
                    best_order = st.session_state.grid_search_result['order'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else (0, 1, 0)
                    best_arima_rmse = st.session_state.grid_search_result['score'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else arima_rmse
                    model_comparison = {
                        "Model": [f"ARIMA{best_order}", f"SARIMA{best_order}(1,0,1)[63]", "TSLM"],
                        "RMSE": [
                            best_arima_rmse if not np.isinf(best_arima_rmse) else "Failed",
                            sarima_rmse if not np.isinf(sarima_rmse) else "Failed",
                            tslm_rmse if not np.isinf(tslm_rmse) else "Failed"
                        ]
                    }
                    comparison_df = pd.DataFrame(model_comparison)
                    comparison_df['RMSE'] = comparison_df['RMSE'].apply(lambda x: f"{x:.6f}" if isinstance(x, float) else x)
                    st.write(comparison_df)
                except Exception as e:
                    st.error(f"Model comparison failed: {e}")
                    logging.error(f"Model comparison failed for {symbol}: {str(e)}")

            with tab5:
                st.subheader("Predictions and Validation")
                st.write(f"**{forecast_steps}-Day Forecast (ARIMA)**")
                try:
                    best_order = st.session_state.grid_search_result['order'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else (0, 1, 0)
                    fig, forecast = analysis.predict_future(order=best_order, steps=forecast_steps)
                    if fig:
                        st.pyplot(fig)
                        st.write("**Forecasted Prices**")
                        st.write(forecast)
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    logging.error(f"Forecast failed for {symbol}: {str(e)}")

                st.write("**Model Validation (Last Year)**")
                try:
                    validation_data = analysis.data['Close'][-period:]
                    train_data = analysis.data['Close'][:-period]
                    fig, predictions, rmse, comparison_df = analysis.validate_model(train_data, validation_data, best_order, f"Validation: Expected vs Predicted {symbol} Closing Prices (Last {period} Trading Days)")
                    if fig:
                        st.pyplot(fig)
                        st.write("**Predicted vs Expected Values**")
                        st.write(comparison_df)
                        st.write(f"Validation RMSE: {rmse:.6f}")
                except Exception as e:
                    st.error(f"Validation failed: {e}")
                    logging.error(f"Validation failed for {symbol}: {str(e)}")

        except Exception as e:
            st.error(f"Error: {e}")
            logging.error(f"Application error for {symbol}: {str(e)}")

st.markdown("---")
st.write("**Team Contributions**: Pardis Sh")
st.write("Tasks and Time Spent:")
st.write("- Data description: 7 hours")
st.write("- Data collection, cleaning, processing, visualization, model design, time series analysis, evaluation: 57 hours")
st.write("- Final report and video presentation: 24 hours")
st.write("Total Time Spent: 88 hours")
st.write("**Resources**:")
st.write("- statsmodels documentation: https://www.statsmodels.org/stable/index.html")
st.write("- Pandas documentation: https://pandas.pydata.org/docs/")
st.write("- Matplotlib documentation: https://matplotlib.org/stable/contents.html")
st.write("All code was written by Pardis Sh, using the above resources for reference.")