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
from statsmodels.tsa.tsatools import add_trend
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf as compute_acf, pacf as compute_pacf
from math import sqrt
from yfinance import Ticker
import time
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import time


# Initialize session state
if 'grid_search_result' not in st.session_state:
    st.session_state.grid_search_result = None
if 'grid_search_status' not in st.session_state:
    st.session_state.grid_search_status = "not_started"

# StockMarketAnalysis class
class StockMarketAnalysis:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_data()


    def load_data(self):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                stock = yf.Ticker(self.symbol, session=None)  # Explicitly pass session=None to avoid internal issues
                data = stock.history(start=self.start_date, end=self.end_date)
                if data.empty:
                    raise ValueError(f"No data found for symbol {self.symbol} between {self.start_date} and {self.end_date}")
                return data
            except Exception as e:
                if "Too Many Requests" in str(e) and attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise ValueError(f"Failed to fetch data for {self.symbol}: {str(e)}")

    def calculate_yearly_trading_days(self):
        if isinstance(self.data.index, pd.DatetimeIndex):
            yearly_trading_days = {}
            start_year = self.data.index.min().year
            end_year = self.data.index.max().year
            tz = self.data.index.tz
            for year in range(start_year, end_year + 1):
                start_date = pd.to_datetime(f'{year}-01-01').tz_localize(tz) if tz else pd.to_datetime(f'{year}-01-01')
                end_date = pd.to_datetime(f'{year}-12-31').tz_localize(tz) if tz else pd.to_datetime(f'{year}-12-31')
                year_data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
                yearly_trading_days[year] = len(year_data)
            return yearly_trading_days
        return None

    '''
    def decompose_time_series(self, model='multiplicative', period=None, plot=True):
        if period is None:
            st.error("Period not specified for decomposition.")
            return None
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
                return result, fig
            return result, None
        except Exception as e:
            st.error(f"Error during decomposition: {e}")
            return None, None
    '''

    def decompose_time_series(self, model='multiplicative', period=None, plot=True):
        if period is None:
            st.error("Period not specified for decomposition.")
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
                return result, fig
            return result, None
        except Exception as e:
            st.error(f"Error during decomposition: {e}")
            return None, None
        
            
    def generate_decomposition_interpretation(self, decomposition, period):
        if decomposition is None:
            return "Unable to generate interpretation due to decomposition failure."
        
        try:
            # Analyze Trend
            # In generate_decomposition_interpretation
            trend = decomposition.trend.dropna()
            if len(trend) > 1:
                trend_change = (trend.iloc[-1] - trend.iloc[0]) / trend.iloc[0] * 100
                # Check for significant mid-period dip
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

            # Analyze Seasonality
            seasonal = decomposition.seasonal.dropna()
            seasonal_amplitude = seasonal.max() - seasonal.min()
            seasonal_std = self.data['Close'].std()
            if seasonal_amplitude / seasonal_std > 0.1:
                seasonal_desc = f"significant yearly patterns with a period of {period} trading days"
            else:
                seasonal_desc = f"subtle yearly patterns with a period of {period} trading days, suggesting seasonality is not dominant"

            # Analyze Residuals
            resid = decomposition.resid.dropna()
            resid_std = resid.std()
            if resid_std / self.data['Close'].std() > 0.5:
                resid_desc = "highly noisy residuals, indicating significant random fluctuations or external events"
            else:
                resid_desc = "moderately noisy residuals, suggesting minor random fluctuations"

            # Combine into interpretation
            interpretation = (
                f"The trend component shows {trend_desc} (change: {trend_change:.2f}%). "
                f"The seasonal component exhibits {seasonal_desc} (amplitude: {seasonal_amplitude:.2f}). "
                f"The residuals are {resid_desc} (std: {resid_std:.2f})."
             
                )
            return interpretation
        except Exception as e:
            return f"Error generating interpretation: {e}"
        

    def summarize_data(self):
        return self.data['Close'].describe()

    def plot_time_series(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.data.index, self.data['Close'], label='Closing Price')
        ax.set_title(f'{self.symbol} Closing Price Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
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
            return fig
        return None

    def plot_volume(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.data.index, self.data['Volume'], label='Volume', color='orange')
        ax.set_title(f'{self.symbol} Trading Volume Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.legend()
        ax.grid(True)
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
        return fig

    def plot_rolling_mean_std(self, window=30):
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
        return fig

    def describe_time_series(self):
        # Original series ADF test
        result = adfuller(self.data['Close'])
        original_adf = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
        # 1st-order differenced series ADF test
        diff_series = self.create_differenced_series(self.data['Close'])
        result_diff = adfuller(diff_series.dropna())
        differenced_adf = {
            'adf_statistic': result_diff[0],
            'p_value': result_diff[1],
            'critical_values': result_diff[4]
        }
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
        return None

    def create_differenced_series(self, data, interval=1):
        diff = [data[i] - data[i - interval] for i in range(interval, len(data))]
        return pd.Series(diff, index=data.index[interval:])

    def plot_acf_pacf(self, series, lags=20):
        plt.figure(figsize=(12, 5))  # Create a new figure
        plt.subplot(1, 2, 1)
        plot_acf(series, lags=lags, ax=plt.gca())
        plt.title('ACF')
        plt.subplot(1, 2, 2)
        plot_pacf(series, lags=lags, ax=plt.gca())
        plt.title('PACF')
        plt.tight_layout()
        return plt.gcf()

    def analyze_stationarity(self):
        d_level = self.check_stationarity(self.data['Close'])
        if d_level is not None:
            differenced_close = self.data['Close']
            for _ in range(d_level):
                differenced_close = self.create_differenced_series(differenced_close)
            return d_level, differenced_close
        return None, None
        
        

    def generate_stationarity_interpretation(self, d_level, diff_series):
        try:
            if d_level is None:
                return "Stationarity not achieved with differencing, making ARIMA parameter estimation challenging."
            
            # Analyze ACF/PACF for p, q suggestions
            acf_vals = compute_acf(diff_series.dropna(), nlags=20, fft=False)
            pacf_vals = compute_pacf(diff_series.dropna(), nlags=20)
            significant_lags_acf = sum(abs(acf_vals[1:]) > 0.1)  # Exclude lag 0
            significant_lags_pacf = sum(abs(pacf_vals[1:]) > 0.1)  # Exclude lag 0
            
            q_suggestion = "q=0" if significant_lags_acf <= 1 else f"q={significant_lags_acf}"
            p_suggestion = "p=0" if significant_lags_pacf <= 1 else f"p={significant_lags_pacf}"
            
            model_desc = "aligning with a random walk model" if p_suggestion == "p=0" and q_suggestion == "q=0" else "suggesting a more complex ARIMA model"
            
            return (
                f"The ACF shows {significant_lags_acf} significant lag(s), suggesting {q_suggestion}. "
                f"The PACF shows {significant_lags_pacf} significant lag(s), suggesting {p_suggestion}, "
                f"{model_desc}."
            )
        except Exception as e:
            return f"Error generating stationarity interpretation: {e}"

    def evaluate_arima_model(self, series, arima_order):
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        history = [x for x in train]
        predictions = []
        for t in range(len(test)):
            try:
                model = ARIMA(history, order=arima_order)
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
                predictions.append(yhat)
                history.append(test[t])
            except:
                return float("inf")  # Return high RMSE on failure
        rmse = sqrt(mean_squared_error(test, predictions))
        return rmse

    def evaluate_sarima_model(self, series, order, seasonal_order):
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(start=train_size, end=len(series)-1, dynamic=True)
            rmse = sqrt(mean_squared_error(test, predictions))
            return rmse
        except:
            return float("inf")

    def evaluate_tslm(self, series, seasonal_period):
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        train_df = pd.DataFrame({'Close': train})
        train_df['t'] = range(1, len(train) + 1)
        train_df['season'] = pd.Series(train.index).apply(lambda x: (x.dayofyear % seasonal_period) + 1).astype(str)
        train_df = pd.get_dummies(train_df, columns=['season'], drop_first=True)
        predictors = ['t'] + [col for col in train_df.columns if col.startswith('season_')]
        try:
            model = OLS(train_df['Close'], train_df[predictors])
            model_fit = model.fit()
            test_df = pd.DataFrame({'Close': test})
            test_df['t'] = range(len(train) + 1, len(train) + len(test) + 1)
            test_df['season'] = pd.Series(test.index).apply(lambda x: (x.dayofyear % seasonal_period) + 1).astype(str)
            test_df = pd.get_dummies(test_df, columns=['season'], drop_first=True)
            for col in predictors:
                if col not in test_df.columns:
                    test_df[col] = 0
            predictions = model_fit.predict(test_df[predictors])
            rmse = sqrt(mean_squared_error(test, predictions))
            return rmse, model_fit
        except:
            return float("inf"), None
    '''
    def grid_search_arima(self, p_values, d_values, q_values):
        best_score, best_order = float("inf"), None
        total_iterations = len(p_values) * len(d_values) * len(q_values)
        progress = st.progress(0)
        iteration = 0
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        rmse = self.evaluate_arima_model(self.data['Close'], order)
                        if rmse < best_score:
                            best_score, best_order = rmse, order
                        st.write(f'ARIMA{order} RMSE={rmse:.3f}')
                    except Exception as e:
                        st.warning(f"Skipping order {order}: {e}")
                        continue
                    iteration += 1
                    progress.progress(iteration / total_iterations)
        progress.empty()
        return best_order, best_score
    '''
    def grid_search_arima(self, p_values, d_values, q_values):
        best_score, best_order = float("inf"), None
        results = []  # Store all order and RMSE pairs
        total_iterations = len(p_values) * len(d_values) * len(q_values)
        progress = st.progress(0)
        iteration = 0
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        rmse = self.evaluate_arima_model(self.data['Close'], order)
                        results.append({'order': order, 'p': p, 'd': d, 'q': q, 'rmse': rmse})  # Store p, d, q
                        if rmse < best_score:
                            best_score, best_order = rmse, order
                        st.write(f'ARIMA({p},{d},{q}) RMSE={rmse:.3f}')
                    except Exception as e:
                        st.warning(f"Skipping ARIMA({p},{d},{q}): {e}")
                        results.append({'order': order, 'p': p, 'd': d, 'q': q, 'rmse': 'Failed'})  # Log failed attempts
                        continue
                    iteration += 1
                    progress.progress(iteration / total_iterations)
        progress.empty()
        return best_order, best_score, results  # Return results list
    
    
    def plot_model_residuals(self, order):
        try:
            train_size = int(len(self.data['Close']) * 0.8)
            train = self.data['Close'][:train_size]
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            residuals = pd.DataFrame(model_fit.resid)
            # Debug: Check for NaN values in residuals
            if residuals.isna().any().any():
                st.warning(f"Residuals contain {residuals.isna().sum().sum()} NaN values, which may affect analysis.")
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
            return fig, residuals
        except Exception as e:
            st.error(f"Error plotting residuals for order {order}: {e}")
            return None, None 
            
    def generate_residual_interpretation(self, residuals):
        try:
            if residuals is None or residuals.empty:
                return "Unable to generate residual interpretation due to model failure."
            
            # Extract the residuals series (it's a DataFrame with one column)
            residuals_series = residuals.iloc[:, 0] if isinstance(residuals, pd.DataFrame) else residuals
            
            # Clean residuals by removing NaN values
            residuals_series = residuals_series.dropna()
            if residuals_series.empty:
                return "Unable to generate residual interpretation due to insufficient data after removing NaN values."
            
            # Check mean for centering
            mean_residual = residuals_series.mean()
            centering_desc = "centered around zero" if abs(mean_residual) < 0.1 * residuals_series.std() else "not centered around zero, suggesting potential model bias"
            
            # Check for patterns in time series
            resid_std = residuals_series.std()
            pattern_desc = "random with no clear patterns" if resid_std / self.data['Close'].std() < 0.5 else "some patterns, indicating possible unmodeled structure"
            
            # Check ACF/PACF for significant lags
            acf_vals = compute_acf(residuals_series, nlags=20, fft=False)
            significant_lags = sum(abs(acf_vals[1:]) > 0.1)  # Exclude lag 0
            acf_desc = "insignificant lags, indicating the model captures the data’s structure well" if significant_lags <= 1 else f"{significant_lags} significant lags, suggesting some residual autocorrelation"
            
            return (
                f"The histogram and KDE show residuals are {centering_desc}. "
                f"The time series plot of residuals appears {pattern_desc}. "
                f"ACF/PACF plots show {acf_desc}."
            )
        except Exception as e:
            return f"Error generating residual interpretation: {e}"
        

    def predict_future(self, order, steps=30):
        try:
            data_filled = self.data.copy()
            data_filled['Close'] = data_filled['Close'].interpolate().ffill().bfill()
            data_filled.index = data_filled.index.tz_localize(None)
            model = ARIMA(data_filled['Close'], order=order, trend='t')
            model_fit = model.fit()
            forecast_obj = model_fit.get_forecast(steps=steps)
            forecast = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            last_date = data_filled.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq='B')[1:]
            from pandas.tseries.holiday import USFederalHolidayCalendar
            cal = USFederalHolidayCalendar()
            holidays = cal.holidays(start='2024-01-01', end='2024-05-31')
            forecast_dates = forecast_dates[~forecast_dates.isin(holidays)]
            while len(forecast_dates) < steps:
                last_date = forecast_dates[-1]
                extra_dates = pd.date_range(start=last_date, periods=steps - len(forecast_dates) + 1, freq='B')[1:]
                extra_dates = extra_dates[~extra_dates.isin(holidays)]
                forecast_dates = forecast_dates.union(extra_dates)
            forecast_dates = forecast_dates[:steps].tz_localize(None)
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
            return fig, forecast_series
        except Exception as e:
            st.error(f"Error making future predictions: {e}")
            return None, None

    def validate_model(self, train_data, validation_data, order, title):
        try:
            model = ARIMA(train_data, order=order, trend='t')
            model_fit = model.fit()
            steps = len(validation_data)
            forecast_obj = model_fit.get_forecast(steps=steps)
            predictions = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            validation_dates = validation_data.index.tz_localize(None)
            predicted_series = pd.Series(predictions.values, index=validation_dates)
            lower_series = pd.Series(conf_int['lower Close'].values, index=validation_dates)
            upper_series = pd.Series(conf_int['upper Close'].values, index=validation_dates)
            rmse = sqrt(mean_squared_error(validation_data, predictions))
            
            # Create a DataFrame for Predicted vs Expected values
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
            return fig, predictions, rmse, comparison_df
        except Exception as e:
            st.error(f"Error validating model: {e}")
            return None, None, None, None 
                
            
            
    def generate_boxplot_interpretation(self):
        try:
            grouped = self.data.groupby(self.data.index.year)['Close']
            years_data = {year: group.values for year, group in grouped}
            medians = [np.median(data) for data in years_data.values()]
            spreads = [np.std(data) for data in years_data.values()]
            
            # Analyze median trend
            if len(medians) > 1:
                median_change = (medians[-1] - medians[0]) / medians[0] * 100
                median_desc = "increases" if median_change > 5 else "decreases" if median_change < -5 else "remains stable"
            else:
                median_desc = "cannot be assessed (insufficient years)"

            # Analyze spread
            spread_change = (spreads[-1] - spreads[0]) / spreads[0] * 100 if spreads[0] != 0 else 0
            spread_desc = "widening spread indicating growing volatility" if spread_change > 10 else "narrowing spread indicating decreasing volatility" if spread_change < -10 else "stable spread indicating consistent volatility"

            return f"The median price {median_desc} yearly, with {spread_desc}."
        except Exception as e:
            return f"Error generating boxplot interpretation: {e}"
        

# Streamlit app
st.title("Stock Market Analysis App")
st.write("Analyze stock prices with time series techniques (by Pardis Sh)")

# User inputs
stock_symbols =['AAPL', 'META', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA'] #["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "JPM", "V", "WMT"]
symbol = st.selectbox("Select Stock Symbol", stock_symbols)

start_date = st.date_input("Start Date", value=pd.to_datetime("2023-05-01"))
#end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-05-01"))
forecast_steps = st.slider("Forecast Steps (Days)", min_value=5, max_value=60, value=30, step=5)

if st.button("Run Analysis"):
    try:
        # Initialize analysis
        analysis = StockMarketAnalysis(symbol, start_date, end_date)
        yearly_trading_days = analysis.calculate_yearly_trading_days()
        first_year = analysis.data.index.min().year
        period = yearly_trading_days.get(first_year, 252)

        # Tabs for organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Decomposition", "Visualizations", "Time Series Description", "Models", "Predictions"])

        with tab1:
            st.subheader("Time Series Decomposition")
            st.write("Multiplicative model used due to exponential growth in stock prices.")
            decomposition, fig = analysis.decompose_time_series(period=period)
            if fig:
                st.pyplot(fig)
                interpretation = analysis.generate_decomposition_interpretation(decomposition, period)
                st.write(f"**Interpretation**: {interpretation}")
                
        with tab2:
            st.subheader("Visualizations")
            st.write("**Summary Statistics**")
            st.write(analysis.summarize_data())

            st.write("**Closing Price Over Time**")
            st.pyplot(analysis.plot_time_series())

            st.write("**Histogram and Kernel Density**")
            st.pyplot(analysis.plot_density())

            st.write("**Yearly Box Plot**")
            fig = analysis.plot_yearly_boxplot()
            if fig:
                st.pyplot(fig)
                st.write(f"**Interpretation**: {analysis.generate_boxplot_interpretation()}")

            st.write("**Trading Volume**")
            st.pyplot(analysis.plot_volume())

            st.write("**Daily Returns**")
            st.pyplot(analysis.plot_returns())

            st.write("**Rolling Mean and Std (Window=60)**")
            st.pyplot(analysis.plot_rolling_mean_std(window=60))

        with tab3:
            st.subheader("Time Series Description")
            desc = analysis.describe_time_series()
            st.write("**Summary Statistics**")
            st.write(desc['describe'])
            
            # Statements about trend and seasonality (based on decomposition)
            st.write("There is a noticeable trend in the data.")
            st.write("There is seasonality in the data.")
            st.write("The ACF/PACF plots of the differenced series guide my choice of p and q values (see below).")
            
            # Stationarity Analysis
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
                st.pyplot(analysis.plot_acf_pacf(diff_series.dropna()))
            else:
                st.write("Stationarity not achieved with differencing.")

        with tab4:
            st.subheader("Time Series Models")
            
            # Default ARIMA Model
            st.write("**ARIMA Model (Default: ARIMA(0,1,0))**")
            try:
                arima_rmse = analysis.evaluate_arima_model(analysis.data['Close'], arima_order=(0, 1, 0))
                st.write(f"ARIMA(0,1,0) RMSE: {arima_rmse:.3f}")
                st.write("**Interpretation**: ARIMA(0,1,0) models stock prices as a random walk, assuming price changes are unpredictable. This aligns with the efficient market hypothesis for financial time series.")
            except Exception as e:
                st.error(f"ARIMA(0,1,0) failed: {e}")

            # ARIMA Grid Search (Automatic Execution)
            if st.session_state.grid_search_status == "not_started":
                st.session_state.grid_search_status = "running"
                st.session_state.grid_search_result = None
                # Define grid search parameters
                p_values = [0, 1]
                d_values = [1]
                q_values = [0, 1]
                # Dynamically generate message
                st.write(f"Running ARIMA grid search (p={p_values}, d={d_values}, q={q_values})...")
                try:
                    import threading
                    result = [None]
                    def run_grid_search():
                        best_order, best_score, results = analysis.grid_search_arima(p_values, d_values, q_values)
                        result[0] = {'order': best_order, 'score': best_score, 'results': results}
                    thread = threading.Thread(target=run_grid_search)
                    thread.start()
                    thread.join(timeout=60)
                    if thread.is_alive():
                        st.session_state.grid_search_status = "error"
                        st.session_state.grid_search_result = {'error': "Grid search timed out after 60 seconds."}
                    else:
                        st.session_state.grid_search_result = result[0]
                        st.session_state.grid_search_status = "completed"
                except Exception as e:
                    st.session_state.grid_search_status = "error"
                    st.session_state.grid_search_result = {'error': str(e)}

            # Display grid search results
            if st.session_state.grid_search_status == "running":
                st.write("Grid search in progress... Please wait (up to 60 seconds).")
            elif st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result and 'order' in st.session_state.grid_search_result:
                st.write("**Grid Search Results**")
                # Display all results in a table
                results_df = pd.DataFrame([
                    {
                        'Order': f'ARIMA({r["p"]},{r["d"]},{r["q"]})',
                        'p': r['p'],
                        'd': r['d'],
                        'q': r['q'],
                        'RMSE': f"{r['rmse']:.3f}" if isinstance(r['rmse'], float) else r['rmse']
                    }
                    for r in st.session_state.grid_search_result['results']
                ])
                st.table(results_df)
                st.write(f"**Best ARIMA order**: {st.session_state.grid_search_result['order']}")
                st.write(f"**Best RMSE**: {st.session_state.grid_search_result['score']:.3f}")
                st.write("**Model Selection Discussion**:")
                best_order = st.session_state.grid_search_result['order']
                best_rmse = st.session_state.grid_search_result['score']
                other_rmses = [r['rmse'] for r in st.session_state.grid_search_result['results'] if isinstance(r['rmse'], float) and r['rmse'] != best_rmse]
                other_rmse_example = min(other_rmses, default=best_rmse) if other_rmses else best_rmse
                st.write(f"The best model is ARIMA{best_order}, with an RMSE of {best_rmse:.3f}. This model suggests stock prices follow a pattern best captured by these parameters, potentially indicating a random walk if p=0, q=0.")
                st.write(f"The RMSE values are close (e.g., {best_rmse:.3f} vs. {other_rmse_example:.3f} for other orders), so ARIMA{best_order} is selected for its balance of fit and simplicity, adhering to the principle of parsimony.")
                st.write("ARIMA was chosen because it is well-established for time series forecasting, and the decomposition showed weak seasonality. Machine learning models like Prophet could be explored in future work.")
            elif st.session_state.grid_search_status == "error" and st.session_state.grid_search_result:
                st.error(f"Grid search failed: {st.session_state.grid_search_result['error']}")

            # SARIMA Model
            st.write("**SARIMA Model**")
            try:
                best_order = st.session_state.grid_search_result['order'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else (0, 1, 0)
                sarima_rmse = analysis.evaluate_sarima_model(analysis.data['Close'], order=best_order, seasonal_order=(1, 0, 1, period))
                st.write(f"SARIMA{best_order}(1,0,1)[{period}] RMSE: {sarima_rmse:.3f}")
                st.write(f"**Interpretation**: SARIMA extends ARIMA with seasonal components, modeling yearly patterns ({period} trading days). A lower RMSE suggests better fit for seasonal trends.")
            except Exception as e:
                st.error(f"SARIMA failed: {e}")

            # TSLM Model
            st.write("**TSLM Model**")
            try:
                tslm_rmse, _ = analysis.evaluate_tslm(analysis.data['Close'], seasonal_period=period)
                st.write(f"TSLM (trend + {period} seasonal dummies) RMSE: {tslm_rmse:.3f}")
                st.write(f"**Interpretation**: TSLM uses a linear model with trend and seasonal dummies, capturing long-term growth and yearly cycles ({period} trading days). It’s simpler but may miss complex dynamics.")
            except Exception as e:
                st.error(f"TSLM failed: {e}")

            # Residual Diagnostics
            st.write("**Residual Diagnostics**")
            try:
                best_order = st.session_state.grid_search_result['order'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else (0, 1, 0)
                st.write(f"Diagnostics for ARIMA{best_order}")
                fig, residuals = analysis.plot_model_residuals(order=best_order)
                if fig:
                    st.pyplot(fig)
                    # Clean residuals for further analysis
                    residuals_clean = residuals.dropna()
                    if residuals_clean.empty:
                        st.error("Residuals are empty after removing NaN values, cannot proceed with diagnostics.")
                    else:
                        st.write("**Residual Statistics**")
                        st.write(residuals_clean.describe())
                        st.write(f"**Interpretation**: {analysis.generate_residual_interpretation(residuals_clean)}")
                        lb_test = acorr_ljungbox(residuals_clean, lags=[20], return_df=True)
                        st.write("**Ljung-Box Test for Residual Autocorrelation (lag=20)**")
                        st.write(lb_test)
                        st.write("**Interpretation**: A high p-value (>0.05) in the Ljung-Box test suggests no significant autocorrelation in residuals, confirming a good model fit.")
            except Exception as e:
                st.error(f"Residual diagnostics failed: {e}") 


            # Model Comparison
            st.write("**Model Comparison**")
            try:
                best_order = st.session_state.grid_search_result['order'] if st.session_state.grid_search_status == "completed" and st.session_state.grid_search_result else (0, 1, 0)
                model_comparison = {
                    "Model": [f"ARIMA{best_order}", f"SARIMA{best_order}(1,0,1)[{period}]", "TSLM"],
                    "RMSE": [arima_rmse, sarima_rmse, tslm_rmse]
                }
                st.write(pd.DataFrame(model_comparison))
                st.write("**Interpretation**: Compare RMSE values to select the best model. Lower RMSE indicates better predictive accuracy. SARIMA often outperforms due to seasonal modeling, but ARIMA is simpler and robust for non-seasonal trends.")
            except Exception as e:
                st.error(f"Model comparison failed: {e}")
                    
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

            st.write("**Model Validation (Last Year)**")
            try:
                validation_data = analysis.data['Close'][-period:]
                train_data = analysis.data['Close'][:-period]
                fig, predictions, rmse, comparison_df = analysis.validate_model(train_data, validation_data, best_order, f"Validation: Expected vs Predicted {symbol} Closing Prices (Last {period} Trading Days)")
                if fig:
                    st.pyplot(fig)
                    st.write("**Predicted vs Expected Values**")
                    st.dataframe(comparison_df)  # Display the table
                    st.write(f"Validation RMSE: {rmse:.3f}")
            except Exception as e:
                st.error(f"Validation failed: {e}")

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.write("**Team Contributions**: Pardis Sh")
st.write("Tasks and Time Spent:")
st.write("- Data description: 7 hours")
st.write("- Data collection, cleaning, processing, visualization, model design, time series analysis, evaluation: 57 hours")
st.write("- Final report and video presentation: 24 hours")
st.write("Total Time Spent: 88 hours")
st.write("**Resources**:")
st.write("- yfinance library documentation: https://pypi.org/project/yfinance/")
st.write("- statsmodels documentation: https://www.statsmodels.org/stable/index.html")
st.write("- Pandas documentation: https://pandas.pydata.org/docs/")
st.write("- Matplotlib documentation: https://matplotlib.org/stable/contents.html")
st.write("All code was written by Pardis Sh, using the above resources for reference.")