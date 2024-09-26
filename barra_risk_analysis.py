# -*- coding = utf-8 -*-
# @Time: 2024/09/26
# @Author: Xinyue
# @File: barra_risk_analysis.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


from Utils.get_wind_data import WindData

class PortfolioRiskAttribution:
    def __init__(self, start_date, end_date, factor_returns_by_day, benchmark=True, custom_portfolio=None):
        self.start_date = start_date
        self.end_date = end_date
        self.factor_returns_by_day = factor_returns_by_day
        self.benchmark = benchmark
        self.custom_portfolio = custom_portfolio
        self.benchmark_df = None
        self.regression_df = None
        self.factor_columns = factor_returns_by_day.columns.drop('TRADE_DT').tolist()

        self._prepare_data()

    def _prepare_data(self):
        # Prepare the benchmark data if benchmark is True
        if self.benchmark:
            wind = WindData(start_date=self.start_date, end_date=self.end_date)
            self.benchmark_df = wind.get_index_price(index_code='000985.CSI')
            self.benchmark_df['TRADE_DT'] = pd.to_datetime(self.benchmark_df['TRADE_DT'])
            self.benchmark_df['RETURN_BENCHMARK'] = self.benchmark_df['S_DQ_CLOSE'].astype('float') / \
                                                    self.benchmark_df['S_DQ_PRECLOSE'].astype('float') - 1

            self.benchmark_df['LOG_RETURN_BENCHMARK'] = np.log(self.benchmark_df['S_DQ_CLOSE'].astype('float')/\
                                                               self.benchmark_df['S_DQ_PRECLOSE'].astype('float'))
            #self.benchmark_df = self.benchmark_df[['TRADE_DT', 'LOG_RETURN_BENCHMARK']]

        # Prepare the custom portfolio data
        if self.custom_portfolio is not None:
            self.custom_portfolio['RETURN_PORTFOLIO'] = self.custom_portfolio['S_DQ_CLOSE'].astype('float') /\
                                                        self.custom_portfolio['S_DQ_PRECLOSE'].astype('float') - 1

            self.custom_portfolio['LOG_RETURN_PORTFOLIO'] = np.log(self.custom_portfolio['S_DQ_CLOSE'].astype('float')/\
                                                                   self.custom_portfolio['S_DQ_PRECLOSE'].astype('float'))
            #self.custom_portfolio = self.custom_portfolio[['TRADE_DT', 'LOG_RETURN_PORTFOLIO']]

        # Combine factor returns, benchmark, and portfolio into a single DataFrame
        self.regression_df = self.factor_returns_by_day.merge(self.benchmark_df, on='TRADE_DT', how='right')
        if self.custom_portfolio is not None:
            self.regression_df = self.regression_df.merge(self.custom_portfolio, on='TRADE_DT', how='right')

    def fit_model(self):
        # Fit regression model for benchmark and portfolio

        if self.benchmark_df is not None:
            model_benchmark = sm.OLS(self.regression_df['LOG_RETURN_BENCHMARK'],
                                     self.regression_df[self.factor_columns])
            self.results_benchmark = model_benchmark.fit()
            print("Benchmark Model Summary:")
            print(self.results_benchmark.summary())

        if self.custom_portfolio is not None:
            model_portfolio = sm.OLS(self.regression_df['LOG_RETURN_PORTFOLIO'],
                                     self.regression_df[self.factor_columns])
            self.results_portfolio = model_portfolio.fit()
            print("Portfolio Model Summary:")
            print(self.results_portfolio.summary())

    def get_factor_exposure(self):
        # Obtain factor exposure DataFrame
        factor_exposure_df = pd.DataFrame()
        if self.benchmark_df is not None:
            factor_exposure_df['factor_exposure_benchmark'] = self.results_benchmark.params
        if self.custom_portfolio is not None:
            factor_exposure_df['factor_exposure_portfolio'] = self.results_portfolio.params

        return factor_exposure_df

    def get_residuals_and_t_stats(self):
        # Obtain residuals DataFrame and calculate t-statistics
        residual_df = pd.DataFrame()
        if self.benchmark_df is not None:
            residual_df['residual_benchmark'] = self.results_benchmark.resid
            print("t-statistic for benchmark residuals:")
            print(residual_df['residual_benchmark'].mean() / (
                        residual_df['residual_benchmark'].std() / np.sqrt(len(residual_df['residual_benchmark']))))

        if self.custom_portfolio is not None:
            residual_df['residual_portfolio'] = self.results_portfolio.resid
            print("t-statistic for portfolio residuals:")
            print(residual_df['residual_portfolio'].mean() / (
                        residual_df['residual_portfolio'].std() / np.sqrt(len(residual_df['residual_portfolio']))))

        return residual_df

    def plot_nav_curve(self, initial_investment=1):
        """Plot NAV curve for portfolio and benchmark if they exist."""
        plt.figure(figsize=(12, 6))

        # Calculate NAV for the custom portfolio first
        if self.custom_portfolio is not None:
            nav_portfolio = initial_investment * (1 + self.custom_portfolio['RETURN_PORTFOLIO']).cumprod()
            plt.plot(self.custom_portfolio['TRADE_DT'], nav_portfolio, label='Portfolio NAV')

        # Calculate NAV for the benchmark if it exists
        if self.benchmark_df is not None:
            nav_benchmark = initial_investment * (1 + self.benchmark_df['RETURN_BENCHMARK']).cumprod()
            plt.plot(self.benchmark_df['TRADE_DT'], nav_benchmark, label='Benchmark NAV')

        # Formatting the plot
        plt.title('NAV Curve for Portfolio and Benchmark')
        plt.xlabel('Date')
        plt.ylabel('NAV')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    # Load factor returns from a Parquet file
    factor_returns_by_day = pd.read_parquet('/Volumes/quanyi4g/factor/day_frequency/barra/factor_return.parquet')

    # Fetch the custom portfolio DataFrame for the index code '000852.SH'
    wind = WindData(start_date='20190101', end_date='20240827')
    custom_portfolio = wind.get_index_price(index_code='000852.SH')
    custom_portfolio['TRADE_DT'] = pd.to_datetime(custom_portfolio['TRADE_DT'])

    # Instantiate the PortfolioRiskAttribution class
    risk_attribution = PortfolioRiskAttribution(
        start_date='20190101',
        end_date='20240827',
        factor_returns_by_day=factor_returns_by_day,
        benchmark=True,
        custom_portfolio=custom_portfolio
    )

    # Fit the model
    risk_attribution.fit_model()

    # Get factor exposure DataFrame
    factor_exposure_df = risk_attribution.get_factor_exposure()
    print("Factor Exposure DataFrame:")
    print(factor_exposure_df)

    # Get residuals DataFrame and t-statistics
    residual_df = risk_attribution.get_residuals_and_t_stats()
    print("\nMean Residual:")
    print(residual_df.mean())

    # Plot the NAV curves
    risk_attribution.plot_nav_curve()
