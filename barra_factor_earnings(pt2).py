# -*- coding = utf-8 -*-
# @Time: 2024/09/23
# @Author: Xinyue
# @File:xxx.py
# @Software: PyCharm

import pandas as pd
import statsmodels.api as sm
import numpy as np

factor_df = pd.read_parquet('Data/style_factors_updated.parquet')
factor_df.dropna(subset = 'S_VAL_MV', inplace = True)

# Define the factor columns excluding 'WIND_PRI_IND'
factor_columns = ['COUNTRY', 'BETA', 'RSTR', 'LNCAP', 'EARNYILD', 'GROWTH',
                  'LEVERAGE', 'LIQUIDITY', 'RESVOL', 'BTOP', 'NLSIZE']


# Define a function to run weighted OLS regression for each day
def weighted_ols_for_day(df):
    # Convert 'WIND_PRI_IND' into dummy variables
    df_with_dummies = pd.get_dummies(df, columns=['WIND_PRI_IND'],
                                     drop_first=True, dtype = float)  # drop_first=True avoids multicollinearity

    # Append the dummy variable columns to factor columns
    X = df_with_dummies[factor_columns + [col for col in df_with_dummies if col.startswith('WIND_PRI_IND_')]]

    # Dependent variable (stock returns)
    y = df['RETURN_T1']

    # Market value as weights
    df['S_VAL_MV'] = pd.to_numeric(df['S_VAL_MV'])
    weights = df['S_VAL_MV']

    # Perform weighted OLS regression without a constant
    try:
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()

        # Return the factor returns (coefficients)
        return results.params

    except np.linalg.LinAlgError as e:
        print(f"SVD did not converge for {df['TRADE_DT'].iloc[0]}: {e}")
        return pd.Series([None] * len(X.columns), index=X.columns)


# Group by 'TRADE_DT' and apply the regression for each day
factor_returns_by_day = factor_df.groupby('TRADE_DT').apply(weighted_ols_for_day, include_groups=False)

# Reset the index to make 'TRADE_DT' a column
factor_returns_by_day = factor_returns_by_day.reset_index()

import scipy.stats as stats

# Assuming 'factor_returns_by_day' has columns 'COUNTRY' (coefficients) and 'COUNTRY_se' (standard errors)
t_statistic = factor_returns_by_day['COUNTRY'] / factor_returns_by_day['COUNTRY_se']

# Calculate the p-value
df = len(factor_returns_by_day) - 1  # degrees of freedom
p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))  # two-tailed test

# Add t-statistic and p-values to DataFrame
factor_returns_by_day['t_statistic'] = t_statistic
factor_returns_by_day['p_value'] = p_values

# Print results
for index, row in factor_returns_by_day.iterrows():
    print(f"Date: {index}, COUNTRY Coefficient: {row['COUNTRY']}, "
          f"t-statistic: {row['t_statistic']}, p-value: {row['p_value']}")
