# -*- coding = utf-8 -*-
# @Time: 2024/09/27
# @Author: Xinyue
# @File:long_short_factor_earnings.py
# @Software: PyCharm

import pandas as pd


from Utils.decile_analysis import DecileAnalysis

factor_df = pd.read_parquet('Data/style_factors_updated.parquet')
factor_columns = ['BETA', 'RSTR', 'LNCAP', 'EARNYILD', 'GROWTH',
       'LEVERAGE', 'LIQUIDITY', 'RESVOL', 'BTOP', 'NLSIZE']

factor_return_long_short = pd.DataFrame()
for factor in factor_columns:
    single_factor_df = factor_df[['S_INFO_WINDCODE', 'TRADE_DT', 'S_VAL_MV', 'S_DQ_PRECLOSE', 'S_DQ_CLOSE', 'WIND_PRI_IND', factor]]
    decile_analysis = DecileAnalysis(single_factor_df, 5, factor, 'd', trade_pool=False, positive_return=False)
    long_short_df = decile_analysis.long_short_NAV()
    single_factor_return = long_short_df[['TRADE_DT', 'long_short_rt_adj']].rename(columns = {'long_short_rt_adj': factor}).set_index('TRADE_DT')
    factor_return_long_short = pd.concat([factor_return_long_short, single_factor_return], axis=1)
    print(f'Finished calculating {factor}.')

factor_return_long_short.reset_index().to_parquet('/Volumes/quanyi4g/factor/day_frequency/barra/factor_return(long-short).parquet')

