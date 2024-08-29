# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:xxx.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from barra_cne5_factor import GetData
from Utils.decile_analysis import DecileAnalysis

trade_pool = pd.read_parquet('/Volumes/quanyi4g/data/index/pool/tradable_components.parquet')
all_mkt_mom = pd.read_parquet('Data/all_market_momentum.parquet')

pool_mom = pd.merge(trade_pool, all_mkt_mom, on = ['TRADE_DT', 'S_INFO_WINDCODE'])

analysis = DecileAnalysis(pool_mom, 5, 'RSTR', 'w')
analysis.plot_decile_returns()
mom_decile_rt_df = analysis.factor_decile_rt_df
long_short_df = analysis.long_short_NAV(mom_decile_rt_df)
analysis.print_ic_metrics()


benchmark = GetData.industry_index('8841388.WI')
benchmark['TRADE_DT'] = pd.to_datetime(benchmark['TRADE_DT'])
benchmark['S_DQ_PCTCHANGE'] = benchmark['S_DQ_PCTCHANGE'].astype(float)
benchmark['S_DQ_PCTCHANGE'] = benchmark['S_DQ_PCTCHANGE'] * 0.01
benchmark['NAV'] = (1 + benchmark['S_DQ_PCTCHANGE']).cumprod()

if isinstance(long_short_df.columns, pd.MultiIndex):
    long_short_df.columns = ['_'.join(map(str, col)).strip() if type(col) is tuple else col for col in long_short_df.columns]
long_short_df.rename(columns={'TRADE_DT_': 'TRADE_DT',
                              'NAV_adj_': 'NAV_adj',
                              'long_short_rt_adj_': 'long_short_rt'}, inplace=True)

aligned_df = pd.merge(long_short_df[['TRADE_DT', 'NAV_adj']], benchmark[['TRADE_DT', 'NAV']], on='TRADE_DT', how='inner')
plt.figure(figsize=(12, 8))
plt.title('NAV')
plt.xlabel('Date')
plt.ylabel('Cumulative NAV')
plt.plot(aligned_df['TRADE_DT'], aligned_df['NAV_adj'], label='Long-Short Portfolio Adjusted (Exposure 1)')
plt.plot(aligned_df['TRADE_DT'], aligned_df['NAV'], label='EW A Index')
plt.legend()
plt.show()

