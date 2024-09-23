# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:xxx.py
# @Software: PyCharm

import numpy as np
import pandas as pd


from barra_cne5_factor import growth
from Utils.decile_analysis import DecileAnalysis

# 全市场去除北交所
all_mkt_price_df = pd.read_parquet('../Data/all_market_data.parquet')
all_mkt_price_df = all_mkt_price_df[~all_mkt_price_df['S_INFO_WINDCODE'].str.endswith('BJ')].reset_index(drop=True)
all_mkt_price_df = all_mkt_price_df.sort_values('TRADE_DT')

# 获取成长因子相关数据
growth_calculator = growth(all_mkt_price_df)
all_mkt_growth = growth_calculator.GROWTH()

grouped = all_mkt_growth.groupby('S_INFO_WINDCODE')
for stock_code, group in grouped:
    data = group[['S_INFO_WINDCODE', 'TRADE_DT', 'GROWTH']]
    data.to_parquet(f'/Volumes/quanyi4g/factor/day_frequency/barra/Growth/{stock_code.replace('.','_')}_growth.parquet', index=False)



'''all_mkt_growth = all_mkt_growth[all_mkt_growth['TRADE_DT'].dt.year >= 2013]

# 创建成长因子分层分析实例
growth_analysis = DecileAnalysis(all_mkt_growth, decile_num=5, factor='GROWTH', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = growth_analysis.df_with_decile
growth_analysis.plot_decile_returns()
growth_analysis.plot_long_short_NAV(double_axis=True)
growth_analysis.print_long_short_metrics()
growth_analysis.print_icir_bystock()'''



