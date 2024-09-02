# -*- coding = utf-8 -*-
# @Time: 2024/09/2
# @Author: Xinyue
# @File:non_linear_size.py
# @Software: PyCharm

import pandas as pd


from barra_cne5_factor import Size
from Utils.decile_analysis import DecileAnalysis


# 全市场去除北交所，格式：日期、股票代码、昨收、今收、市值、行业
all_market_data = pd.read_parquet('../Data/all_market_data.parquet')
all_market_data = all_market_data[~all_market_data['S_INFO_WINDCODE'].str.endswith('BJ')]

# 创建计算市值因子实例
size_calculator = Size()
all_market_nlsize_df = size_calculator.NLSIZE(all_market_data)
print(all_market_nlsize_df.head())

# 创建市值分层分析实例
nlsize_analysis = DecileAnalysis(all_market_nlsize_df, decile_num=5, factor='NLSIZE', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = nlsize_analysis.df_with_decile
nlsize_analysis.plot_decile_returns()
nlsize_analysis.plot_long_short_NAV()
nlsize_analysis.print_long_short_metrics()
nlsize_analysis.print_icir_bystock()