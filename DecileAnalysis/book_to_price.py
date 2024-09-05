# -*- coding = utf-8 -*-
# @Time: 2024/09/03
# @Author: Xinyue
# @File:book_to_price.py
# @Software: PyCharm


import pandas as pd


from barra_cne5_factor import btop
from Utils.decile_analysis import DecileAnalysis

# 全市场去除北交所，格式：日期、股票代码、昨收、今收、市值、行业
all_market_data = pd.read_parquet('../Data/all_market_data.parquet')
all_market_data = all_market_data[~all_market_data['S_INFO_WINDCODE'].str.endswith('BJ')]

# 创建计算市值因子实例
bp_calculator = btop()
all_market_btop_df = bp_calculator.BTOP(all_market_data)
print(all_market_btop_df.head())

# 创建市值分层分析实例
bp_analysis = DecileAnalysis(all_market_btop_df, decile_num=5, factor='BTOP', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = bp_analysis.df_with_decile
bp_analysis.plot_decile_returns()
bp_analysis.plot_long_short_NAV()
bp_analysis.print_long_short_metrics()
bp_analysis.print_icir_bystock()