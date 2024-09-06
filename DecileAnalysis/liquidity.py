# -*- coding = utf-8 -*-
# @Time: 2024/09/06
# @Author: Xinyue
# @File:liquidity.py
# @Software: PyCharm

import pandas as pd

from barra_cne5_factor import liquidity
from Utils.decile_analysis import DecileAnalysis

# 全市场去除北交所
all_mkt_price_df = pd.read_parquet('../Data/all_market_data.parquet')
all_mkt_price_df = all_mkt_price_df[~all_mkt_price_df['S_INFO_WINDCODE'].str.endswith('BJ')].reset_index(drop=True)
all_mkt_price_df = all_mkt_price_df.sort_values('TRADE_DT')

# 获取杠杆因子相关数据
liquidity_calculator = liquidity(all_mkt_price_df)
all_mkt_liquidity = liquidity_calculator.LIQUIDITY()

# 创建杠杆因子分层分析实例
liquidity_analysis = DecileAnalysis(all_mkt_liquidity, decile_num=5, factor='LIQUIDITY', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = liquidity_analysis.df_with_decile
liquidity_analysis.plot_decile_returns()
liquidity_analysis.plot_long_short_NAV()
liquidity_analysis.print_long_short_metrics()
liquidity_analysis.print_icir_bystock()