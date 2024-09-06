# -*- coding = utf-8 -*-
# @Time: 2024/09/04
# @Author: Xinyue
# @File:leverage.py
# @Software: PyCharm

import numpy as np
import pandas as pd


from barra_cne5_factor import leverage
from Utils.decile_analysis import DecileAnalysis


# 全市场去除北交所
all_mkt_price_df = pd.read_parquet('../Data/all_market_data.parquet')
all_mkt_price_df = all_mkt_price_df[~all_mkt_price_df['S_INFO_WINDCODE'].str.endswith('BJ')].reset_index(drop=True)
all_mkt_price_df = all_mkt_price_df.sort_values('TRADE_DT')

# 获取杠杆因子相关数据
leverage_calculator = leverage(all_mkt_price_df)
all_mkt_leverage = leverage_calculator.LEVERAGE()

# 创建杠杆因子分层分析实例
leverage_analysis = DecileAnalysis(all_mkt_leverage, decile_num=5, factor='LEVERAGE', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = leverage_analysis.df_with_decile
leverage_analysis.plot_decile_returns()
leverage_analysis.plot_long_short_NAV()
leverage_analysis.print_long_short_metrics()
leverage_analysis.print_icir_bystock()