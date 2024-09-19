# -*- coding = utf-8 -*-
# @Time: 2024/09/03
# @Author: Xinyue
# @File:earnings_yield.py
# @Software: PyCharm

import pandas as pd

from barra_cne5_factor import GetData, EarningsYield
from Utils.decile_analysis import DecileAnalysis

# 全市场去除北交所
all_mkt_price_df = pd.read_parquet('../Data/all_market_data.parquet')
all_mkt_price_df = all_mkt_price_df[~all_mkt_price_df['S_INFO_WINDCODE'].str.endswith('BJ')].reset_index(drop=True)


ey_calculator = EarningsYield(all_mkt_price_df)
all_mkt_ey = ey_calculator.EARNYILD()
#epibs_df = GetData.est_consensus()
#check_000004 = epibs_df[epibs_df['S_INFO_WINDCODE'] == '000004.SZ']

#epibs_df_rolling = GetData.est_consensus_rolling()


# 创建收益因子分层分析实例
ey_analysis = DecileAnalysis(prepared_df, decile_num=5, factor='EARNYILD', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = ey_analysis.df_with_decile
ey_analysis.plot_decile_returns()
ey_analysis.plot_long_short_NAV()
ey_analysis.print_long_short_metrics()
ey_analysis.print_icir_bystock()