# -*- coding = utf-8 -*-
# @Time: 2024/09/03
# @Author: Xinyue
# @File:residual_vol.py
# @Software: PyCharm


import pandas as pd


from barra_cne5_factor import ResidualVolatility
from Utils.decile_analysis import DecileAnalysis

# 全市场去除北交所
all_mkt_price_df = pd.read_parquet('../Data/alpha_beta_all_market.parquet')
all_mkt_price_df = all_mkt_price_df[~all_mkt_price_df['S_INFO_WINDCODE'].str.endswith('BJ')].reset_index(drop=True)

# 创建计算RESVOL实例
calculate_resvol = ResidualVolatility()
all_mkt_resvol = calculate_resvol.RESVOL(all_mkt_price_df)

# 创建分层分析实例
resvol_analysis = DecileAnalysis(all_mkt_resvol, decile_num=5, factor='RESVOL', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = resvol_analysis.df_with_decile
resvol_analysis.plot_decile_returns()
resvol_analysis.plot_long_short_NAV()
resvol_analysis.print_long_short_metrics()
resvol_analysis.print_icir_bystock()