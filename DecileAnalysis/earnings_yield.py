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
prepared_df = ey_calculator.df

grouped = prepared_df.groupby('S_INFO_WINDCODE')
no_epibs_list = []
no_earnings_list = []
no_cash_list = []
for stock_code, group in grouped:
    if group['EPIBS'].isna().all():
        no_epibs_list.append(stock_code)
    if group['EARNINGS_TTM'].isna().all():
        no_earnings_list.append(stock_code)
    if group['CASH_EARNINGS_TTM'].isna().all():
        no_cash_list.append(stock_code)

prepared_df['ETOP'] = prepared_df['EARNINGS_TTM'] / prepared_df['S_VAL_MV']
prepared_df['CETOP'] = prepared_df['CASH_EARNINGS_TTM'] / prepared_df['S_VAL_MV']
for columns in ['EPIBS', 'ETOP', 'CETOP']:
    prepared_df[columns] = prepared_df[columns].astype('float64')
prepared_df['EARNYILD'] = 0.68 * prepared_df['EPIBS'] + 0.11 * prepared_df['ETOP'] + 0.21 * prepared_df['CETOP']
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