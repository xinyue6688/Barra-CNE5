# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:momentum.py
# @Software: PyCharm

import pandas as pd


from barra_cne5_factor import Momentum
from Utils.decile_analysis import DecileAnalysis

'''
all_price_df = GetData.all_price()
all_mv_df = GetData.all_mv()
all_price_mv_df = pd.merge(all_price_df, all_mv_df, how='left', on = ['TRADE_DT', 'S_INFO_WINDCODE'])
START_DATE = '20071112'
END_DATE = datetime.now().strftime('%Y%m%d')
data_processer = DataProcess(START_DATE, END_DATE)
all_market_data = data_processer.assign_industry(all_price_mv_df)
all_market_data.to_parquet('Data/all_market_data.parquet')
'''

# 全市场去除北交所，格式：日期、股票代码、昨收、今收、市值、行业
all_market_data = pd.read_parquet('../Data/all_market_data.parquet')
all_market_data = all_market_data[~all_market_data['S_INFO_WINDCODE'].str.endswith('BJ')]

# 创建计算动量因子实例
calculate_mom = Momentum()
all_market_rstr = calculate_mom.RSTR(all_market_data)
all_market_rstr = all_market_rstr.dropna(subset = ['RSTR'])

'''grouped = all_market_rstr.groupby('S_INFO_WINDCODE')
for stock_code, group in grouped:
    data = group[['S_INFO_WINDCODE', 'TRADE_DT', 'RSTR']]
    data.to_parquet(f'/Volumes/quanyi4g/factor/day_frequency/barra/Momentum/{stock_code.replace('.','_')}_momentum.parquet', index=False)'''

# 创建市值分层分析实例
rstr_analysis = DecileAnalysis(all_market_rstr, decile_num=5, factor='RSTR', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = rstr_analysis.df_with_decile
rstr_analysis.plot_decile_returns()
rstr_analysis.plot_long_short_NAV(double_axis=True)
rstr_analysis.print_long_short_metrics()
rstr_analysis.print_icir_bystock()

