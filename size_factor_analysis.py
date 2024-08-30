# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:size_factor_analysis.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from datetime import datetime


from barra_cne5_factor import Size
from Utils.decile_analysis import DecileAnalysis
from Utils.data_clean import DataProcess

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
all_market_data = pd.read_parquet('Data/all_market_data.parquet')
all_market_data = all_market_data[~all_market_data['S_INFO_WINDCODE'].str.endswith('BJ')]

# 创建计算市值因子实例
size_calculator = Size()
all_market_lncap_df = size_calculator.LNCAP(all_market_data)
print(all_market_lncap_df.head())

# 创建市值分层分析实例
lncap_analysis = DecileAnalysis(all_market_lncap_df, decile_num=5, factor='LNCAP', rebal_freq='w', mv_neutral=False, trade_pool=True)
df_with_decile = lncap_analysis.df_with_decile
lncap_analysis.plot_decile_returns()
lncap_analysis.plot_long_short_NAV()
lncap_analysis.print_long_short_metrics()
lncap_analysis.print_icir_bystock()






