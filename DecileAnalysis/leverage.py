# -*- coding = utf-8 -*-
# @Time: 2024/09/04
# @Author: Xinyue
# @File:leverage.py
# @Software: PyCharm

import numpy as np
import pandas as pd


from barra_cne5_factor import leverage
from Utils.connect_wind import ConnectDatabase


# 获取杠杆因子相关数据
leverage_calculator = leverage()

# 全市场去除北交所
all_mkt_price_df = pd.read_parquet('../Data/all_market_data.parquet')
all_mkt_price_df = all_mkt_price_df[~all_mkt_price_df['S_INFO_WINDCODE'].str.endswith('BJ')].reset_index(drop=True)
all_mkt_price_df = all_mkt_price_df.sort_values('TRADE_DT')


# 每个交易日合并到最新披露的长期负债、总负债、和总资产
leverage_calculator.balance_data['ANN_DT'] = pd.to_datetime(leverage_calculator.balance_data['ANN_DT'])
leverage_calculator.balance_data = leverage_calculator.balance_data.sort_values('ANN_DT')
all_mkt_price_df = pd.merge_asof(
    all_mkt_price_df,
    leverage_calculator.balance_data,
    by = 'S_INFO_WINDCODE',
    left_on='TRADE_DT',
    right_on='ANN_DT',
    direction='backward'
)

no_data_stock = []
grouped = all_mkt_price_df.groupby('S_INFO_WINDCODE')
for stock_code, group in grouped:
    if group['ANN_DT'].isna().all():
        no_data_stock.append(stock_code)

# 每股净资产
leverage_calculator.bppershare_data.dropna(how = 'any', inplace = True)
leverage_calculator.bppershare_data['ANN_DT'] = pd.to_datetime(leverage_calculator.bppershare_data['ANN_DT'])
leverage_calculator.bppershare_data = leverage_calculator.bppershare_data.sort_values('ANN_DT')

all_mkt_price_df = pd.merge_asof(
    all_mkt_price_df,
    leverage_calculator.bppershare_data,
    by = 'S_INFO_WINDCODE',
    left_on='TRADE_DT',
    right_on='ANN_DT',
    direction='backward'
)

# 优先股股数（单位万）
leverage_calculator.ncapital_data.dropna(how = 'any', inplace = True)
leverage_calculator.ncapital_data['CHANGE_DT'] = pd.to_datetime(leverage_calculator.ncapital_data['CHANGE_DT'])
leverage_calculator.ncapital_data = leverage_calculator.ncapital_data.sort_values('CHANGE_DT')

all_mkt_price_df = pd.merge_asof(
    all_mkt_price_df,
    leverage_calculator.ncapital_data,
    by = 'S_INFO_WINDCODE',
    left_on='TRADE_DT',
    right_on='CHANGE_DT',
    direction='backward'
)

# 删除不需要的日期标注列
all_mkt_price_df = all_mkt_price_df.drop(columns = ['ANN_DT_x', 'ANN_DT_y', 'CHANGE_DT'])
all_mkt_price_df['PE'] = all_mkt_price_df['S_FA_BPS'] * all_mkt_price_df['S_SHARE_NTRD_PRFSHARE']
one_stock_df = all_mkt_price_df[all_mkt_price_df['S_INFO_WINDCODE'] == '002112.SZ']

# 删除需要用到的数据中有空值的列
all_mkt_price_df = all_mkt_price_df.dropna(subset = ['LD', 'TD', 'TA', 'PE'])
