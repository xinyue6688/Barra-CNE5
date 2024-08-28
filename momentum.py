# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:xxx.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats.mstats import winsorize
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

from barra_cne5_factor import GetData, Calculation
from Utils.data_clean import DataProcess
from Utils.decile_analysis import DecileAnalysis
from Utils.connect_wind import ConnectDatabase

risk_free = GetData.risk_free()
trade_pool = pd.read_parquet('/Volumes/quanyi4g/data/index/pool/tradable_components.parquet')

exp_weight = Calculation._exp_weight(window = 504, half_life = 126)


all_mkt_mom = pd.DataFrame()
all_mkt_codes = trade_pool['S_INFO_WINDCODE'].unique()

count = 0
for i in range(len(all_mkt_codes)):
    stock_rt = GetData.daily_price(all_mkt_codes[i])
    if len(stock_rt) < 504 + 21:
        count += 1
        print(f'Not enough data to calculate Momentum for {all_mkt_codes[i]}')
        continue
    stock_rt['TRADE_DT'] = pd.to_datetime(stock_rt['TRADE_DT'])
    stock_rt['STOCK_RETURN'] = stock_rt['S_DQ_CLOSE'] / stock_rt['S_DQ_PRECLOSE'] - 1
    stock_rt['STOCK_RETURN'] = stock_rt['STOCK_RETURN'].astype(float)
    stock_rt = pd.merge(stock_rt, risk_free, how='left', on = 'TRADE_DT')
    stock_rt['EX_LG_RT'] = np.log(1+stock_rt['STOCK_RETURN']) - np.log(1+stock_rt['RF_RETURN'])
    try:
        stock_rt['RSTR'] = stock_rt['EX_LG_RT'].shift(21).rolling(window = 504).apply(lambda x: np.sum(x * exp_weight))
    except Exception as e:
        print(f'Error processing {all_mkt_codes[i]}: {e}')
        break

    store_df = stock_rt[['TRADE_DT', 'S_INFO_WINDCODE', 'RSTR']].copy()
    store_df.to_parquet(
        f'/Volumes/quanyi4g/factor/day_frequency/barra/Momentum/{all_mkt_codes[i].replace(".", "_")}_momentum.parquet')

    stock_rt = stock_rt.dropna(subset=['RSTR'])
    stock_rt = stock_rt.drop(columns = ['EX_LG_RT']).reset_index(drop=True)

    all_mkt_mom = pd.concat([all_mkt_mom, stock_rt])

    count += 1
    print(f'STOCK {all_mkt_codes[i]}:')
    print(stock_rt.head(1))
    print(f'Finished {count}, {len(all_mkt_codes) - count} left')


print('All market Momentum dataframe:')
print(all_mkt_mom.head())

START_DATE = '20071112'
END_DATE = datetime.now().strftime('%Y%m%d')
sql = f'''
               SELECT S_INFO_WINDCODE, TRADE_DT, S_VAL_MV
               FROM ASHAREEODDERIVATIVEINDICATOR      
               WHERE TRADE_DT BETWEEN {START_DATE} AND {END_DATE}
               '''

connection = ConnectDatabase(sql)
mv_df = connection.get_data()
mv_df.sort_values(by='TRADE_DT', inplace=True)
mv_df.reset_index(drop=True, inplace=True)
mv_df['TRADE_DT'] = pd.to_datetime(mv_df['TRADE_DT'])
all_mkt_mom = pd.merge(all_mkt_mom, mv_df, on=['S_INFO_WINDCODE', 'TRADE_DT'])

data_processer = DataProcess(START_DATE, END_DATE)
all_mkt_mom = data_processer.assign_industry(all_mkt_mom)
all_mkt_mom.to_parquet('Data/all_market_momentum.parquet')


analysis = DecileAnalysis(decile_num=5, factor='RSTR', df=all_mkt_mom)
mom_decile_rt_df = analysis.calculate_average_daily_returns()

########检查问题###########
mom_decile_nav_pivot = mom_decile_rt_df.pivot(index='TRADE_DT', columns='DECILE', values=['NAV'])
group_by_date = all_mkt_mom.groupby(['TRADE_DT'])
date_lengths = []
for date, group in group_by_date:
    date_lengths.append({'TRADE_DT': date, 'Group_Length': len(group)})
date_lengths_df = pd.DataFrame(date_lengths)
#########################

mom_metrics = calculate_ic_metrics(all_mkt_mom_decile, mom_decile_rt_df)
