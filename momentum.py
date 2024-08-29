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
from Utils.get_wind_data import WindData

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


trade_pool = pd.read_parquet('/Volumes/quanyi4g/data/index/pool/tradable_components.parquet')
pool_mom = pd.merge(trade_pool, all_mkt_mom, on = ['TRADE_DT', 'S_INFO_WINDCODE'])

analysis = DecileAnalysis(pool_mom, 5, 'RSTR', 'm')
mom_decile_rt_df = analysis.calculate_decile_returns()
long_short_df = analysis.long_short_NAV(mom_decile_rt_df)

wind_data = WindData(START_DATE, END_DATE)
benchmark = wind_data.get_industry_index('8841388.WI')
benchmark['TRADE_DT'] = pd.to_datetime(benchmark['TRADE_DT'])
benchmark['S_DQ_PCTCHANGE'] = benchmark['S_DQ_PCTCHANGE'].astype(float)
benchmark['S_DQ_PCTCHANGE'] = benchmark['S_DQ_PCTCHANGE'] * 0.01
benchmark['NAV'] = (1 + benchmark['S_DQ_PCTCHANGE']).cumprod()

if isinstance(long_short_df.columns, pd.MultiIndex):
    long_short_df.columns = ['_'.join(map(str, col)).strip() if type(col) is tuple else col for col in long_short_df.columns]
long_short_df.rename(columns={'TRADE_DT_': 'TRADE_DT',
                              'NAV_adj_': 'NAV_adj',
                              'long_short_rt_adj_': 'long_short_rt'}, inplace=True)

aligned_df = pd.merge(long_short_df[['TRADE_DT', 'NAV_adj']], benchmark[['TRADE_DT', 'NAV']], on='TRADE_DT', how='inner')
plt.figure(figsize=(12, 8))
plt.title('NAV')
plt.xlabel('Date')
plt.ylabel('Cumulative NAV')
plt.plot(aligned_df['TRADE_DT'], aligned_df['NAV_adj'], label='Long-Short Portfolio Adjusted (Exposure 1)')
plt.plot(aligned_df['TRADE_DT'], aligned_df['NAV'], label='EW A Index')
plt.legend()
plt.show()
results = analysis.calculate_ic_metrics()
print(results)
