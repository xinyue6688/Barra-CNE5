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

def long_short_NAV(df, factor_decile_rt_df):
    """
    多空净值和基准净值对比

    :param
    :return: long_short_df 多空净值数据框，新增列：'long_short_turnover'（多空换手率）,
                                               'long_short_diff'（多空回报差异）,
                                               'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）,
                                               'NAV_adj'（调整后净值）
    """
    decile_factor = df.groupby(['TRADE_DT', 'DECILE'])['RSTR_LAG1'].mean().reset_index()
    factor_decile_rt_df = pd.merge(factor_decile_rt_df, decile_factor, how='left', on=['TRADE_DT', 'DECILE'])
    long_short_df = factor_decile_rt_df.pivot(index='TRADE_DT', columns='DECILE', values=['STOCK_RETURN', 'RSTR_LAG1'])
    long_short_df['long_short_rstr'] = long_short_df['RSTR_LAG1', 5] - long_short_df['RSTR_LAG1', 1]
    long_short_df['long_short_diff'] = long_short_df['STOCK_RETURN', 1] - long_short_df['STOCK_RETURN', 5]
    long_short_df['long_short_rt_adj'] = long_short_df['long_short_diff'] * (
            1 / long_short_df['long_short_rstr'])
    long_short_df['NAV_adj'] = (1 + long_short_df['long_short_rt_adj']).cumprod()
    long_short_df.reset_index(inplace=True)

    return long_short_df

long_short_df = long_short_NAV(all_mkt_mom_decile, mom_decile_rt_df)

from scipy.stats import pearsonr, spearmanr
def calculate_ic_metrics(df, factor_decile_rt_df):
    """
    计算IC、RankIC、ICIR、RankICIR、t-test

    :param self:  self.factor_decile_rt_df 包含以下列：'TRADE_DT'（交易日期）,
                                               'DECILE'（分组标签）,
                                               'STOCK_RETURN'（下一个交易日的回报率）,
                                               'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）
    :return: 各项指标结果
    """
    decile_factor = df.groupby(['TRADE_DT', 'DECILE'])['RSTR_LAG1'].mean().reset_index()
    factor_decile_rt_df = pd.merge(factor_decile_rt_df, decile_factor, how='left', on=['TRADE_DT', 'DECILE'])

    ic_values = []
    rank_ic_values = []

    for date, group in factor_decile_rt_df.groupby('TRADE_DT'):
        group = group.dropna(subset=['DECILE', 'STOCK_RETURN'])
        decile = pd.to_numeric(group['DECILE'], errors='coerce')
        future_return = pd.to_numeric(group['STOCK_RETURN'], errors='coerce')

        if len(decile) < 2 or decile.isnull().any() or future_return.isnull().any():
            ic_values.append(np.nan)
            rank_ic_values.append(np.nan)
            continue

        ic, _ = pearsonr(decile, future_return)
        ic_values.append(ic)

        rank_ic, _ = spearmanr(decile, future_return)
        rank_ic_values.append(rank_ic)

    ic_series = pd.Series(ic_values)
    rank_ic_series = pd.Series(rank_ic_values)

    icir = ic_series.mean() / ic_series.std()
    rank_icir = rank_ic_series.mean() / rank_ic_series.std()

    lambda_hat = long_short_df['long_short_rt_adj'].mean()
    se_lambda = np.std(long_short_df['long_short_rt_adj']) / np.sqrt(len(long_short_df))
    t_stat = lambda_hat / se_lambda if se_lambda != 0 else np.nan

    results = pd.DataFrame({
        'IC': [ic_series.mean()],
        'RankIC': [rank_ic_series.mean()],
        'ICIR': [icir],
        'RankICIR': [rank_icir],
        't-stat': [t_stat]
    })

    return results

mom_metrics = calculate_ic_metrics(all_mkt_mom_decile, mom_decile_rt_df)