# -*- coding = utf-8 -*-
# @Time: 2024/08/21
# @Author: Xinyue
# @File:barra_cne5_factor.py
# @Software: PyCharm

import pandas_market_calendars as mcal
import pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

from Utils.connect_wind import ConnectDatabase

cn_trade_calender = mcal.get_calendar('XSHG')
trade_day = cn_trade_calender.schedule(pd.Timestamp('2007-06-01'), pd.Timestamp('2010-01-01'))

START_DATE = trade_day.index[-525].strftime('%Y%m%d')
END_DATE = datetime.now().strftime('%Y%m%d')


class lazyproperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.func(instance)
        setattr(instance, self.name, value)  # Cache the computed value
        return value


class GetData(ConnectDatabase):

    @staticmethod
    def risk_free():
        """
        获取无风险利率（十年国债收益率）
        :return: 无风险利率数据框 格式：日期，年化收益
        """
        current_df_start_time = datetime.strptime(START_DATE, "%Y%m%d")
        end_date_time = datetime.strptime(END_DATE, "%Y%m%d")
        yield10yr_df = pd.DataFrame()

        while current_df_start_time < end_date_time:
            current_df_end_time = min(current_df_start_time + timedelta(days=365), end_date_time)

            bond_china_yield_df = ak.bond_china_yield(
                start_date=current_df_start_time.strftime("%Y%m%d"),
                end_date=current_df_end_time.strftime("%Y%m%d")
            )

            filtered_df = bond_china_yield_df[
                (bond_china_yield_df['曲线名称'] == '中债国债收益率曲线')
            ][['日期', '10年']]

            yield10yr_df = pd.concat([yield10yr_df, filtered_df])

            current_df_start_time = current_df_end_time + timedelta(days=1)

        yield10yr_df.reset_index(drop=True, inplace=True)
        yield10yr_df['RF_RETURN_ANN'] = yield10yr_df['10年'] / 100
        yield10yr_df['TRADE_DT'] = pd.to_datetime(yield10yr_df['日期'])
        yield10yr_df['RF_RETURN'] = (1 + yield10yr_df['RF_RETURN_ANN']) ** (1 / 252) - 1

        rf = yield10yr_df[['TRADE_DT', 'RF_RETURN']]
        return rf

    @staticmethod
    def daily_price(windcode):
        """
        :param windcode: 股票代码 ‘000001.SZ’
        :return: 日频收益数据框，格式：日期，股票代码，收益，市值
        """
        sql = f'''
        SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_CLOSE
        FROM ASHAREEODPRICES
        WHERE S_INFO_WINDCODE = '{windcode}'
        AND TRADE_DT BETWEEN {START_DATE} AND {END_DATE}
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def all_mv():

        sql = f'''
               SELECT S_INFO_WINDCODE, TRADE_DT, S_VAL_MV
               FROM ASHAREEODDERIVATIVEINDICATOR      
               WHERE TRADE_DT BETWEEN {START_DATE} AND {END_DATE}
               '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
        return df

    @staticmethod
    def all_price():
        sql = f'''
                SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_CLOSE
                FROM ASHAREEODPRICES
                WHERE TRADE_DT BETWEEN {START_DATE} AND {END_DATE}
                '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
        return df



class Calculation:
    @staticmethod
    def _exp_weight(window, half_life):
        exp_weight = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_weight[::-1] / np.sum(exp_weight)

    @staticmethod
    def _winsorize(x):
        """
        去极值，使因子值在均值的3个标准差范围内
        """
        mean = x.mean()
        std = x.std()
        winsorized = x.copy()
        winsorized[x < mean - 3 * std] = mean - 3 * std
        winsorized[x > mean + 3 * std] = mean + 3 * std
        return winsorized

    @staticmethod
    def _standardize(x, market_value):
        """
        市值加权标准化
        """
        if market_value.dtype != np.float64:
            market_value = market_value.astype(np.float64)

        w_mean = np.sum(x * market_value) / np.sum(market_value)
        std = x.std()
        standardized = (x - w_mean) / std
        return standardized

    def _preprocess(self, data, factor_column):
        data[f'{factor_column}_wsr'] = data.groupby('TRADE_DT')[f'{factor_column}'].transform(lambda x: self._winsorize(x))
        data[f'{factor_column}_ppd'] = data.groupby('TRADE_DT').apply(lambda g: self._standardize(g[f'{factor_column}_wsr'], g['S_VAL_MV'])).reset_index(level=0, drop=True)
        data.drop(columns=[f'{factor_column}_wsr'], inplace=True)

        return data


class Momentum(Calculation):


    def RSTR(self, df, raw = False):
        """
        :param df:
        :return:
        """
        df['STOCK_RETURN'] = df['S_DQ_CLOSE'] / df['S_DQ_PRECLOSE'] - 1
        df['STOCK_RETURN'] = df['STOCK_RETURN'].astype('float')

        exp_weight = self._exp_weight(window = 504, half_life = 126)
        grouped = df.groupby('S_INFO_WINDCODE')

        for stock_code, group in grouped:
            if len(group) < 504 + 21:
                print(f'Not enough data to calculate Momentum for {stock_code}')
                df.loc[group.index, 'RSTR'] = np.nan

            else:
                group['EX_LG_RT'] = np.log(1 + group['STOCK_RETURN']) - np.log(1 + group['RF_RETURN'])

                try:
                    group['RSTR'] = group['EX_LG_RT'].shift(21).rolling(window=504).apply(
                        lambda x: np.sum(x * exp_weight))

                    df.loc[group.index, 'RSTR'] = group['RSTR']

                except Exception as e:
                    print(f'Error processing {stock_code}: {e}')
                    df.loc[group.index, 'RSTR'] = np.nan

        if not raw:
            df = self._preprocess(df, factor_column = 'RSTR')

        return df

class Size(Calculation):

    @lazyproperty
    def LNCAP(self, df, raw = False):
        df['LNCAP'] = np.log(df['S_VAL_MV'])
        if not raw:
            df = self._preprocess(df, factor_column = 'LNCAP')
        return df

    def NLSIZE(self, df, raw = False):
        df = self.LNCAP(df)
        df['SIZE_CUBED'] = np.power(df['LNCAP'], 3)
        df['CONSTANT'] = 1
        X = df[['LNCAP','CONSTANT']]
        y = df['SIZE_CUBED']
        model = sm.OLS(y, X).fit()
        df['NLSIZE'] = model.resid
        if not raw:
            df = self._preprocess(df, factor_column = 'NLSIZE')

        df.drop(columns=['SIZE_CUBED', 'LNCAP', 'CONSTANT'], inplace=True)
        return df


if __name__ == '__main__':
    all_market_mom = pd.read_parquet('/Users/xinyuezhang/Desktop/中量投/Project/ZLT-Project Barra/Data/all_market_data.parquet')
    risk_free = pd.read_parquet('/Volumes/quanyi4g/factor/day_frequency/fundamental/RiskFree/risk_free.parquet')
    all_market_mom = pd.merge(all_market_mom, risk_free, on = 'TRADE_DT', how = 'left')
    mom = Momentum()
    all_market_mom = mom.RSTR(all_market_mom)

