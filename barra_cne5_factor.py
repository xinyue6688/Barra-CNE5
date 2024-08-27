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


class Calculation:
    @staticmethod
    def _exp_weight(window, half_life):
        exp_weight = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_weight[::-1] / np.sum(exp_weight)

    @staticmethod
    def _regress(y, X, intercept=True, weight=1, verbose=True):
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.DataFrame(y)
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)

        if intercept:
            cols = X.columns.tolist()
            X['const'] = 1
            X = X[['const'] + cols]

        model = sm.WLS(y, X, weights=weight)
        result = model.fit()
        params = result.params

        if verbose:
            resid = y - pd.DataFrame(np.dot(X, params), index=y.index,
                                     columns=y.columns)
            if intercept:
                return params.iloc[1:], params.iloc[0], resid
            else:
                return params, None, resid
        else:
            if intercept:
                return params.iloc[1:]
            else:
                return params


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
    def merged_data():
        """
        :param 股票代码 ‘000001.SZ’
        :return: 日频收益数据框，格式：日期，股票代码，收益，市值
        """
        sql = f'''
        SELECT p.S_INFO_WINDCODE, p.TRADE_DT, p.S_DQ_PRECLOSE, p.S_DQ_CLOSE, d.S_VAL_MV
        FROM ASHAREEODPRICES AS p
        LEFT JOIN ASHAREEODDERIVATIVEINDICATOR AS d
        ON p.S_INFO_WINDCODE = d.S_INFO_WINDCODE
        AND p.TRADE_DT = d.TRADE_DT
        AND p.TRADE_DT BETWEEN {START_DATE} AND {END_DATE}
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['STOCK_RETURN'] = df['S_DQ_CLOSE'] / df['S_DQ_PRECLOSE'] - 1
        merged_data = df[['TRADE_DT', 'S_INFO_WINDCODE', 'STOCK_RETURN', 'S_VAL_MV']]
        return merged_data

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

# TODO: class Beta(GetData):


# TODO: def beta(self):

class Momentum(Calculation):

    @staticmethod
    def RSTR(df):
        """
        :param df:
        :return:
        """
        exp_weight = Calculation._exp_weight(window = 504+21, half_life = 126)[:504]
        stock_code = df['S_INFO_WINDCODE'].unique()

        if len(df) < 504 + 21:
            print(f'Not enough data to calculate Momentum for {stock_code}')
            return df
        df['EX_LG_RT'] = np.log(1 + df['STOCK_RETURN']) - np.log(1 + df['RF_RETURN'])
        try:
            df['RSTR'] = df['EX_LG_RT'].shift(21).rolling(window=504).apply(
                lambda x: np.sum(x * exp_weight))
        except Exception as e:
            print(f'Error processing {stock_code}: {e}')
            return df
        df = df.dropna(subset=['RSTR'])
        df = df.drop(columns=['EX_LG_RT']).reset_index(drop=True)

        return df

class Size:

    def __init__(self, df):
        self.df = df

    def LNCAP(self):
        df = self.df.copy()
        df['LNCAP'] = np.log(df['S_VAL_MV'])
        return df

    def NLSIZE(self):
        df = self.LNCAP()






if __name__ == '__main__':
    risk_free = GetData.risk_free()
    risk_free.to_parquet('/Volumes/quanyi4g/factor/day_frequency/fundamental/RiskFree/risk_free.parquet')
