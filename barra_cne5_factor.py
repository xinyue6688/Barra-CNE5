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

    @staticmethod
    def wind_index(index_code: str):
        fields_sql = 'S_INFO_WINDCODE , TRADE_DT, S_DQ_PCTCHANGE'
        table = 'AINDEXWINDINDUSTRIESEOD'

        sql = f'''SELECT {fields_sql}               
                     FROM {table}
                     WHERE (TRADE_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
                     AND (S_INFO_WINDCODE = '{index_code}')
                  '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def mkt_index(index_code: str):
        fields_sql = 'S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_CLOSE'

        sql = f'''SELECT {fields_sql}               
                             FROM AINDEXEODPRICES
                             WHERE (TRADE_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
                             AND (S_INFO_WINDCODE = '{index_code}')
                          '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def analyst_earnest(stock_code: str):
        fields_sql = 'S_INFO_WINDCODE, EST_DT, REPORTING_PERIOD, S_EST_ENDDATE, S_EST_PE'

        sql = f'''
                SELECT {fields_sql}
                FROM ASHAREEARNINGEST
                WHERE S_INFO_WINDCODE = '{stock_code}'
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='EST_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def pb_ratio_all():
        fields_sql = 'S_INFO_WINDCODE, TRADE_DT, S_VAL_PB_NEW'

        sql = f'''
                SELECT {fields_sql}
                FROM ASHAREEODDERIVATIVEINDICATOR
                WHERE TRADE_DT BETWEEN '{START_DATE}' AND '{END_DATE}'
                '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def balancesheet_data():
        fields_sql = 'S_INFO_WINDCODE, ANN_DT, LT_BORROW, TOT_LIAB, TOT_ASSETS'

        sql = f'''
                SELECT {fields_sql} 
                FROM ASHAREBALANCESHEET
                WHERE (ANN_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
                AND (REPORT_PERIOD BETWEEN '{START_DATE}' AND '{END_DATE}')
                AND (REPORT_PERIOD LIKE '%1231' OR REPORT_PERIOD LIKE '%0630')
                AND (STATEMENT_TYPE = '408001000')
                '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='ANN_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={'LT_BORROW': 'LD',
                           'TOT_LIAB': 'TD',
                           'TOT_ASSETS': 'TA'}, inplace=True)
        return df

    @staticmethod
    def capital():
        sql = f'''
                SELECT S_INFO_WINDCODE, CHANGE_DT, S_SHARE_NTRD_PRFSHARE
                FROM ASHARECAPITALIZATION 
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        return df

    @staticmethod
    def bps():
        sql = f'''
                SELECT S_INFO_WINDCODE, ANN_DT, S_FA_BPS
                FROM ASHAREFINANCIALINDICATOR
                WHERE REPORT_PERIOD BETWEEN '{START_DATE}' AND '{END_DATE}'
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
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
        data[f'{factor_column}_pp'] = data.groupby('TRADE_DT').apply(lambda g: self._standardize(g[f'{factor_column}_wsr'], g['S_VAL_MV'])).reset_index(level=0, drop=True)
        data[f'{factor_column}'] = data[f'{factor_column}_pp']
        data.drop(columns=[f'{factor_column}_wsr', f'{factor_column}_pp'], inplace=True)

        return data

    @staticmethod
    def _cumulative_range(x):
        T = np.arange(1, 13)
        cumulative_ranges = [x[-(t * 21):].sum() for t in T]
        return np.max(cumulative_ranges) - np.min(cumulative_ranges)

    @staticmethod
    def _weighted_regress(df, weight = 1):
        y = df['STOCK_RETURN'] - df['RF_RETURN']
        X = df[['CONSTANT', 'MKT_RETURN']]
        model = sm.WLS(y, X, weights=weight).fit()
        alpha, beta = model.params.iloc[0], model.params.iloc[1]
        return alpha, beta


class Beta(Calculation):

    def __init__(self, df):
        self.rf_df = pd.read_parquet('/Volumes/quanyi4g/factor/day_frequency/fundamental/RiskFree/risk_free.parquet')
        self.csi_df = GetData.mkt_index('000985.CSI')
        self.csi_df['TRADE_DT'] = pd.to_datetime(self.csi_df['TRADE_DT'])
        self.csi_df['MKT_RETURN'] = self.csi_df['S_DQ_CLOSE'] / self.csi_df['S_DQ_PRECLOSE'] - 1
        self.csi_df['MKT_RETURN'] = self.csi_df['MKT_RETURN'].astype(float)
        self.beta_df = self.BETA(df)

    def BETA(self, df):

        df['STOCK_RETURN'] = df['S_DQ_CLOSE'] / df['S_DQ_PRECLOSE'] - 1
        df['STOCK_RETURN'] = df['STOCK_RETURN'].astype('float')

        # 合并市场收益
        df = df.merge(self.csi_df[['TRADE_DT', 'MKT_RETURN']], on='TRADE_DT', how='left')
        # 合并无风险收益
        if 'RF_RETURN' not in df.columns:
            df = df.merge(self.rf_df, on='TRADE_DT', how='left')

        exp_weight = self._exp_weight(window=252, half_life=63)
        df['ALPHA'] = np.nan
        df['BETA'] = np.nan

        grouped = df.groupby('S_INFO_WINDCODE')

        for stock_code, group in grouped:
            if group[group['TRADE_DT'] > '2010-01-01'].empty:
                continue

            elif len(group) < 252:
                continue

            else:
                group['CONSTANT'] = 1
                alphas = []
                betas = []

                for i in range(251, len(group)):
                    window_data = group.iloc[i - 251:i + 1]
                    alpha, beta = self._weighted_regress(window_data, exp_weight)
                    alphas.append(alpha)
                    betas.append(beta)

                # Store the results, shifting by one period
                original_df_index = grouped.indices[f'{stock_code}']
                df.loc[original_df_index[251:], 'ALPHA'] = np.array(alphas)
                df.loc[original_df_index[251:], 'BETA'] = np.array(betas)
                df.loc[original_df_index, 'ALPHA'] = df.loc[original_df_index, 'ALPHA'].shift(1)
                df.loc[original_df_index, 'BETA'] = df.loc[original_df_index, 'BETA'].shift(1)

        df['SIGMA'] = df['STOCK_RETURN']- df['RF_RETURN'] - (df['ALPHA'] + df['BETA'] * df['MKT_RETURN'])

        return df




class Momentum(Calculation):


    def RSTR(self, df, raw = False):
        """
        :param df:
        :return:
        """
        if 'RF_RETURN' not in df.columns:
            rf_df = GetData.risk_free()
            df = df.merge(rf_df, on = 'TRADE_DT', how = 'left')

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

    def LNCAP(self, df, raw = False):
        if df['S_VAL_MV'].dtype != np.float64:
            df['S_VAL_MV'] = df['S_VAL_MV'].astype(np.float64)
        df['LNCAP'] = np.log(df['S_VAL_MV'])
        if not raw:
            df = self._preprocess(df, factor_column = 'LNCAP')
        return df

    def NLSIZE(self, df, raw = False):
        df = self.LNCAP(df)
        df['SIZE_CUBED'] = np.power(df['LNCAP'], 3)
        df['CONSTANT'] = 1
        df = df.dropna(how='any')
        X = df[['LNCAP','CONSTANT']]
        y = df['SIZE_CUBED']
        model = sm.OLS(y, X).fit()
        df['NLSIZE'] = model.resid
        if not raw:
            df = self._preprocess(df, factor_column = 'NLSIZE')

        df.drop(columns=['SIZE_CUBED', 'LNCAP', 'CONSTANT'], inplace=True)
        return df

#class EarningsYield(Calculation):
    # TODO: 找一下万得分析师预测的数据

class ResidualVolatility(Calculation):


    def DASTD(self, df):
        df['STOCK_RETURN'] = df['S_DQ_CLOSE'] / df['S_DQ_PRECLOSE'] - 1
        df['STOCK_RETURN'] = df['STOCK_RETURN'].astype('float')
        exp_weight = self._exp_weight(window = 252, half_life = 42)
        df['DASTD'] = np.nan

        grouped = df.groupby('S_INFO_WINDCODE')

        for stock_code, group in grouped:
            if len(group) < 252:
                print(f'Not enough data to calculate DASTD for {stock_code}')
                continue

            else:
                try:
                    group['DASTD'] = group['STOCK_RETURN'].rolling(window=252).apply(
                        lambda x: np.sum(np.std(x) * exp_weight))

                    df.loc[group.index, 'DASTD'] = group['DASTD']

                except Exception as e:
                    print(f'Error processing {stock_code}: {e}')
                    df.loc[group.index, 'DASTD'] = np.nan

        return df['DASTD']

    def CMRA(self, df):
        if 'RF_RETURN' not in df.columns:
            rf_df = GetData.risk_free()
            df = df.merge(rf_df, on = 'TRADE_DT', how = 'left')

        df['CMRA'] = np.nan
        grouped = df.groupby('S_INFO_WINDCODE')

        for stock_code, group in grouped:
            if len(group) < 252:
                print(f'Not enough data to calculate CMRA for {stock_code}')
                continue

            else:
                try:
                    group['ELR'] = np.log(1 + group['STOCK_RETURN']) - np.log(1 + group['RF_RETURN'])
                    group['CMRA'] = group['ELR'].rolling(window=252).apply(
                        lambda x: self._cumulative_range(x), raw=True)
                    df.loc[group.index, 'CMRA'] = group['CMRA']

                except Exception as e:
                    print(f'Error processing {stock_code}: {e}')
                    df.loc[group.index, 'CMRA'] = np.nan

        return df['CMRA']

    @staticmethod
    def _cumulative_range(x):
        T = np.arange(1, 13)
        cumulative_ranges = [x[-(t * 21):].sum() for t in T]
        return np.max(cumulative_ranges) - np.min(cumulative_ranges)

    def HSIGMA(self, df):
        df['HSIGMA'] = np.nan
        #exp_weight = self._exp_weight(window=252, half_life=62)
        grouped = df.groupby('S_INFO_WINDCODE')

        for stock_code, group in grouped:
            if len(group) < 252:
                print(f'Not enough data to calculate HSIGMA for {stock_code}')
                continue
            group['HSIGMA'] = group['SIGMA'].ewm(halflife=63, span = None, min_periods = 252).std()
            df.loc[group.index, 'HSIGMA'] = group['HSIGMA']

        return df['HSIGMA']
    
    def RESVOL(self, df):
        df['DASTD'] = self.DASTD(df)
        df['CMRA'] = self.CMRA(df)
        df['HSIGMA'] = self.HSIGMA(df)

        df['RESVOL'] = 0.74 * df['DASTD'] + 0.16 * df['CMRA'] + 0.1 * df['HSIGMA']

        df = df.dropna(subset = ['RESVOL', 'BETA'])
        df['CONSTANT'] = 1
        orth_function = sm.OLS(df['RESVOL'], df[['BETA', 'CONSTANT']]).fit()
        df['RESVOL'] = orth_function.resid
        df = self._preprocess(df, factor_column = 'RESVOL')

        return df


class btop(GetData):

    def __init__(self):
        self.pb_df = GetData.pb_ratio_all()
        self.pb_df['TRADE_DT'] = pd.to_datetime(self.pb_df['TRADE_DT'])

    def BTOP(self, df):
        df = df.merge(self.pb_df, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')
        df['BTOP'] = 1/df['S_VAL_PB_NEW']
        df['BTOP'] = df['BTOP'].astype('float')
        df.drop(columns=['S_VAL_PB_NEW'], inplace=True)
        return df


class leverage(GetData):

    def __init__(self):
        self.balance_data = GetData.balancesheet_data()
        self.ncapital_data = GetData.capital()
        self.bppershare_data = GetData.bps()




if __name__ == '__main__':
    all_market_mom = pd.read_parquet('/Users/xinyuezhang/Desktop/中量投/Project/ZLT-Project Barra/Data/all_market_data.parquet')
    risk_free = pd.read_parquet('/Volumes/quanyi4g/factor/day_frequency/fundamental/RiskFree/risk_free.parquet')
    all_market_mom = pd.merge(all_market_mom, risk_free, on = 'TRADE_DT', how = 'left')
    mom = Momentum()
    all_market_mom = mom.RSTR(all_market_mom)

