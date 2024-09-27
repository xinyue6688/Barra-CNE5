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
from Utils.data_clean import DataProcess

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

    @staticmethod
    def turnover_all():
        sql = f'''
            SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_TURN
            FROM ASHAREEODDERIVATIVEINDICATOR
            WHERE TRADE_DT BETWEEN '{START_DATE}' AND '{END_DATE}'
        '''
        connection = ConnectDatabase(sql)
        df = connection.get_data()
        return df

    @staticmethod
    def est_consensus():
        sql = f'''
            SELECT S_INFO_WINDCODE, EST_DT, NET_PROFIT_AVG, S_EST_YEARTYPE
            FROM ASHARECONSENSUSDATA
            WHERE (EST_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
            AND (EST_REPORT_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by = 'EST_DT', inplace = True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def est_consensus_rolling_FY1():
        sql = f'''
                SELECT S_INFO_WINDCODE, EST_DT, EST_PE
                FROM ASHARECONSENSUSROLLINGDATA
                WHERE (EST_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
                AND (ROLLING_TYPE = 'FY1')
            '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='EST_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def est_consensus_rolling_FY0(wind_code):
        sql = f'''
                SELECT S_INFO_WINDCODE, EST_DT, EST_PE
                FROM ASHARECONSENSUSROLLINGDATA
                WHERE (EST_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
                AND (ROLLING_TYPE = 'FY0')
                AND (S_INFO_WINDCODE = '{wind_code}')
            '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='EST_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def ttmhis_all():
        sql = f'''
            SELECT S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, NET_PROFIT_TTM, NET_INCR_CASH_CASH_EQU_TTM
            FROM ASHARETTMHIS
            WHERE (ANN_DT BETWEEN '{START_DATE}' AND '{END_DATE}')
            AND (REPORT_PERIOD BETWEEN '{START_DATE}' AND '{END_DATE}')
            AND (REPORT_PERIOD LIKE '%1231' OR REPORT_PERIOD LIKE '%0630')
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.rename(columns = {'NET_PROFIT_TTM': 'EARNINGS_TTM',
                             'NET_INCR_CASH_CASH_EQU_TTM': 'CASH_EARNINGS_TTM'}, inplace = True)
        return df

    @staticmethod
    def financialid_all():
        sql = f'''
            SELECT S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, S_FA_GRPS, S_FA_EPS_BASIC
            FROM ASHAREFINANCIALINDICATOR
            WHERE (REPORT_PERIOD LIKE '%1231%')
            AND (REPORT_PERIOD BETWEEN '{START_DATE}' AND '{END_DATE}')
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by = 'ANN_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def consensus_factor():
        sql = f'''
         SELECT S_INFO_WINDCODE, TRADE_DT, S_WEST_NETPROFIT_FTM_CHG_1M, S_WEST_NETPROFIT_FTM_CHG_6M
         FROM CONCENSUSEXPECTATIONFACTOR
         WHERE TRADE_DT BETWEEN '{START_DATE}' AND '{END_DATE}'
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.rename(columns = {'S_WEST_NETPROFIT_FTM_CHG_1M': 'EGIBS_s',
                             'S_WEST_NETPROFIT_FTM_CHG_6M': 'EGIBS'}, inplace=True)
        return df


class Calculation:
    """
    计算工具类，包含常用的因子预处理、回归和加权计算方法
    """

    @staticmethod
    def _exp_weight(window: int, half_life: int) -> np.ndarray:
        """
        计算指数加权权重
        :param window: 滑动窗口大小 例: 252
        :param half_life: 半衰期 例: 63
        :return: 归一化后的权重数组
        """
        exp_weight = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_weight[::-1] / np.sum(exp_weight)

    @staticmethod
    def _winsorize(x: pd.Series) -> pd.Series:
        """
        去极值处理，使因子值在均值的3个标准差范围内
        :param x: 输入因子序列
        :return: 去极值后的因子序列
        """
        x = x.replace([np.inf, -np.inf], np.nan)
        mean = x.dropna().mean()
        std = x.dropna().std()
        winsorized = x.copy()
        winsorized[x < mean - 3 * std] = mean - 3 * std
        winsorized[x > mean + 3 * std] = mean + 3 * std
        return winsorized

    @staticmethod
    def _standardize(x: pd.Series, market_value: pd.Series) -> pd.Series:
        """
        市值加权标准化处理
        :param x: 输入因子序列 例: pd.Series
        :param market_value: 对应的市值序列 例: pd.Series
        :return: 标准化后的因子序列 例: pd.Series
        """
        if market_value.dtype != np.float64:
            market_value = market_value.astype(np.float64)
        x = x.replace([np.inf, -np.inf], np.nan)
        w_mean = np.sum(x.dropna() * market_value) / np.sum(market_value)
        std = x.dropna().std()
        standardized = (x - w_mean) / std
        return standardized

    def _preprocess(self, data: pd.DataFrame, factor_column: str) -> pd.DataFrame:
        """
        因子预处理函数，包含去极值和标准化步骤
        :param data: 输入的因子数据表
        :param factor_column: 需要处理的因子列名 例: 'LEVERAGE'
        :return: 预处理后的数据表
        """
        if data[f'{factor_column}'].dtype != np.float64:
            data[f'{factor_column}'] = data[f'{factor_column}'].astype('float')
        data[f'{factor_column}_wsr'] = data.groupby('TRADE_DT')[f'{factor_column}'].transform(lambda x: self._winsorize(x))
        data[f'{factor_column}_pp'] = data.groupby('TRADE_DT').apply(lambda g: self._standardize(g[f'{factor_column}_wsr'], g['S_VAL_MV'])).reset_index(level=0, drop=True)
        data[f'{factor_column}'] = data[f'{factor_column}_pp']
        data.drop(columns=[f'{factor_column}_wsr', f'{factor_column}_pp'], inplace=True)
        return data

    @staticmethod
    def _cumulative_range(x: pd.Series) -> float:
        """
        计算累积区间的范围，基于最近12个月的数据
        :param x: 输入的时间序列数据
        :return: 最大累积值与最小累积值的差值
        """
        T = np.arange(1, 13)
        cumulative_ranges = [x[-(t * 21):].sum() for t in T]
        return np.max(cumulative_ranges) - np.min(cumulative_ranges)

    @staticmethod
    def _weighted_regress(df: pd.DataFrame, weight = 1) -> tuple:
        """
        进行加权回归，计算Alpha和Beta
        :param df: 输入数据表，需包含‘STOCK_RETURN’, ‘RF_RETURN’, ‘MKT_RETURN’等列
        :param weight: 回归权重 例: 1.0
        :return: 回归得到的Alpha和Beta值
        """
        y = df['STOCK_RETURN'] - df['RF_RETURN']
        X = df[['CONSTANT', 'MKT_RETURN']]
        model = sm.WLS(y, X, weights=weight).fit()
        alpha, beta = model.params.iloc[0], model.params.iloc[1]
        return alpha, beta

    @staticmethod
    def _regress_w_time(x, n_time):
        """
        随时间进行回归，返回时间斜率与因子的平均值比率
        :param x: 输入的时间序列数据
        :param n_time: 用于回归的时间段数 例: 5
        :return: 回归的斜率值与因子均值的比率
        """
        if len(x) < n_time:
            return np.nan
        else:
            T = np.arange(1, n_time + 1, 1)
            T = sm.add_constant(T)
            model = sm.OLS(x, T).fit()
        if np.sum(x) != 0:
            return model.params.iloc[1] / np.mean(x)
        else:
            return model.params.iloc[1] / np.mean(x[1:])


class Beta(Calculation):
    """
    计算个股市场暴露（Beta值）的类，继承自 Calculation 类，提供个股与市场回报的回归分析
    """
    def __init__(self, df):
        """
        初始化 Beta 类实例，加载无风险收益率和市场指数数据，并计算初始 Beta 值

        :param df: 包含个股交易数据的 DataFrame，需包含以下列:
                    - 'TRADE_DT': 交易日期 例: '2022-08-31'
                    - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                    - 'S_DQ_CLOSE': 收盘价 例: 10.5
                    - 'S_DQ_PRECLOSE': 前一日收盘价 例: 10.0
        """
        self.rf_df = GetData.risk_free()
        self.csi_df = GetData.mkt_index('000985.CSI')
        self.csi_df['TRADE_DT'] = pd.to_datetime(self.csi_df['TRADE_DT'])
        self.csi_df['MKT_RETURN'] = self.csi_df['S_DQ_CLOSE'] / self.csi_df['S_DQ_PRECLOSE'] - 1
        self.csi_df['MKT_RETURN'] = self.csi_df['MKT_RETURN'].astype(float)
        self.beta_df = self.BETA(df)

    def BETA(self, df):
        """
        计算每个个股的Alpha和Beta值，并将结果存储回DataFrame中。

        :param df: 包含个股交易数据的 DataFrame，需包含以下列:
                    - 'TRADE_DT': 交易日期 例: '2022-08-31'
                    - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                    - 'S_DQ_CLOSE': 收盘价 例: 10.5
                    - 'S_DQ_PRECLOSE': 前一日收盘价 例: 10.0
        :return: 包含新增列 'ALPHA', 'BETA', 'SIGMA' 的 DataFrame
        """
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
    """
    计算个股动量因子的类，继承自 Calculation 类
    """

    def RSTR(self, df: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
        """
        计算个股的RSTR

        :param df: 包含个股交易数据的 DataFrame，需包含以下列:
                   - 'TRADE_DT': 交易日期 例: '2022-08-31'
                   - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                   - 'S_DQ_CLOSE': 收盘价 例: 10.5
                   - 'S_DQ_PRECLOSE': 前一日收盘价 例: 10.0
        :param raw: 是否返回未经预处理的原始数据，默认为 False，即返回预处理后的数据
        :return: 返回包含新增列 'RSTR' 的 DataFrame
        """

        # 检查无风险收益率数据并合并
        if 'RF_RETURN' not in df.columns:
            rf_df = GetData.risk_free()
            df = df.merge(rf_df, on='TRADE_DT', how='left')

        # 计算个股日收益率
        df['STOCK_RETURN'] = df['S_DQ_CLOSE'] / df['S_DQ_PRECLOSE'] - 1
        df['STOCK_RETURN'] = df['STOCK_RETURN'].astype('float')

        # 生成指数加权权重
        exp_weight = self._exp_weight(window=504, half_life=126)

        # 按个股代码分组
        grouped = df.groupby('S_INFO_WINDCODE')

        for stock_code, group in grouped:
            # 检查数据长度是否足够计算动量因子
            if len(group) < 504 + 21:
                print(f'Not enough data to calculate Momentum for {stock_code}')
                df.loc[group.index, 'RSTR'] = np.nan

            else:
                # 计算超额对数收益率
                group['EX_LG_RT'] = np.log(1 + group['STOCK_RETURN']) - np.log(1 + group['RF_RETURN'])

                try:
                    # 使用指数加权和滚动窗口计算 RSTR
                    group['RSTR'] = group['EX_LG_RT'].shift(21).rolling(window=504).apply(
                        lambda x: np.sum(x * exp_weight))

                    # 将计算结果存入原始数据框
                    df.loc[group.index, 'RSTR'] = group['RSTR']

                except Exception as e:
                    print(f'Error processing {stock_code}: {e}')
                    df.loc[group.index, 'RSTR'] = np.nan

        # 如果 raw 为 False，则进行预处理
        if not raw:
            df = self._preprocess(df, factor_column='RSTR')

        return df


class Size(Calculation):
    """
    计算个股市值相关因子的类，包括对数市值（LNCAP）和非线性市值因子（NLSIZE）。
    """

    def LNCAP(self, df, raw = False):
        """
        计算对数市值因子（LNCAP），并根据需要进行预处理。

        :param df: 包含股票市值数据的 DataFrame，需包含 'S_VAL_MV' 列。
                    - 'S_VAL_MV': 股票市值 例: 10000000 (市值单位需为浮点数)
        :param raw: 是否返回未经预处理的原始数据，默认为 False。
        :return: 包含 'LNCAP' 因子的 DataFrame。
        """

        if df['S_VAL_MV'].dtype != np.float64:
            df['S_VAL_MV'] = df['S_VAL_MV'].astype(np.float64)

        df['LNCAP'] = np.log(df['S_VAL_MV'])

        if not raw:
            df = self._preprocess(df, factor_column = 'LNCAP')

        return df

    def NLSIZE(self, df, raw = False):
        """
        计算非线性市值因子（NLSIZE），通过回归对数市值的三次方与常数项，并返回残差作为因子。

        :param df: 包含股票市值数据的 DataFrame，需包含 'S_VAL_MV' 列。
        :param raw: 是否返回未经预处理的原始数据，默认为 False。
        :return: 包含 'NLSIZE' 因子的 DataFrame。
        """
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

class EarningsYield(GetData, Calculation):
    """
    计算收益率因子类，继承自 GetData 类，用于获取所需的基础数据。
    """

    def __init__(self, df):
        """
        初始化 EarningsYield 类实例，准备数据以供后续计算。

        :param df: 包含股票交易数据的 DataFrame，需包含以下列：
                    - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                    - 'TRADE_DT': 交易日期 例: '2022-08-31'
                    - 其他相关列，如 'S_VAL_MV'
        """
        self.prepared_df = self._prepare_data(df)

    def _prepare_data(self, df):
        """
        准备和处理数据，合并估值和收益数据，为计算收益率因子做准备。

        :param df: 包含股票交易数据的 DataFrame，需包含以下列：
                    - 'S_INFO_WINDCODE': 个股代码 例: '000001.SZ'
                    - 'TRADE_DT': 交易日期 例: '2022-08-31'
                    - 其他相关列，如 'S_VAL_MV'、'EST_PE' 等
        :return: 处理后的 DataFrame，包含估值和收益数据，准备进行收益率因子的计算。
        """

        epibs_df = GetData.est_consensus_rolling_FY1()
        epibs_df = epibs_df[~epibs_df['S_INFO_WINDCODE'].str.endswith('BJ')]
        epibs_df['EPIBS'] = 1 / epibs_df['EST_PE']

        grouped = epibs_df.groupby('S_INFO_WINDCODE')
        for stock_code, group in grouped:
            if group['EPIBS'].isna().all():
                epibs_df = epibs_df.drop(group.index)

                epibs_fy0_df = GetData.est_consensus_rolling_FY0(stock_code)
                epibs_fy0_df['EPIBS'] = 1 / epibs_fy0_df['EST_PE']

                epibs_df = pd.concat([epibs_df, epibs_fy0_df], ignore_index=True)

        epibs_df.sort_values(by='EST_DT', inplace=True)
        epibs_df['EPIBS'] = epibs_df.groupby('S_INFO_WINDCODE')['EPIBS'].fillna(method='ffill')
        epibs_df['EST_DT'] = pd.to_datetime(epibs_df['EST_DT'])
        df = pd.merge(
            df,
            epibs_df,
            left_on = ['S_INFO_WINDCODE', 'TRADE_DT'],
            right_on = ['S_INFO_WINDCODE', 'EST_DT']
        )

        earnings_df = GetData.ttmhis_all()
        earnings_df.sort_values(by='ANN_DT', inplace=True)
        earnings_df['ANN_DT'] = pd.to_datetime(earnings_df['ANN_DT'])
        earnings_df['REPORT_PERIOD'] = pd.to_datetime(earnings_df['REPORT_PERIOD'])
        earnings_df = earnings_df[(earnings_df['ANN_DT'] - earnings_df['REPORT_PERIOD']) < pd.Timedelta(days=365)]
        earnings_df.reset_index(drop=True, inplace=True)

        df = df.sort_values(by='TRADE_DT')
        df = pd.merge_asof(
            df,
            earnings_df,
            by='S_INFO_WINDCODE',
            left_on='TRADE_DT',
            right_on='ANN_DT',
            direction='backward'
        )

        df['EARNINGS_TTM'] = df.groupby('S_INFO_WINDCODE')['EARNINGS_TTM'].fillna(method='ffill')
        df['CASH_EARNINGS_TTM'] = df.groupby('S_INFO_WINDCODE')['CASH_EARNINGS_TTM'].fillna(method='ffill')

        return df

    def EARNYILD(self, raw: bool = False):
        """
        计算收益率因子（EARNYILD），基于EARNINGS_TTM、CASH_EARNINGS_TTM和EPIBS的加权组合。

        :param raw: 是否返回未经预处理的原始数据，默认为 False，即返回预处理后的数据。
        :return: 返回包含新增列 'EARNYILD' 的 DataFrame。
        """
        df = self.prepared_df
        df['ETOP'] = df['EARNINGS_TTM'] / df['S_VAL_MV']
        df['CETOP'] = df['CASH_EARNINGS_TTM'] / df['S_VAL_MV']

        for columns in ['EPIBS', 'ETOP', 'CETOP']:
             df[columns] = df[columns].astype('float64')

        df['EARNYILD'] = 0.68 * df['EPIBS'] + 0.11 * df['ETOP'] + 0.21 * df[
            'CETOP']

        if not raw:
            df = self._preprocess(df, factor_column='EARNYILD')

        return df


class ResidualVolatility(Calculation):
    """
    计算残差波动率的类，包含多种波动率因子的计算：
    1. DASTD - 日波动率标准差
    2. CMRA - 累积范围测度
    3. HSIGMA - 残差波动率的指数平滑方差
    4. RESVOL - 综合波动率因子

    :param df: 包含股票数据的 DataFrame, 需要包含以下列:
                - 'S_INFO_WINDCODE': 股票代码
                - 'TRADE_DT': 交易日期
                - 'S_DQ_CLOSE': 收盘价
                - 'S_DQ_PRECLOSE': 前收盘价
                - 'RF_RETURN': 无风险收益 (在 CMRA 中需要)
    :return: 包含计算后波动率因子的 DataFrame
    """

    def DASTD(self, df):
        """
        计算日波动率标准差 (DASTD)

        :param df: 包含股票收盘价的 DataFrame
        :return: 计算后的 DASTD 列
        """
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
        """
        计算累积范围测度 (CMRA)

        :param df: 包含股票收盘价和无风险收益的 DataFrame
        :return: 计算后的 CMRA 列
        """
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
        """
        计算 12 个月滚动的最大值与最小值差

        :param x: 包含每月收益的序列
        :return: 12 个月累计范围差
        """
        T = np.arange(1, 13)
        cumulative_ranges = [x[-(t * 21):].sum() for t in T]
        return np.max(cumulative_ranges) - np.min(cumulative_ranges)

    def HSIGMA(self, df):
        """
        计算指数平滑残差波动率 (HSIGMA)

        :param df: 包含股票收益率残差的 DataFrame
        :return: 计算后的 HSIGMA 列
        """
        df['HSIGMA'] = np.nan
        grouped = df.groupby('S_INFO_WINDCODE')

        for stock_code, group in grouped:
            if len(group) < 252:
                print(f'Not enough data to calculate HSIGMA for {stock_code}')
                continue
            group['HSIGMA'] = group['SIGMA'].ewm(halflife=63, span = None, min_periods = 252).std()
            df.loc[group.index, 'HSIGMA'] = group['HSIGMA']

        return df['HSIGMA']
    
    def RESVOL(self, df):
        """
        计算综合残差波动率因子 (RESVOL)

        :param df: 包含 DASTD, CMRA, HSIGMA 和 BETA 的 DataFrame
        :return: 计算后的 RESVOL 列
        """
        df['DASTD'] = self.DASTD(df)
        df = self._preprocess(df, factor_column='DASTD')
        df['CMRA'] = self.CMRA(df)
        df = self._preprocess(df, factor_column='CMRA')
        df['HSIGMA'] = self.HSIGMA(df)
        df = self._preprocess(df, factor_column='HSIGMA')

        df['RESVOL'] = 0.74 * df['DASTD'] + 0.16 * df['CMRA'] + 0.1 * df['HSIGMA']

        # 按日期升序排列
        df.sort_values(by = 'TRADE_DT', inplace = True)
        # 按标的分组向后填充空值
        df['RESVOL'] = df.groupby('S_INFO_WINDCODE')['RESVOL'].ffill()
        df['BETA'] = df.groupby('S_INFO_WINDCODE')['BETA'].ffill()
        # 截面平均填充
        df['RESVOL'] = df.groupby('TRADE_DT')['RESVOL'].transform(lambda x: x.fillna(x.mean()))
        df['BETA'] = df.groupby('TRADE_DT')['BETA'].transform(lambda x: x.fillna(x.mean()))
        # 去除剩余空值
        df = df.dropna(subset = ['RESVOL', 'BETA'])

        df['CONSTANT'] = 1
        orth_function = sm.OLS(df['RESVOL'], df[['BETA', 'CONSTANT']]).fit()
        df['RESVOL'] = orth_function.resid
        df = self._preprocess(df, factor_column = 'RESVOL')

        return df


class growth(GetData, Calculation):
    """
    计算股票增长因子的类，基于销售增长率（SGRO）和每股收益增长率（EGRO）的回归结果计算综合增长因子 GROWTH。

    :param df: 股票数据的 DataFrame，需包含以下列：
                - 'S_INFO_WINDCODE': 股票代码
                - 'TRADE_DT': 交易日期
    """

    def __init__(self, df):
        """
        初始化 Growth 类，获取所有股票的财务指标数据，并处理缺失值。

        :param df: 股票数据的 DataFrame
        """
        self.df = df
        self.growth_df = GetData.financialid_all()
        # 只保留财务日期和公告日期相差小于一年的记录
        self.growth_df['ANN_DT'] = pd.to_datetime(self.growth_df['ANN_DT'])
        self.growth_df['REPORT_PERIOD'] = pd.to_datetime(self.growth_df['REPORT_PERIOD'])
        self.growth_df = self.growth_df[(self.growth_df['ANN_DT'] - self.growth_df['REPORT_PERIOD']) < pd.Timedelta(days=365)]
        # 处理空值，先向后填充再截面均值填充
        self.growth_df['S_FA_GRPS'] = self.growth_df.groupby('S_INFO_WINDCODE')['S_FA_GRPS'].ffill()
        self.growth_df['S_FA_GRPS'] = self.growth_df.groupby('REPORT_PERIOD')['S_FA_GRPS'].transform(
            lambda x: x.fillna(x.mean()))
        self.growth_df['S_FA_EPS_BASIC'] = self.growth_df.groupby('S_INFO_WINDCODE')['S_FA_EPS_BASIC'].ffill()
        self.growth_df['S_FA_EPS_BASIC'] = self.growth_df.groupby('REPORT_PERIOD')['S_FA_EPS_BASIC'].transform(
            lambda x: x.fillna(x.mean()))
        self.growth_df['ANN_DT'] = pd.to_datetime(self.growth_df['ANN_DT'], errors='coerce')
        self.growth_df['ANN_DT'] = self.growth_df.groupby('REPORT_PERIOD')['ANN_DT'].transform(
            lambda x: x.fillna(x.median()))

    def GROWTH(self, raw: bool = False):
        """
        计算增长因子 GROWTH = 0.47 * SGRO + 0.24 * EGRO。
        其中：
        - SGRO = 销售增长率，基于 S_FA_GRPS 数据
        - EGRO = 每股收益增长率，基于 S_FA_EPS_BASIC 数据

        :return: 包含计算出的 GROWTH 因子的 DataFrame
        """

        self.growth_df['SGRO'] = np.nan
        self.growth_df['EGRO'] = np.nan
        grouped = self.growth_df.groupby('S_INFO_WINDCODE')
        for stock_code, group in grouped:
            if len(group) < 5:
                continue

            self.growth_df.loc[group.index, 'SGRO'] = group['S_FA_GRPS'].rolling(window=5).apply(
                lambda x: Calculation._regress_w_time(x, 5), raw=False)

            self.growth_df.loc[group.index, 'EGRO'] = group['S_FA_EPS_BASIC'].rolling(window=5).apply(
                lambda x: Calculation._regress_w_time(x, 5), raw=False)

        self.df = self.df.sort_values(by = 'TRADE_DT')
        self.growth_df = self.growth_df.sort_values(by = 'ANN_DT')

        self.df = pd.merge_asof(
            self.df,
            self.growth_df,
            by='S_INFO_WINDCODE',
            left_on='TRADE_DT',
            right_on='ANN_DT',
            direction='backward'
        )

        # 去除空值（年报数据不足五年的标的）
        self.df.dropna(how = 'any', inplace = True)

        self.df = self._preprocess(data=self.df, factor_column='SGRO')
        self.df = self._preprocess(data=self.df, factor_column='EGRO')

        self.df['GROWTH'] = 0.47 * self.df['SGRO'] + 0.24 * self.df['EGRO']

        if not raw:
            self.df = self._preprocess(data=self.df, factor_column='GROWTH')
        self.df['GROWTH'] = self.df.groupby('S_INFO_WINDCODE')['GROWTH'].ffill()

        return self.df


class btop(GetData, Calculation):
    """
    计算账面市值比 (BTOP)

    :param df: 包含股票信息的 DataFrame，需包含以下列:
                - 'S_INFO_WINDCODE': 股票代码
                - 'TRADE_DT': 交易日期
    :return: 添加了 BTOP 因子的 DataFrame
    """

    def __init__(self):
        """
        初始化 BTOP 类，载入市净率 (PB) 数据。
        """
        self.pb_df = GetData.pb_ratio_all()
        self.pb_df['TRADE_DT'] = pd.to_datetime(self.pb_df['TRADE_DT'])

    def BTOP(self, df):
        """
        计算账面市值比 (BTOP)

        :param df: 包含股票数据的 DataFrame
        :return: 添加了 BTOP 列并删除市净率 (S_VAL_PB_NEW) 列的 DataFrame
        """
        df = df.merge(self.pb_df, on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')
        df['BTOP'] = 1/df['S_VAL_PB_NEW']
        df['BTOP'] = df['BTOP'].astype('float')
        df.drop(columns=['S_VAL_PB_NEW'], inplace=True)
        df = self._preprocess(df, factor_column = 'BTOP')
        df['BTOP'] = df.groupby('S_INFO_WINDCODE')['BTOP'].ffill()

        return df


class leverage(GetData, Calculation):
    """
    计算杠杆率因子的类，包含 MLEV（市场杠杆）、DTOA（总债务/总资产）、BLEV（账面杠杆）和综合杠杆因子 LEVERAGE。

    :param df: 股票数据的 DataFrame，需包含以下列：
                - 'S_INFO_WINDCODE': 股票代码
                - 'TRADE_DT': 交易日期
    """

    def __init__(self, df):
        """
        初始化 Leverage 类，载入资产负债表数据、净资产数据、优先股股数数据并预处理。

        :param df: 股票数据的 DataFrame，按交易日期排序
        """
        self.balance_data = GetData.balancesheet_data()
        self.ncapital_data = GetData.capital()
        self.bppershare_data = GetData.bps()
        self.df = df.sort_values('TRADE_DT')
        self._prep_data()
        
        
    def _prep_data(self):
        """
        数据预处理：
        - 合并资产负债表、每股净资产、优先股股数数据，匹配最近披露的数值
        - 计算优先股股数、账面价值等指标
        """
        self.balance_data['ANN_DT'] = pd.to_datetime(self.balance_data['ANN_DT'])
        self.balance_data = self.balance_data.sort_values('ANN_DT')
        self.df = pd.merge_asof(
            self.df,
            self.balance_data,
            by='S_INFO_WINDCODE',
            left_on='TRADE_DT',
            right_on='ANN_DT',
            direction='backward'
        )

        # 每股净资产
        self.bppershare_data.dropna(how='any', inplace=True)
        self.bppershare_data['ANN_DT'] = pd.to_datetime(self.bppershare_data['ANN_DT'])
        self.bppershare_data = self.bppershare_data.sort_values('ANN_DT')

        self.df = pd.merge_asof(
            self.df,
            self.bppershare_data,
            by='S_INFO_WINDCODE',
            left_on='TRADE_DT',
            right_on='ANN_DT',
            direction='backward'
        )

        # 优先股股数（单位万）
        self.ncapital_data.dropna(how='any', inplace=True)
        self.ncapital_data['CHANGE_DT'] = pd.to_datetime(self.ncapital_data['CHANGE_DT'])
        self.ncapital_data = self.ncapital_data.sort_values('CHANGE_DT')

        self.df = pd.merge_asof(
            self.df,
            self.ncapital_data,
            by='S_INFO_WINDCODE',
            left_on='TRADE_DT',
            right_on='CHANGE_DT',
            direction='backward'
        )

        # 删除不需要的日期标注列
        self.df = self.df.drop(columns=['ANN_DT_x', 'ANN_DT_y', 'CHANGE_DT'])
        self.df['PE'] = self.df['S_FA_BPS'] * self.df['S_SHARE_NTRD_PRFSHARE']
        self.df['ME'] = self.df['S_VAL_MV'].copy()
        self.df['BE'] = (self.df['ME'] / self.df['S_DQ_CLOSE']) * self.df['S_FA_BPS']
        self.df.drop(columns=['S_FA_BPS', 'S_SHARE_NTRD_PRFSHARE'], inplace=True)
        
    def MLEV(self):
        """
        计算市场杠杆率 (MLEV) = (市值 + 优先股价值 + 长期负债) / 市值

        :return: 更新后的 DataFrame，包含 MLEV 因子
        """
        self.df['MLEV'] = (self.df['ME'] + self.df['PE'] + self.df['LD']) / self.df['ME']
        self.df = self._preprocess(self.df, 'MLEV')
        
    def DTOA(self):
        """
        计算总债务/总资产比率 (DTOA) = 总债务 / 总资产

        :return: 更新后的 DataFrame，包含 DTOA 因子
        """
        self.df['TD'] = self.df['TD'].astype('float')
        self.df['TA'] = self.df['TA'].astype('float')
        self.df['DTOA'] = self.df['TD'] / self.df['TA']
        self.df = self._preprocess(self.df, 'DTOA')

    def BLEV(self):
        """
        计算账面杠杆率 (BLEV) = (账面价值 + 优先股价值 + 长期负债) / 账面价值

        :return: 更新后的 DataFrame，包含 BLEV 因子
        """
        self.df['BE'] = self.df['BE'].astype('float')
        self.df['PE'] = self.df['PE'].astype('float')
        self.df['LD'] = self.df['LD'].astype('float')

        self.df['BLEV'] = (self.df['BE'] + self.df['PE'] + self.df['LD']) / \
                                   self.df['BE']
        self.df = self._preprocess(data=self.df, factor_column='BLEV')

    def LEVERAGE(self):
        """
        计算综合杠杆率 (LEVERAGE) = 0.38 * MLEV + 0.35 * DTOA + 0.27 * BLEV
        并进行缺失值处理

        :return: 更新后的 DataFrame，包含 LEVERAGE 因子
        """
        self.MLEV()
        self.DTOA()
        self.BLEV()
        self.df['LEVERAGE'] = 0.38 * self.df['MLEV'] + 0.35 * self.df['DTOA'] + 0.27 * self.df['BLEV']
        self.df = self._preprocess(self.df, 'LEVERAGE')
        self.df['LEVERAGE'] = self.df.groupby('S_INFO_WINDCODE')['LEVERAGE'].ffill()
        self.df['LEVERAGE'] = self.df.groupby('TRADE_DT')['LEVERAGE'].transform(lambda x: x.fillna(x.mean()))

        return self.df


class liquidity(GetData, Calculation):
    """
    计算流动性因子的类，基于交易量和换手率计算 STOM（一个月换手率）、STOQ（三个月换手率）、
    STOA（一年换手率）并综合为流动性因子 LIQUIDITY。

    :param df: 股票数据的 DataFrame，需包含以下列：
                - 'S_INFO_WINDCODE': 股票代码
                - 'TRADE_DT': 交易日期
    """

    def __init__(self, df):
        """
        初始化 Liquidity 类，获取全部股票的换手率数据并与传入的 DataFrame 合并。

        :param df: 股票数据的 DataFrame
        """
        self.turnover_df = GetData.turnover_all()
        self.turnover_df['TRADE_DT'] = pd.to_datetime(self.turnover_df['TRADE_DT'])
        self.all_mkt_turnover = pd.merge(df, self.turnover_df, on = ['S_INFO_WINDCODE', 'TRADE_DT'], how = 'left')
        self.all_mkt_turnover = self.all_mkt_turnover.sort_values(by='TRADE_DT')

    def LIQUIDITY(self):
        """
        计算流动性因子 LIQUIDITY = 0.35 * STOM + 0.35 * STOQ + 0.30 * STOA。
        其中：
        - STOM = 最近 21 天的换手率之和的对数值
        - STOQ = 最近 63 天的换手率之和的对数值
        - STOA = 最近 252 天的换手率之和的对数值

        :return: 包含计算出的 LIQUIDITY 因子的 DataFrame
        """
        grouped = self.all_mkt_turnover.groupby('S_INFO_WINDCODE').filter(lambda x: len(x) >= 252)
        grouped['STOM'] = grouped.groupby('S_INFO_WINDCODE')['S_DQ_TURN'].rolling(window=21).sum().apply(
            lambda x: np.log(x)).reset_index(level=0, drop=True)
        grouped['STOQ'] = grouped.groupby('S_INFO_WINDCODE')['S_DQ_TURN'].rolling(window=63).sum().apply(
            lambda x: np.log(1 / 3 * x)).reset_index(level=0, drop=True)
        grouped['STOA'] = grouped.groupby('S_INFO_WINDCODE')['S_DQ_TURN'].rolling(window=252).sum().apply(
            lambda x: np.log(1 / 12 * x)).reset_index(level=0, drop=True)

        grouped['LIQUIDITY'] = 0.35 * grouped['STOM'] + 0.35 * grouped['STOQ'] + 0.3 * grouped['STOA']

        result = grouped.groupby('S_INFO_WINDCODE').apply(lambda x: x.iloc[251:]).reset_index(drop=True)
        result = result[~result['LIQUIDITY'].isna()]

        df = self._preprocess(data=result, factor_column='LIQUIDITY')
        df['LIQUIDITY'] = df.groupby('S_INFO_WINDCODE')['LIQUIDITY'].ffill()
        df = df[~df['LIQUIDITY'].isna()]

        return df



if __name__ == '__main__':
    factor_exposure = pd.read_parquet('/Volumes/quanyi4g/factor/day_frequency/barra/factor_exposure.parquet')

    last_updated_date = factor_exposure['TRADE_DT'].unique().max()
    todays_date = datetime.now().strftime('%Y%m%d')

    cn_trade_calender = mcal.get_calendar('XSHG')
    trade_day = cn_trade_calender.schedule(pd.Timestamp('2007-06-01'), todays_date)

    START_DATE = trade_day.index[-525].strftime('%Y%m%d')
    END_DATE = todays_date

    # 获取最新的价格
    all_market_new_price = GetData.all_price()
    # 获取最新的市值
    all_market_new_mv = GetData.all_mv()
    # 合并到一个数据框
    all_market_new_data = pd.merge(all_market_new_price, all_market_new_mv, on = ['TRADE_DT', 'S_INFO_WINDCODE'],\
                                   how = 'left')
    # 去除北交所
    all_market_new_data = all_market_new_data[~all_market_new_data['S_INFO_WINDCODE'].str.endswith('BJ')]
    # 分配行业
    all_market_new_data = DataProcess.assign_industry(all_market_new_data)

    # 更新BETA
    calculate_beta = Beta(all_market_new_data)
    beta_df = calculate_beta.beta_df

    # 更新BTOP
    bp_calculator = btop()
    btop_df = bp_calculator.BTOP(all_market_new_data)

    # 合并beta和btop
    new_factor_exposure = pd.merge(beta_df, btop_df[['S_INFO_WINDCODE', 'TRADE_DT', 'BTOP']], on = ['S_INFO_WINDCODE', 'TRADE_DT'], how = 'left')
    # 添加国家因子（常数项）
    new_factor_exposure['COUNTRY'] = 1

    # 更新残差波动率RESVOL
    calculate_resvol = ResidualVolatility()
    resvol_df = calculate_resvol.RESVOL(new_factor_exposure)
    new_factor_exposure = pd.merge(new_factor_exposure, resvol_df[['S_INFO_WINDCODE', 'TRADE_DT','RESVOL']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')

    # 更新动量RSTR
    calculate_mom = Momentum()
    rstr_df = calculate_mom.RSTR(all_market_new_data)
    # 合并到新因子暴露数据框
    new_factor_exposure = pd.merge(new_factor_exposure, rstr_df[['S_INFO_WINDCODE', 'TRADE_DT','RSTR']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how = 'left')

    # 更新市值LNCAP
    size_calculator = Size()
    lncap_df = size_calculator.LNCAP(all_market_new_data)
    # 合并到新因子暴露数据框
    new_factor_exposure = pd.merge(new_factor_exposure, lncap_df[['S_INFO_WINDCODE', 'TRADE_DT','LNCAP']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how = 'left')

    # 更新收益因子EARNYILD
    ey_calculator = EarningsYield(all_market_new_data)
    ey_df = ey_calculator.EARNYILD()
    # 合并到新因子暴露数据框
    new_factor_exposure = pd.merge(new_factor_exposure, ey_df[['S_INFO_WINDCODE', 'TRADE_DT','EARNYILD']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how = 'left')

    # 更新成长因子GROWTH
    growth_calculator = growth(all_market_new_data)
    growth_df = growth_calculator.GROWTH()
    # 合并到新因子暴露数据框
    new_factor_exposure = pd.merge(new_factor_exposure, growth_df[['S_INFO_WINDCODE', 'TRADE_DT','GROWTH']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how = 'left')

    # 更新杠杆因子LEVERAGE
    leverage_calculator = leverage(all_market_new_data)
    leverage_df = leverage_calculator.LEVERAGE()
    # 合并到新因子暴露数据框
    new_factor_exposure = pd.merge(new_factor_exposure, leverage_df[['S_INFO_WINDCODE', 'TRADE_DT','LEVERAGE']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')

    # 更新流动性LIQUIDITY
    liquidity_calculator = liquidity(all_market_new_data)
    liquidity_df = liquidity_calculator.LIQUIDITY()
    new_factor_exposure = pd.merge(new_factor_exposure, liquidity_df[['S_INFO_WINDCODE', 'TRADE_DT','LIQUIDITY']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')

    # 更新非线性市值NLSIZE
    size_calculator = Size()
    nlsize_df = size_calculator.NLSIZE(all_market_new_data)
    new_factor_exposure = pd.merge(new_factor_exposure, nlsize_df[['S_INFO_WINDCODE', 'TRADE_DT','NLSIZE']], on=['S_INFO_WINDCODE', 'TRADE_DT'], how='left')

    # 保留需要更新的数据
    new_factor_exposure = new_factor_exposure[
        (new_factor_exposure['TRADE_DT'] > last_updated_date) &
        (new_factor_exposure['TRADE_DT'] <= pd.to_datetime(todays_date, format='%Y%m%d'))
        ]

    # 对齐原数据框的顺序
    new_factor_exposure = new_factor_exposure[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_PRECLOSE', 'S_DQ_CLOSE',
       'S_VAL_MV', 'WIND_PRI_IND', 'RF_RETURN', 'ALPHA', 'BETA', 'SIGMA',
       'COUNTRY', 'RSTR', 'LNCAP', 'EARNYILD', 'GROWTH', 'LEVERAGE',
       'LIQUIDITY', 'RESVOL', 'BTOP', 'NLSIZE']]

    # 更新因子暴露并存储到数据库
    factor_exposure_updated = pd.concat([factor_exposure, new_factor_exposure], axis=0, ignore_index=True)
    factor_exposure_updated.to_parquet('/Volumes/quanyi4g/factor/day_frequency/barra/factor_exposure(updated).parquet')












