# -*- coding = utf-8 -*-
# @Time: 2024/08/18
# @Author: Xinyue
# @File:barra_loadings.py
# @Software: PyCharm

import pandas as pd
from datetime import datetime, timedelta
import akshare as ak
import numpy as np
import statsmodels.api as sm

from Utils.get_wind_data_per_stock import DataPerStock
from Utils.connect_wind import ConnectDatabase

class Barra_Beta(DataPerStock):
    """
    Barra 计算类, 用于计算Barra定义下, 个股的市场暴露
    """

    def __init__(self, start_date, end_date):
        '''
        :param start_date: 起始时间 例：'20100101'
        :param end_date: 结束时间 例：datetime.now().strftime('%Y%m%d')
        '''
        self.start_date = start_date
        self.end_date = end_date
        self.weights = self.reg_weights()
        self.rf = None
        self.fixed_df = None
    def risk_free_rate(self):
        """
        获取无风险利率（十年国债回报率）
        :return: 无风险利率数据框 格式：日期，年化回报率 
        """
        current_df_start_time = datetime.strptime(self.start_date, "%Y%m%d")
        yield10yr_df = pd.DataFrame()

        while current_df_start_time.strftime("%Y%m%d") < self.end_date:
            current_df_end_time = min(current_df_start_time + timedelta(days=365), datetime.strptime(self.end_date, "%Y%m%d"))

            bond_china_yield_df = ak.bond_china_yield(
                start_date=current_df_start_time.strftime("%Y%m%d"),
                end_date=current_df_end_time.strftime("%Y%m%d")
            )

            filtered_df = bond_china_yield_df[
                (bond_china_yield_df['曲线名称'] == '中债国债收益率曲线')
            ][['日期', '10年']]

            yield10yr_df= pd.concat([yield10yr_df, filtered_df])

            current_df_start_time = current_df_end_time + timedelta(days=1)

        yield10yr_df.reset_index(drop=True, inplace=True)
        yield10yr_df['RF_RETURN_ANN'] = yield10yr_df['10年'] / 100
        yield10yr_df['TRADE_DT'] = pd.to_datetime(yield10yr_df['日期'])
        yield10yr_df['RF_RETURN'] = (1 + yield10yr_df['RF_RETURN_ANN']) ** (1 / 252) - 1
        self.rf = yield10yr_df[['TRADE_DT', 'RF_RETURN']]
        return self.rf

    def reg_weights(self):
        """
        计算Barra测量beta的数据权重
        :return: 升序权重(np.array)
        """
        window_size = 252
        half_life = 63
        lambda_ = np.exp(-np.log(2) / half_life)
        t = np.arange(1, window_size + 1)
        weights = lambda_ ** t
        weights = weights[::-1]
        return weights

    def prepare_rm_n_rf(self, csi_all, my_ew_index):
        """
        准备市场回报数据
        :param csi_all: 中证全指数据框
        :param my_ew_index: 万得全A等权指数数据框
        :return: 市场回报数据框
        """
        market_returns = pd.merge(csi_all[['TRADE_DT', 'CSI_RETURN']], my_ew_index[['TRADE_DT', 'EWI_RETURN']],
                                  on='TRADE_DT')
        market_returns = market_returns[['TRADE_DT', 'EWI_RETURN', 'CSI_RETURN']]

        self.fixed_df = pd.merge(self.rf, market_returns, on='TRADE_DT', how='left')
        return self.fixed_df

    def beta_exposure(self, stock_code):
        """
        计算市场暴露
        :param stock_code: 万得股票代码 例：'000001.SZ'
        :return: 个股市场暴露数据框
        """
        sql = f'''
            SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_PRECLOSE, S_DQ_CLOSE
            FROM ASHAREEODPRICES
            WHERE S_INFO_WINDCODE = '{stock_code}' AND TRADE_DT BETWEEN '{self.start_date}' AND '{self.end_date}'
        '''

        connection = ConnectDatabase(sql)
        price_df = connection.get_data()
        price_df.sort_values(by='TRADE_DT', inplace=True)
        price_df.reset_index(drop=True, inplace=True)

        if price_df[price_df['TRADE_DT'] > '2010-01-01'].empty:
            print(f'Stock {stock_code} is delisted before 2010')
            return price_df
        elif len(price_df) < 252:
            print(f'Unable to meet Barra requirements for {stock_code} because of the following reason:')
            print(f'Not enough data to calculate market exposure.')
            return price_df
        else:
            price_df[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = price_df[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
            price_df['STOCK_RETURN'] = price_df['S_DQ_CLOSE'] / price_df['S_DQ_PRECLOSE'] - 1
            price_df['TRADE_DT'] = pd.to_datetime(price_df['TRADE_DT'])
            price_df = price_df.merge(self.fixed_df, on='TRADE_DT', how='left')
            beta_results = []
            start_index = 252

        for index in range(start_index, len(price_df['TRADE_DT'])):
            window_data = price_df.iloc[max(0, index - 252): index].dropna()
            time = price_df.loc[index, 'TRADE_DT']
            reg_data = window_data[['S_INFO_WINDCODE', 'STOCK_RETURN', 'RF_RETURN', 'EWI_RETURN', 'CSI_RETURN']].copy()
            reg_data['y'] = reg_data['STOCK_RETURN'] - reg_data['RF_RETURN']
            reg_data['const'] = 1

            weights_adj = self.weights[-len(reg_data):] if len(reg_data) != len(self.weights) else self.weights

            try:
                EW_model = sm.WLS(reg_data['y'], reg_data[['const', 'EWI_RETURN']], weights=weights_adj).fit()
                EW_beta_t = EW_model.params['EWI_RETURN']
            except Exception as e:
                print(f"EW_model error: {e} at iteration {index}")
                continue

            try:
                CSI_model = sm.WLS(reg_data['y'], reg_data[['const', 'CSI_RETURN']], weights=weights_adj).fit()
                CSI_beta_t = CSI_model.params['CSI_RETURN']
            except Exception as e:
                print(f"CSI_model error: {e} at iteration {index}")
                continue

            beta_results.append({
                'TRADE_DT': time,
                'EW_Beta': EW_beta_t,
                'CSI_Beta': CSI_beta_t
            })

        beta_df = pd.DataFrame(beta_results)
        beta_df['TRADE_DT'] = pd.to_datetime(beta_df['TRADE_DT'])

        try:
            beta_df = beta_df.merge(price_df, on='TRADE_DT', how='left')
        except KeyError as e:
            print(f"Merge error for stock {stock_code}: {e}")

        beta_df['EW_Alpha'] = beta_df['STOCK_RETURN'] - beta_df['RF_RETURN'] - beta_df['EWI_RETURN'] * beta_df['EW_Beta']
        beta_df['CSI_Alpha'] = beta_df['STOCK_RETURN'] - beta_df['RF_RETURN'] - beta_df['CSI_RETURN'] * beta_df['CSI_Beta']
        beta_df = beta_df[['TRADE_DT', 'EW_Beta', 'CSI_Beta', 'EW_Alpha', 'CSI_Alpha']]
        beta_df['S_INFO_WINDCODE'] = stock_code

        return beta_df