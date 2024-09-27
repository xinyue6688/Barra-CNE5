# -*- coding:UTF-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from decimal import Decimal

from Utils.get_wind_data import WindData

class DataProcess(WindData):
    """
    数据处理类，用于筛选指数成分股，并分配行业信息。
    """

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

        super().__init__(self.start_date, self.end_date)
        self.get_data = super()

    def filter_index_cons(self, stock_list, index_code):
        """
        筛选指数成分股数据
        :param stock_list: 股票数据表 (pd.DataFrame)
        :param index_code: 指数代码 (str, 例:'000852.SH')
        :return: 指数成分股数据表 (pd.DataFrame)
        """
        index_member = self.get_data.get_index_con(index_code)
        if 'S_INFO_WINDCODE' in index_member.columns:
            index_member.drop(columns=['S_INFO_WINDCODE'], inplace=True)
        else:
            index_member.drop(columns=['F_INFO_WINDCODE'], inplace = True)

        index_member.rename(columns={'S_CON_WINDCODE': 'S_INFO_WINDCODE'}, inplace=True)

        index_member['S_CON_INDATE'] = pd.to_datetime(index_member['S_CON_INDATE'].astype(str))
        index_member['S_CON_OUTDATE'] = pd.to_datetime(index_member['S_CON_OUTDATE'].astype(str),errors='coerce')

        start_date = sorted(index_member['S_CON_INDATE'].unique())[0]
        initiated_list = index_member[index_member['S_CON_INDATE'] == start_date]['S_INFO_WINDCODE']
        member_prices = pd.merge(stock_list, index_member, on='S_INFO_WINDCODE', how='left')
        member_prices = member_prices[
            ((member_prices['TRADE_DT'] >= member_prices['S_CON_INDATE']) &
            (member_prices['TRADE_DT'] <= member_prices['S_CON_OUTDATE'])) |
            (member_prices['S_CON_OUTDATE'].isna() & (member_prices['TRADE_DT'] >= member_prices['S_CON_INDATE'])) |
            ((member_prices['TRADE_DT'] < start_date) & (member_prices['S_INFO_WINDCODE'].isin(initiated_list)))
        ]
        before_date = member_prices[member_prices['TRADE_DT'] < start_date]
        before_date_unique = before_date.drop_duplicates(subset=['TRADE_DT', 'S_INFO_WINDCODE'])
        after_date = member_prices[member_prices['TRADE_DT'] >= start_date]
        member_prices = pd.concat([before_date_unique, after_date])
        member_prices.drop(columns=['S_CON_INDATE', 'S_CON_OUTDATE', 'CUR_SIGN'], inplace=True)
        return member_prices

    @staticmethod
    def assign_industry(stock_list):
        """
        分配行业信息
        :param stock_list: 股票数据表 格式：TRADE_DT(datetime64[ns]), S_INFO_WINDCODE(str)
        :return: 分配行业信息后的股票数据表
        """
        ashareind_df = pd.read_parquet('Data/wind_pri_industry_Ashare.parquet')
        stock_list_withind = pd.merge(stock_list, ashareind_df, on='S_INFO_WINDCODE', how='left')
        stock_list_withind = stock_list_withind[
            ((stock_list_withind['TRADE_DT'] >= stock_list_withind['ENTRY_DT']) &
             ((stock_list_withind['TRADE_DT'] <= stock_list_withind['REMOVE_DT']) | stock_list_withind['REMOVE_DT'].isna())) |
            (stock_list_withind['REMOVE_DT'].isna() & (stock_list_withind['TRADE_DT'] >= stock_list_withind['ENTRY_DT']))
        ]
        stock_list_withind.drop(columns=['ENTRY_DT', 'REMOVE_DT', 'CUR_SIGN'], inplace=True)
        return stock_list_withind

    @staticmethod
    def add_future_rt(stock_list):
        """
        添加未来收益列
        :param stock_list: 包含当日收盘价、昨日收盘价、和交易日的数据表 (pd.DataFrame)
        :return: 计算未来收益后的数据表 (pd.DataFrame)
        """
        stock_list = stock_list.sort_values(by='TRADE_DT', ascending=True).reset_index(drop=True)
        stock_list['STOCK_RETURN_NXT'] = stock_list.groupby('S_INFO_WINDCODE')['STOCK_RETURN'].shift(-1)
        stock_list.dropna(subset=['STOCK_RETURN_NXT'], inplace=True)
        return stock_list


if __name__ == "__main__":
    start = '20071112'
    end = '20240822'
    data_processer = DataProcess(start, end)
    data = data_processer.get_prices(fields)

    data['TRADE_DT'] = pd.to_datetime(data['TRADE_DT'].astype('str'))
    data_csi = data_processer.filter_index_cons(data, '000852.SH')
    print(data_csi.head())


