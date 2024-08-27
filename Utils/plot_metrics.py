# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:plot_metrics.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Utils.get_wind_data import WindData

class FactorPerformanceYoY:
    """因子年度表现分析类"""

    def __init__(self, data, years_to_include, isindustry=False, windcode=None):
        """
        初始化因子年度表现分析类

        :param data: 包含日期和收益率的数据框，需包含以下列：'TRADE_DT', 'RETURN'
        :param years_to_include: 要包含的年份列表
        :param isindustry: 是否为行业分析，默认为False
        :param windcode: 行业代码，若isindustry为True，则需提供
        """
        self.data = data
        self.years_to_include = years_to_include
        self.isindustry = isindustry
        self.windcode = windcode
        if self.isindustry and self.windcode:
            self.data = self.prepare_data(self.windcode)

    def prepare_data(self, windcode):
        """
        准备行业分析数据

        :param windcode: 行业代码, 例：'882010.WI'
        :return: 包含行业多空收益的数据框
        """
        mask = self.data['S_INFO_WINDCODE'] == windcode
        return_data = self.data.loc[mask, ['TRADE_DT', 'RETURN']]
        return_data.reset_index(drop=True, inplace=True)
        average_return_other = self.data[~mask].groupby('TRADE_DT')['RETURN'].mean()
        merged_data = pd.merge(return_data, average_return_other, on='TRADE_DT', how='left')
        merged_data['NET RETURN'] = merged_data['RETURN_x'] - merged_data['RETURN_y']
        merged_data = merged_data.rename(columns={'RETURN_x': 'single_ind',
                                                  'RETURN_y': 'others',
                                                  'NET RETURN': 'RETURN'})
        return merged_data

    def plot_nav_comparison(self, label):
        """
        绘制指定年份的NAV曲线对比

        :param label: 图表标题, 例：'Market'
        """
        # 过滤数据
        filtered_data = self.data[self.data['TRADE_DT'].dt.year.isin(self.years_to_include)].copy()

        # 计算每年的NAV曲线
        filtered_data['NAV'] = filtered_data.groupby(filtered_data['TRADE_DT'].dt.year)['RETURN'].transform(
            lambda x: (1 + x).cumprod())

        # 计算一年中的天数
        filtered_data['DayOfYear'] = filtered_data['TRADE_DT'].dt.dayofyear

        # 绘制每年的NAV曲线
        plt.figure(figsize=(12, 8))
        for year in self.years_to_include:
            yearly_data = filtered_data[filtered_data['TRADE_DT'].dt.year == year]
            plt.plot(yearly_data['DayOfYear'], yearly_data['NAV'], label=str(year))

        # 计算每年平均NAV
        average_nav = filtered_data.groupby('DayOfYear')['RETURN'].mean().reset_index()
        average_nav['NAV'] = (1 + average_nav['RETURN']).cumprod()
        plt.plot(average_nav['DayOfYear'], average_nav['NAV'], label='Average', color='black', linewidth=2)

        plt.xlabel('Day of the Year')
        plt.ylabel('NAV')
        plt.title(f'{label}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def performance_metrics(self):
        """
        计算指定年份的年度表现指标

        :return: 包含年度回报率、波动率和夏普比率的数据框
        """
        # 过滤数据
        filtered_data = self.data[self.data['TRADE_DT'].dt.year.isin(self.years_to_include)].copy()
        filtered_data['NAV'] = filtered_data.groupby(filtered_data['TRADE_DT'].dt.year)['RETURN'].transform(
            lambda x: (1 + x).cumprod())
        filtered_data['DayOfYear'] = filtered_data['TRADE_DT'].dt.dayofyear

        # 计算每年平均NAV
        average_nav = filtered_data.groupby('DayOfYear')['RETURN'].mean().reset_index()
        average_nav['NAV'] = (1 + average_nav['RETURN']).cumprod()

        avg_rt = average_nav['NAV'].iloc[-1] - 1
        avg_vol = average_nav['RETURN'].std() * np.sqrt(len(average_nav))
        avg_sharpe = avg_rt / avg_vol

        yearly_return = []
        yearly_volatility = []
        yearly_sharpe = []
        for year in self.years_to_include:
            yearly_data = filtered_data[filtered_data['TRADE_DT'].dt.year == year]
            rt = yearly_data['NAV'].iloc[-1] - 1
            vol = yearly_data['RETURN'].std() * np.sqrt(len(yearly_data))
            sharpe = rt / vol
            yearly_return.append(rt)
            yearly_volatility.append(vol)
            yearly_sharpe.append(sharpe)

        years_to_include_with_avg = self.years_to_include + ['Average']
        yearly_return.append(avg_rt)
        yearly_volatility.append(avg_vol)
        yearly_sharpe.append(avg_sharpe)

        df = pd.DataFrame({
            'Year': years_to_include_with_avg,
            'Annual Return': yearly_return,
            'Annual Volatility': yearly_volatility,
            'Annual Sharpe': yearly_sharpe
        })

        return df


if __name__ == '__main__':
    wind = WindData('20100101', '20130101')
    field_prices = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_PRECLOSE', 'S_DQ_CLOSE']
    data = wind.get_prices(field_prices)

    data = data[data['S_INFO_WINDCODE'] == '000001.SZ']
    data['TRADE_DT'] = pd.to_datetime(data['TRADE_DT'].astype(str))
    data[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']] = data[['S_DQ_PRECLOSE', 'S_DQ_CLOSE']].astype(float)
    data['RETURN'] = data['S_DQ_CLOSE'] / data['S_DQ_PRECLOSE'] - 1

    years_to_include = [2010, 2012]
    fpy = FactorPerformanceYoY(data, years_to_include)

    # 绘制NAV对比图
    fpy.plot_nav_comparison('000001.SZ')

    # 计算年度表现指标
    performance_metrics_df = fpy.performance_metrics()
    print("年度表现指标:")
    print(performance_metrics_df)