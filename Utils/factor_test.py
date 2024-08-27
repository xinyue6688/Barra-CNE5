# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:factor_test.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from Utils.get_wind_data import WindData


class FactorDecileAnalysis:
    """因子分层分析类"""

    def __init__(self, group_num):
        """
        初始化因子分析类
        :param group_num: 分组数量，例：5
        """
        self.group_num = group_num
        self.ew_date_decile = None
        self.long_short_df = None

    def industry_neutralize_and_group(self, df, factor_column):
        """
        对数据进行行业中性化及分组
        :param df: 包含因子值和行业的数据框，格式 TRADE_DT(类型不限), S_INFO_WINDCODE, factor_column(例 'MOMENTUM'), WIND_PRI_IND
        :return: 带分组的数据框，新增列：'DECILE'（分组标签）
        """
        
        df.reset_index(inplace=True, drop=True)
        df['DECILE'] = np.nan
        grouped = df.groupby(['TRADE_DT', 'WIND_PRI_IND'])

        for (date, industry), group in grouped:
            if len(group) < 1:
                print(f"Not enough Data points for date: {date}, industry: {industry}. Group size: {len(group)}")
                continue
            try:
                q = min(self.group_num, len(group))
                labels = np.arange(1, q + 1)
                df.loc[group.index, 'DECILE'] = pd.qcut(group[factor_column], q, labels=labels,
                                                                duplicates='drop').values.tolist()
            except ValueError as e:
                print(f"ValueError: {e} for date: {date}, industry: {industry}")
                continue

        df['DECILE'] = df['DECILE'].astype(int)
        return df

    def calculate_average_daily_returns(self):
        """
        计算每个交易日期和分位数的平均日回报率

        :param self: self.df_with_decile 数据框，包含以下列：'TRADE_DT'（交易日期）,
                                                          'RETURN_NXT'（下一个交易日的回报率）,
                                                          'DECILE'（分组标签）
        :return: 每日分组收益数据框及分层效果图
        """
        trade_dates = self.df_with_decile['TRADE_DT'].unique()
        ew_date_decile = pd.DataFrame({'TRADE_DT': np.repeat(trade_dates, self.group_num),
                                       'DECILE': np.tile(np.arange(1, self.group_num + 1), len(trade_dates))})
        mean_returns = self.df_with_decile.groupby(['TRADE_DT', 'DECILE'])['RETURN_NXT'].mean().reset_index()
        ew_date_decile = ew_date_decile.merge(mean_returns, on=['TRADE_DT', 'DECILE'], how='left')

        ew_date_decile['NAV'] = ew_date_decile.groupby('DECILE')['RETURN_NXT'].transform(lambda x: (1 + x).cumprod())

        deciles = range(1, self.group_num + 1)

        plt.figure(figsize=(12, 8))
        plt.title('Cumulative Net Value by Decile')
        plt.xlabel('Date')
        plt.ylabel('Daily Returns')

        for decile in deciles:
            decile_data = ew_date_decile[ew_date_decile['DECILE'] == decile]
            plt.plot(decile_data['TRADE_DT'], decile_data['NAV'], label=f'Decile {decile}')

        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        self.ew_date_decile = ew_date_decile
        return ew_date_decile

    def long_short_NAV(self):
        """
        多空净值和基准净值对比

        :param
        :return: long_short_df 多空净值数据框，新增列：'long_short_turnover'（多空换手率）,
                                                   'long_short_diff'（多空回报差异）,
                                                   'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）,
                                                   'NAV_adj'（调整后净值）
        """
        decile_turnover = self.df.groupby(['TRADE_DT', 'DECILE'])['turnover'].mean().reset_index()
        self.ew_date_decile = pd.merge(self.ew_date_decile, decile_turnover, how='left', on=['TRADE_DT', 'DECILE'])
        long_short_df = self.ew_date_decile.pivot(index='TRADE_DT', columns='DECILE', values=['RETURN_NXT', 'turnover'])
        long_short_df['long_short_turnover'] = long_short_df['turnover', 5] - long_short_df['turnover', 1]
        long_short_df['long_short_diff'] = long_short_df['RETURN_NXT', 1] - long_short_df['RETURN_NXT', 5]
        long_short_df['long_short_rt_adj'] = long_short_df['long_short_diff'] * (
                    1 / long_short_df['long_short_turnover'])
        long_short_df['NAV_adj'] = (1 + long_short_df['long_short_rt_adj']).cumprod()
        long_short_df.reset_index(inplace=True)

        self.long_short_df = long_short_df
        return long_short_df

    def calculate_ic_metrics(self):
        """
        计算IC、RankIC、ICIR、RankICIR、t-test

        :param self:  self.ew_date_decile 包含以下列：'TRADE_DT'（交易日期）,
                                                   'DECILE'（分组标签）,
                                                   'RETURN_NXT'（下一个交易日的回报率）,
                                                   'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）
        :return: 各项指标结果
        """
        ic_values = []
        rank_ic_values = []

        for date, group in self.ew_date_decile.groupby('TRADE_DT'):
            group = group.dropna(subset=['DECILE', 'RETURN_NXT'])
            decile = pd.to_numeric(group['DECILE'], errors='coerce')
            future_return = pd.to_numeric(group['RETURN_NXT'], errors='coerce')

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

        lambda_hat = self.long_short_df['long_short_rt_adj'].mean()
        se_lambda = np.std(self.long_short_df['long_short_rt_adj']) / np.sqrt(len(self.long_short_df))
        t_stat = lambda_hat / se_lambda if se_lambda != 0 else np.nan

        results = pd.DataFrame({
            'IC': [ic_series.mean()],
            'RankIC': [rank_ic_series.mean()],
            'ICIR': [icir],
            'RankICIR': [rank_icir],
            't-stat': [t_stat]
        })

        return results


class FamaFrenchFactor:
    """Fama-French SMB, HML因子检验类"""

    def __init__(self):
        """
        初始化Fama-French 因子检验类
        """
        self.data = None
        self.portfolio_return = None
        self.ff_pivot = None
        self.SMB = None
        self.HML = None

    def split_BP(self, x):
        """
        根据市净率分组

        :param x: 分组数据
        :return: 带有BM分组标签的数据
        """
        x.loc[x['BP'] >= x.BP.quantile(0.7), 'group_BM'] = 'H'
        x.loc[x['BP'] < x.BP.quantile(0.3), 'group_BM'] = 'L'
        return x

    def split_SIZE(self, x):
        """
        根据市值分组

        :param x: 分组数据
        :return: 带有SIZE分组标签的数据
        """
        x.loc[x['S_VAL_MV'] >= x.S_VAL_MV.median(), 'group_SIZE'] = 'B'
        return x

    def assign_ffgroup(self, data):
        """
        对数据进行BM和SIZE分组，并生成FF组合标签

        :param data: 带有市值和PB数据的合并数据框，需包含以下列：'TRADE_DT', 'S_VAL_MV', 'BP'
        :return: 带有分组标签的数据框，新增列：'group_BM', 'group_SIZE', 'group_ff'
        """
        self.data = data
        self.data['group_BM'] = 'M'
        self.data = self.data.groupby(['TRADE_DT']).apply(self.split_BP).reset_index(drop=True)
        self.data['group_SIZE'] = 'S'
        self.data = self.data.groupby(['TRADE_DT']).apply(self.split_SIZE).reset_index(drop=True)
        self.data['group_ff'] = self.data.group_SIZE + '/' + self.data.group_BM
        return self.data

    def calculate_portfolio_return(self):
        """
        计算每个FF组合的每日收益率

        :return: 每日组合收益数据框，需包含以下列：'TRADE_DT', 'group_ff', 'RETURN_NXT'
        """
        portfolio_return = self.data.groupby(['TRADE_DT', 'group_ff']).apply(
            lambda x: (x.RETURN_NXT * x.S_VAL_MV).sum() / x.S_VAL_MV.sum())
        portfolio_return = portfolio_return.reset_index()
        portfolio_return = portfolio_return.rename(columns={portfolio_return.columns[-1]: 'RETURN'})
        portfolio_return['TRADE_DT'] = pd.to_datetime(portfolio_return['TRADE_DT'].astype(str))
        self.portfolio_return = portfolio_return
        return portfolio_return

    def calculate_factors(self):
        """
        计算SMB和HML因子

        :return: SMB和HML因子数据框
        """
        ff_pivot = self.portfolio_return.pivot(index='TRADE_DT', columns='group_ff', values='RETURN')
        SMB = (ff_pivot['S/H'] + ff_pivot['S/M'] + ff_pivot['S/L']) / 3 - (
                    ff_pivot['B/H'] + ff_pivot['B/M'] + ff_pivot['B/L']) / 3
        HML = (ff_pivot['S/H'] + ff_pivot['B/H']) / 2 - (ff_pivot['S/L'] + ff_pivot['B/L']) / 2
        self.SMB = pd.DataFrame(SMB, columns=['RETURN']).reset_index()
        self.HML = pd.DataFrame(HML, columns=['RETURN']).reset_index()
        return self.SMB, self.HML



if __name__ == "__main__":
    '''因子分层分析类使用样例'''
    cleaned_data = pd.read_csv('../Data/sample_w_decile.csv')
    factor_analysis = FactorDecileAnalysis(cleaned_data, 5)

    # 行业中性化及分组
    cleaned_df = factor_analysis.industry_neutralize_and_group()
    print(cleaned_df)

    # 计算每个交易日期和分位数的平均日回报率
    ew_date_decile = factor_analysis.calculate_average_daily_returns()
    print(ew_date_decile)

    # 多空净值和基准净值对比
    factor_analysis.long_short_NAV()
    print(factor_analysis)

    # 计算IC、RankIC、ICIR、RankICIR、t-test
    results = factor_analysis.calculate_ic_metrics()

    # 输出结果
    print(results)

    '''Fama因子类使用样例'''
    data = pd.read_csv('../Data/sample_w_decile.csv')
    data['TRADE_DT'] = pd.to_datetime(data['TRADE_DT'])

    wind = WindData('20100104', '20100105')
    pb_value = wind.get_indicator(['S_INFO_WINDCODE','TRADE_DT','S_VAL_PB_NEW'])
    pb_value['TRADE_DT'] = pd.to_datetime(pb_value['TRADE_DT'].astype(str))
    data = data.merge(pb_value, on = ['S_INFO_WINDCODE', 'TRADE_DT'], how = 'left')
    data['S_VAL_PB_NEW'] = data['S_VAL_PB_NEW'].astype(float)
    data['BP'] = 1 / data['S_VAL_PB_NEW']
    # 初始化Fama-French 3因子类
    ff3 = FamaFrenchFactor()

    # 执行分组
    grouped_data = ff3.assign_ffgroup(data)
    print(grouped_data.head())
    # 计算组合收益
    portfolio_return = ff3.calculate_portfolio_return()
    print(portfolio_return)

    # 计算因子
    SMB, HML = ff3.calculate_factors()

    # 打印结果
    print("SMB因子:")
    print(SMB.head())
    print("\nHML因子:")
    print(HML.head())