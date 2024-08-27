# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:xxx.py
# @Software: PyCharm

from scipy.stats import zscore
from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class lazyproperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.func(instance)
        setattr(instance, self.name, value)
        return value

class DecileAnalysis:

    def __init__(self, decile_num, factor, df):
        self.decile_num = decile_num
        self.factor = factor
        self.cleaned_df = self._clean_data(df)

    def _clean_data(self, df):
        """
        处理输入的数据框：
        - 去除市值（S_VAL_MV）空值的行（如果用户同意）
        - 将市值列转换为float类型（如果不是float）
        - 对因子值（factor）进行截面标准化和缩尾处理

        :param df: 数据框，包含以下列及格式：
                               TRADE_DT                datetime64[ns]
                               S_INFO_WINDCODE                 object
                               S_VAL_MV                       float64
                               STOCK_RETURN                   float64
                               RF_RETURN                      float64
                               RSTR                           float64
                               WIND_PRI_IND                    object
        :return: 处理好的数据框
        """
        count_mv_nan = df['S_VAL_MV'].isna().sum()
        print(f"Number of NaN values in 'S_VAL_MV': {count_mv_nan}")

        # 根据 count_mv_nan 的值决定是否删除这些行
        if count_mv_nan > 0:
            # 选择是否要删除空值行，或者采取其他处理方式
            print("Rows with NaN values in 'S_VAL_MV:")
            print(df.loc[df['S_VAL_MV'].isna(),:])
            decision = input("Do you want to drop rows with NaN values in 'S_VAL_MV'? (yes/no): ").strip().lower()
            if decision == 'yes':
                df = df.dropna(subset=['S_VAL_MV'])
                print(f"Dropped {count_mv_nan} rows with NaN values in 'S_VAL_MV'.")

        # 检查 S_VAL_MV 的类型
        if df['S_VAL_MV'].dtype != 'float64':
            print(f"Converting 'S_VAL_MV' from {df['S_VAL_MV'].dtype} to float64.")
            df['S_VAL_MV'] = df['S_VAL_MV'].astype('float64')

        # 获取截面缩尾因子值
        df[f'{self.factor}_winsorized'] = df.groupby('TRADE_DT')[f'{self.factor}'].transform(
            lambda x: winsorize(x, limits=[0.05, 0.05]))

        # 获取截面标准化因子值
        df[f'{self.factor}_norm'] = df.groupby('TRADE_DT')[f'{self.factor}_winsorized'].transform(lambda x: zscore(x))

        return df

    @lazyproperty
    def mv_neutralization(self):
        """
        市值中性化处理
        :return: 市值中性化处理后的数据表 (pd.DataFrame)
        """
        df = self.cleaned_df.copy()
        df = df.reset_index(drop=True)
        df['const'] = 1.0
        df['lnMV'] = np.log(df['S_VAL_MV'])

        # 处理NaN值
        nan_count = df['lnMV'].isna().sum()
        if nan_count > 0:
            print(f"Total rows with NaN values in 'lnMV': {nan_count}")
            decision = input("Do you want to drop rows with NaN values in 'lnMV'? (yes/no): ").strip().lower()
            if decision == 'yes':
                df.dropna(subset=['lnMV'], inplace=True)
                print(f"Dropped {nan_count} rows with NaN values in 'lnMV'.")

        df[['const', 'lnMV']] = df[['const', 'lnMV']].apply(pd.to_numeric, errors='coerce')
        df[f'{self.factor}_norm'] = pd.to_numeric(df[f'{self.factor}_norm'], errors='coerce')

        grouped = df.groupby('TRADE_DT')

        for date, group in grouped:
            X = group[['const', 'lnMV']]
            y = group[f'{self.factor}_norm']

            X = X.loc[y.notnull()]
            y = y.loc[y.notnull()]

            if not X.empty and not y.empty:
                model = sm.OLS(y, X)
                results = model.fit()

                # 将残差保存到原始数据框中
                df.loc[group.index, f'{self.factor}_RDY'] = results.resid

        # 删除不再需要的列
        df.drop(['const', 'lnMV', f'{self.factor}_norm', f'{self.factor}_winsorized'], axis=1, inplace=True)

        return df

    @lazyproperty
    def industry_neutral_decile(self):
        """
        对数据进行行业中性化及分组

        :return: 带分组的数据框，新增列：'DECILE'（分组标签）
        """
        df = self.mv_neutralization()
        df.reset_index(inplace=True, drop=True)

        df['DECILE'] = np.nan
        grouped = df.groupby(['TRADE_DT', 'WIND_PRI_IND'])

        for (date, industry), group in grouped:
            if len(group) < 1:
                print(f"Not enough Data points for date: {date}, industry: {industry}. Group size: {len(group)}")
                continue
            try:
                q = min(self.decile_num, len(group))
                labels = np.arange(1, q + 1)
                df.loc[group.index, 'DECILE'] = pd.qcut(group[f'{self.factor}_RDY'], q, labels=labels,
                                                        duplicates='drop').values.tolist()
            except ValueError as e:
                print(f"ValueError: {e} for date: {date}, industry: {industry}")
                continue

        df['DECILE'] = df['DECILE'].astype(int)
        return df


    def calculate_average_daily_returns(self, mv_ind_neutral = True):
        """
        计算每个交易日期和分位数的平均日回报率
        :param df_with_decile 数据框，包含以下列：'TRADE_DT'（交易日期）,
                                                'STOCK_RETURN'（交易日的回报率）,
                                                'DECILE'（分组标签）
        :return: 每日分组收益数据框及分层效果图
        """
        if mv_ind_neutral:
            df_with_decile = self.industry_neutral_decile()
        else:
            df_with_decile = self.cleaned_df.copy()
            df_with_decile['DECILE'] = np.nan
            grouped = df_with_decile.groupby('TRADE_DT')
            for date, group in grouped:
                try:
                    q = min(self.decile_num, len(group))
                    labels = np.arange(1, q + 1)
                    df_with_decile.loc[group.index, 'DECILE'] = pd.qcut(group[f'{self.factor}_norm'], q, labels=labels,
                                                            duplicates='drop').values.tolist()
                    df_with_decile['DECILE'] = df_with_decile['DECILE'].astype(int)
                except ValueError as e:
                    print(f"ValueError: {e} for date: {date}")
                    continue

        df_with_decile = df_with_decile.sort_values(by='TRADE_DT', ascending=True).reset_index(drop=True)
        df_with_decile['STOCK_RETURN_NXTD'] = df_with_decile.groupby('S_INFO_WINDCODE')['STOCK_RETURN'].shift(-1)
        df_with_decile = df_with_decile.dropna(subset=['STOCK_RETURN_NXTD'])
        mean_returns = df_with_decile.groupby(['TRADE_DT', 'DECILE'])['STOCK_RETURN_NXTD'].mean().reset_index()

        trade_dates = df_with_decile['TRADE_DT'].unique()
        factor_decile_rt_df = pd.DataFrame({'TRADE_DT': np.repeat(trade_dates, self.decile_num),
                                            'DECILE': np.tile(np.arange(1, self.decile_num + 1), len(trade_dates))})
        factor_decile_rt_df = factor_decile_rt_df.merge(mean_returns, on=['TRADE_DT', 'DECILE'], how='left')
        factor_decile_rt_df = factor_decile_rt_df.sort_values(['TRADE_DT', 'DECILE'], ascending=[True, True])
        factor_decile_rt_df['NAV'] = factor_decile_rt_df.groupby('DECILE')['STOCK_RETURN_NXTD'].transform(
            lambda x: (1 + x).cumprod())

        deciles = range(1, self.decile_num + 1)
        plt.ioff()
        plt.figure(figsize=(12, 8))
        plt.title(f'Factor {self.factor}: Cumulative Net Value by Decile')
        plt.xlabel('Date')
        plt.ylabel('NAV')
        for decile in deciles:
            decile_data = factor_decile_rt_df[factor_decile_rt_df['DECILE'] == decile]
            plt.plot(decile_data['TRADE_DT'], decile_data['NAV'], label=f'Decile {decile}')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        return factor_decile_rt_df

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
        long_short_df = factor_decile_rt_df.pivot(index='TRADE_DT', columns='DECILE',
                                                  values=['STOCK_RETURN', 'RSTR_LAG1'])
        long_short_df['long_short_rstr'] = long_short_df['RSTR_LAG1', 5] - long_short_df['RSTR_LAG1', 1]
        long_short_df['long_short_diff'] = long_short_df['STOCK_RETURN', 1] - long_short_df['STOCK_RETURN', 5]
        long_short_df['long_short_rt_adj'] = long_short_df['long_short_diff'] * (
                1 / long_short_df['long_short_rstr'])
        long_short_df['NAV_adj'] = (1 + long_short_df['long_short_rt_adj']).cumprod()
        long_short_df.reset_index(inplace=True)

        return long_short_df


if __name__ == '__main__':
    all_mkt_mom = pd.read_parquet('/Users/xinyuezhang/Desktop/中量投/Project/ZLT-Project Barra/Data/all_market_momentum.parquet')

    analysis = DecileAnalysis(decile_num=5, factor='RSTR', df=all_mkt_mom)
    mom_decile_rt_df = analysis.calculate_average_daily_returns()