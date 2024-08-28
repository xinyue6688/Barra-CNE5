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
from scipy.stats import pearsonr, spearmanr


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

    def __init__(self, df, decile_num, factor, rebal_freq):
        self.factor = factor
        self.cleaned_df = self._clean_data(df)
        self.decile_num = decile_num
        self.rebal_freq = rebal_freq
        self.df_with_decile = None
        self.decile_rt_factorval = None
        self.long_short_df = None

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
            df.dropna(subset=['lnMV'], inplace=True)

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
                df.loc[group.index, f'{self.factor}'] = results.resid

        # 删除不再需要的列
        df.drop(['const', 'lnMV', f'{self.factor}_norm', f'{self.factor}_winsorized'], axis=1, inplace=True)

        return df

    def _assign_decile(self, group, date_or_period, industry, df):
        try:
            q = min(self.decile_num, len(group))
            labels = np.arange(1, q + 1)
            df.loc[group.index, 'DECILE'] = pd.qcut(group[self.factor], q, labels=labels,
                                                    duplicates='drop').values.tolist()
        except ValueError as e:
            print(f"ValueError: {e} for date/period: {date_or_period}, industry: {industry}")

    def industry_neutral_decile(self, mv_neutral = True):
        """
        对数据进行行业中性化及分组

        :return: 带分组的数据框，新增列：'DECILE'（分组标签）
        """

        # 选择是否市值中性
        if mv_neutral:
            df = self.mv_neutralization()
        else:
            df = self.cleaned_df.copy()

        df.reset_index(inplace=True, drop=True)
        df['DECILE'] = np.nan

        # 日频调仓
        if self.rebal_freq == 'd':
            grouped = df.groupby(['TRADE_DT', 'WIND_PRI_IND'])

            for (date, industry), group in grouped:
                if len(group) < 1:
                    print(f"Not enough Data points for date: {date}, industry: {industry}. Group size: {len(group)}")
                    continue
                else:
                    self._assign_decile(group, date, industry, df)

        # 周频调仓
        elif self.rebal_freq == 'w':
            df['W'] = df['TRADE_DT'].dt.to_period('W')
            grouped = df.groupby(['W', 'WIND_PRI_IND'])

            for (week, industry), group in grouped:

                try:
                    q = min(self.decile_num, len(group))
                    labels = np.arange(1, q + 1)
                    df.loc[group.index, 'DECILE'] = pd.qcut(group[f'{self.factor}'], q, labels=labels,
                                                            duplicates='drop').values.tolist()
                except ValueError as e:
                    print(f"ValueError: {e} for date: {week}, industry: {industry}")
                    continue
        # 月频调仓
        elif self.rebal_freq == 'm':
            df['M'] = df['TRADE_DT'].dt.to_period('M')
            grouped = df.groupby(['M', 'WIND_PRI_IND'])

            for (month, industry), group in grouped:

                try:
                    q = min(self.decile_num, len(group))
                    labels = np.arange(1, q + 1)
                    df.loc[group.index, 'DECILE'] = pd.qcut(group[f'{self.factor}'], q, labels=labels,
                                                            duplicates='drop').values.tolist()
                except ValueError as e:
                    print(f"ValueError: {e} for month: {month}, industry: {industry}")
                    continue

        df['DECILE'] = df['DECILE'].astype(int)
        return df

# TODO: 增加月频调仓
    def calculate_decile_returns(self, mv_neutral = True):
        """
        计算每个交易日期和分位数的平均日回报率
        :param df_with_decile 数据框，包含以下列：'TRADE_DT'（交易日期）,
                                                'STOCK_RETURN'（交易日的回报率）,
                                                'DECILE'（分组标签）
        :return: 每日分组收益数据框及分层效果图
        """
        df_with_decile = self.industry_neutral_decile(mv_neutral)
        df_with_decile = df_with_decile.sort_values(by='TRADE_DT', ascending=True).reset_index(drop=True)
        df_with_decile['STOCK_RETURN_NXTD'] = df_with_decile.groupby('S_INFO_WINDCODE')['STOCK_RETURN'].shift(-1)
        df_with_decile = df_with_decile.dropna(subset=['STOCK_RETURN_NXTD'])
        self.df_with_decile = df_with_decile


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

    def long_short_NAV(self, factor_decile_rt_df):
        """
        多空净值和基准净值对比

        :param
        :return: long_short_df 多空净值数据框，新增列：'long_short_turnover'（多空换手率）,
                                                   'long_short_diff'（多空回报差异）,
                                                   'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）,
                                                   'NAV_adj'（调整后净值）
        """
        df = self.df_with_decile

        decile_factor = df.groupby(['TRADE_DT', 'DECILE'])[f'{self.factor}'].mean().reset_index()
        factor_decile_rt_df = pd.merge(factor_decile_rt_df, decile_factor, how='left', on=['TRADE_DT', 'DECILE'])
        self.decile_rt_factorval = factor_decile_rt_df

        long_short_df = factor_decile_rt_df.pivot(index='TRADE_DT', columns='DECILE',
                                                  values=['STOCK_RETURN_NXTD', f'{self.factor}'])
        long_short_df['long_short_rstr'] = long_short_df[f'{self.factor}', 5] - long_short_df[f'{self.factor}', 1]
        long_short_df['long_short_diff'] = long_short_df['STOCK_RETURN_NXTD', 1] - long_short_df['STOCK_RETURN_NXTD', 5]
        long_short_df['long_short_rt_adj'] = long_short_df['long_short_diff'] * (
                1 / long_short_df['long_short_rstr'])
        long_short_df['NAV_adj'] = (1 + long_short_df['long_short_rt_adj']).cumprod()
        long_short_df.reset_index(inplace=True)

        self.long_short_df = long_short_df

        return long_short_df


    def calculate_ic_metrics(self):
        """
        计算IC、RankIC、ICIR、RankICIR、t-test

        :param self:  self.factor_decile_rt_df 包含以下列：'TRADE_DT'（交易日期）,
                                                   'DECILE'（分组标签）,
                                                   'STOCK_RETURN'（下一个交易日的回报率）,
                                                   'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）
        :return: 各项指标结果
        """

        factor_decile_rt_df = self.decile_rt_factorval
        long_short_df = self.long_short_df

        ic_values = []
        rank_ic_values = []

        if self.rebal_freq == 'd':
            rebal_col = 'TRADE_DT'
        elif self.rebal_freq == 'w':
            rebal_col = 'W'
        elif self.rebal_freq == 'm':
            rebal_col = 'M'

        for date, group in factor_decile_rt_df.groupby(f'{rebal_col}'):
            group = group.dropna(subset=['DECILE', 'STOCK_RETURN_NXTD'])
            decile = pd.to_numeric(group['DECILE'], errors='coerce')
            future_return = pd.to_numeric(group['STOCK_RETURN_NXTD'], errors='coerce')

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


if __name__ == '__main__':
    all_mkt_mom = pd.read_parquet('/Users/xinyuezhang/Desktop/中量投/Project/ZLT-Project Barra/Data/all_market_momentum.parquet')

    analysis = DecileAnalysis(all_mkt_mom, 5, 'RSTR', 'w')
    mom_decile_rt_df = analysis.calculate_decile_returns()
    long_short_df = analysis.long_short_NAV(mom_decile_rt_df)
    results = analysis.calculate_ic_metrics()
    print(results)


