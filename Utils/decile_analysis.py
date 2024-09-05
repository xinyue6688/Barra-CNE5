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

from Utils.return_metrics import MetricsCalculator
from barra_cne5_factor import GetData

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

    def __init__(self, df, decile_num, factor, rebal_freq, mv_neutral= True, trade_pool = False):
        self.factor = factor
        self.clean_df = self._clean_data(df)
        self.decile_num = decile_num
        self.rebal_freq = rebal_freq
        self.mv_neutral = mv_neutral
        self.df_with_decile = self.industry_neutral_decile(trade_pool)
        self.factor_decile_rt_df = self.calculate_decile_returns()
        self.long_short_df = self.long_short_NAV()

    def _clean_data(self, df):
        """
        处理输入的数据框：
        - 在数据框中添加收益列（如没有）
        - 去除市值（S_VAL_MV）空值的行
        - 将市值列转换为float类型（如果不是float）

        :param df: 数据框，包含以下列及格式：
                               TRADE_DT                datetime64[ns]
                               S_INFO_WINDCODE                 object
                               S_VAL_MV                       float64
                               STOCK_RETURN                   float64
                               RF_RETURN                      float64
                               {factor}                       float64
                               WIND_PRI_IND                    object
        :return: 处理好的数据框
        """

        # 确保数据框中有正确的收益列
        if 'STOCK_RETURN' not in df.columns:
            df['STOCK_RETURN'] = df['S_DQ_CLOSE'] / df['S_DQ_PRECLOSE'] - 1

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

        return df

    def mv_neutralization(self):
        """
        市值中性化处理
        :return: 市值中性化处理后的数据表 (pd.DataFrame)
        """
        df = self.clean_df
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
            df.loc[group.index, 'DECILE'] = pd.qcut(group[f'{self.factor}_LAG1'], q, labels=labels,
                                                    duplicates='drop').values.tolist()

        except ValueError as e:
            print(
                f"ValueError: {e} for date/period: {date_or_period}, industry: {industry}. Adjusting labels and retrying...")

            # 当捕获到ValueError时，获取实际的区间，并+1变为分组数直接赋值回原数据框
            bins = pd.qcut(group[f'{self.factor}_LAG1'], q, labels=False, duplicates='drop').values.tolist()
            deciles = [x + 1 for x in bins]
            df.loc[group.index, 'DECILE'] = deciles

    def industry_neutral_decile(self, trade_pool):
        """
        对数据进行行业中性化及分组

        :return: 带分组的数据框，新增列：'DECILE'（分组标签）
        """

        # 选择是否市值中性
        if self.mv_neutral:
            df = self.mv_neutralization()
        else:
            df = self.clean_df

        df = df.sort_values(by = ['S_INFO_WINDCODE', 'TRADE_DT'])
        df[f'{self.factor}_LAG1'] = df.groupby('S_INFO_WINDCODE')[f'{self.factor}'].shift(1)
        df = df.dropna(subset = [f'{self.factor}_LAG1'])
        df.reset_index(inplace=True, drop=True)
        df['DECILE'] = np.nan

        if trade_pool:
            trade_pool = pd.read_parquet('/Volumes/quanyi4g/data/index/pool/tradable_components.parquet')
            df = pd.merge(trade_pool, df, how='left', on = ['TRADE_DT', 'S_INFO_WINDCODE'])

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
            # 按日期升序排列
            df['W'] = df['TRADE_DT'].dt.to_period('W')

            # 标记每周的换仓日
            df['REBALANCE_DT'] = df.groupby('W')['TRADE_DT'].transform('max') == df['TRADE_DT']
            grouped = df.groupby(['W', 'WIND_PRI_IND'])

            for (week, industry), group in grouped:

                # 每一周的换仓日这天，在各个行业根据因子值分层，把分层定位回原df数据框
                rebalance_df = group.loc[group['REBALANCE_DT'], :]
                self._assign_decile(rebalance_df, week, industry, df)

            # 原数据框根据标的分组
            grouped_by_stock = df.groupby('S_INFO_WINDCODE')

            for _, group in grouped_by_stock:

                # 分层信息前向填充
                df.loc[group.index, 'DECILE'] = group['DECILE'].ffill()

            # 删除带有分层缺失值的行，因缺少换仓日前一天收盘时的因子值导致的分层值缺失
            df = df.dropna(subset=['DECILE'])

        elif self.rebal_freq == 'm':
            df['M'] = df['TRADE_DT'].dt.to_period('M')
            df = df.sort_values(['M', 'TRADE_DT'], ascending=True).reset_index(drop=True)

            # 标记每月的换仓日
            df['REBALANCE_DT'] = df.groupby('M')['TRADE_DT'].transform('max') == df['TRADE_DT']
            grouped = df.groupby(['M', 'WIND_PRI_IND'])

            for (month, industry), group in grouped:
                # 每月的换仓日这天，在各个行业根据因子值分层，把分层定位回原df数据框
                rebalance_df = group.loc[group['REBALANCE_DT'], :]
                self._assign_decile(rebalance_df, month, industry, df)

            # 原数据框根据标的分组
            grouped_by_stock = df.groupby('S_INFO_WINDCODE')

            for _, group in grouped_by_stock:
                # 分层信息前向填充
                df.loc[group.index, 'DECILE'] = group['DECILE'].ffill()

            # 删除带有分层缺失值的行，因缺少因子值导致的分层值缺失
            df = df.dropna(subset=['DECILE'])

        df['DECILE'] = df['DECILE'].astype(int)
        return df


    def calculate_decile_returns(self):
        """
        计算每个交易日期和分位数的平均日回报率
        :param df_with_decile 数据框，包含以下列：'TRADE_DT'（交易日期）,
                                                'STOCK_RETURN'（交易日的回报率）,
                                                'DECILE'（分组标签）
        :return: 每日分组收益数据框及分层效果图
        """
        df_with_decile =  self.df_with_decile

        mean_returns = df_with_decile.groupby(['TRADE_DT', 'DECILE'])['STOCK_RETURN'].mean().reset_index()
        trade_dates = df_with_decile['TRADE_DT'].unique()

        factor_decile_rt_df = pd.DataFrame({'TRADE_DT': np.repeat(trade_dates, self.decile_num),
                                            'DECILE': np.tile(np.arange(1, self.decile_num + 1), len(trade_dates))})
        factor_decile_rt_df = factor_decile_rt_df.merge(mean_returns, on=['TRADE_DT', 'DECILE'], how='left')
        factor_decile_rt_df = factor_decile_rt_df.sort_values(['TRADE_DT', 'DECILE'], ascending=[True, True])
        factor_decile_rt_df['NAV'] = factor_decile_rt_df.groupby('DECILE')['STOCK_RETURN'].transform(
            lambda x: (1 + x).cumprod())

        factor_decile_rt_df = pd.merge(factor_decile_rt_df, df_with_decile.groupby(['TRADE_DT', 'DECILE'])[f'{self.factor}_LAG1'].mean().reset_index(),
                                       how='left', on=['TRADE_DT', 'DECILE'])

        return factor_decile_rt_df

    def plot_decile_returns(self):
        deciles = range(1, self.decile_num + 1)
        plt.ioff()
        plt.figure(figsize=(12, 8))
        plt.title(f'Factor {self.factor}: Cumulative Net Value by Decile')
        plt.xlabel('Date')
        plt.ylabel('NAV')
        for decile in deciles:
            decile_data = self.factor_decile_rt_df[self.factor_decile_rt_df['DECILE'] == decile]
            plt.plot(decile_data['TRADE_DT'], decile_data['NAV'], label=f'Decile {decile}')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def long_short_NAV(self):
        """
        多空净值和基准净值对比

        :param
        :return: long_short_df 多空净值数据框，新增列：'long_short_turnover'（多空换手率）,
                                                   'long_short_diff'（多空回报差异）,
                                                   'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）,
                                                   'NAV_adj'（调整后净值）
        """
        factor_decile_rt_df = self.factor_decile_rt_df

        long_short_df = factor_decile_rt_df.pivot(index='TRADE_DT', columns='DECILE',
                                                  values=['STOCK_RETURN', f'{self.factor}_LAG1'])
        long_short_df[f'long_short_{self.factor}'] = long_short_df[f'{self.factor}_LAG1', 5] - long_short_df[f'{self.factor}_LAG1', 1]
        if (1 + long_short_df['STOCK_RETURN', 1]).cumprod().iloc[-1] > (1 + long_short_df['STOCK_RETURN', 5]).cumprod().iloc[-1]:
            long_short_df['long_short_diff'] = long_short_df['STOCK_RETURN', 1] - long_short_df['STOCK_RETURN', 5]
        else:
            long_short_df['long_short_diff'] = long_short_df['STOCK_RETURN', 5] - long_short_df['STOCK_RETURN', 1]
        long_short_df['long_short_rt_adj'] = long_short_df['long_short_diff'] * (1 / long_short_df[f'long_short_{self.factor}'])
        long_short_df['NAV_adj'] = (1 + long_short_df['long_short_rt_adj']).cumprod()
        long_short_df.reset_index(inplace=True)

        if isinstance(long_short_df.columns, pd.MultiIndex):
            long_short_df.columns = ['_'.join(map(str, col)).strip() if type(col) is tuple else col for col in
                                     long_short_df.columns]
        long_short_df.rename(columns={'TRADE_DT_': 'TRADE_DT',
                                      'NAV_adj_': 'NAV_adj',
                                      'long_short_rt_adj_': 'long_short_rt_adj'}, inplace=True)

        return long_short_df

    def plot_long_short_NAV(self, double_axis = False):
        benchmark = GetData.wind_index('8841388.WI')
        benchmark['TRADE_DT'] = pd.to_datetime(benchmark['TRADE_DT'])
        benchmark['S_DQ_PCTCHANGE'] = benchmark['S_DQ_PCTCHANGE'].astype(float)
        benchmark['S_DQ_PCTCHANGE'] = benchmark['S_DQ_PCTCHANGE'] * 0.01
        benchmark['NAV'] = (1 + benchmark['S_DQ_PCTCHANGE']).cumprod()

        long_short_df = self.long_short_df

        aligned_df = pd.merge(long_short_df[['TRADE_DT', 'NAV_adj']], benchmark[['TRADE_DT', 'NAV']], on='TRADE_DT',
                              how='inner')

        if double_axis:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.plot(aligned_df['TRADE_DT'], aligned_df['NAV_adj'], color='blue',
                     label='Long-Short Portfolio (Factor Exposure 1)')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Long-Short Portfolio', color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            ax2 = ax1.twinx()
            ax2.plot(aligned_df['TRADE_DT'], aligned_df['NAV'], color='red', label='EW A Index')
            ax2.set_ylabel('EW A Index', color='black')
            ax2.tick_params(axis='y', labelcolor='black')

            ax1.set_title(f'{self.factor}: Long-short Portfolio NAV ')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            plt.show()

        else:
            plt.figure(figsize=(12, 8))
            plt.title('NAV')
            plt.xlabel('Date')
            plt.ylabel('Cumulative NAV')
            plt.plot(aligned_df['TRADE_DT'], aligned_df['NAV_adj'], label='Long-Short Portfolio (Factor Exposure 1)')
            plt.plot(aligned_df['TRADE_DT'], aligned_df['NAV'], label='EW A Index')
            plt.legend()
            plt.show()


    def _print_ic_metrics(self, df_type = 'decile'):
        """
        计算IC、RankIC、ICIR、RankICIR、t-test

        :param self:  self.factor_decile_rt_df 包含以下列：'TRADE_DT'（交易日期）,
                                                   'DECILE'（分组标签）,
                                                   'STOCK_RETURN'（下一个交易日的回报率）,
                                                   'long_short_rt_adj'（根据因子暴露为1调整后的多空回报率）
        :return: 各项指标结果
        """

        ic_values = []
        rank_ic_values = []
        if df_type == 'decile':
            rt_factor_val_df = self.factor_decile_rt_df
        elif df_type == 'stock':
            rt_factor_val_df = self.df_with_decile
        else:
            print("Value df_type must be 'decile' or 'stock'")

        for date, group in rt_factor_val_df.groupby('TRADE_DT'):
            #group = group.dropna(subset=['DECILE', 'STOCK_RETURN'])
            factor_val = pd.to_numeric(group[f'{self.factor}_LAG1'], errors='coerce')
            future_return = pd.to_numeric(group['STOCK_RETURN'], errors='coerce')

            if len(factor_val) < 2 or factor_val.isnull().any() or future_return.isnull().any():
                ic_values.append(np.nan)
                rank_ic_values.append(np.nan)
                continue

            ic, _ = pearsonr(factor_val, future_return)
            ic_values.append(ic)

            rank_ic, _ = spearmanr(factor_val, future_return)
            rank_ic_values.append(rank_ic)

        ic_series = pd.Series(ic_values)
        rank_ic_series = pd.Series(rank_ic_values)

        ic = ic_series.mean()
        ic_t_stat = ic/(ic_series.std()/np.sqrt(len(ic_series)))
        rank_ic = rank_ic_series.mean()
        rank_ic_t_stat = rank_ic / (rank_ic_series.std() / np.sqrt(len(rank_ic_series)))
        icir = ic_series.mean() / ic_series.std()
        rank_icir = rank_ic_series.mean() / rank_ic_series.std()

        results = pd.DataFrame({
            'IC': [ic],
            'IC t-stat': [ic_t_stat],
            'RankIC': [rank_ic],
            'RankIC t-stat': [rank_ic_t_stat],
            'ICIR': [icir],
            'RankICIR': [rank_icir],
        })

        print(results)

    def _calculate_return_features(self):
        daily_return = self.long_short_df['long_short_rt_adj']
        date = self.long_short_df['TRADE_DT'].unique()
        self.df = pd.DataFrame({
            'daily_return': daily_return,
            'date': date
        })
        # NAV
        self.nav = self.df['daily_return'].transform(lambda x: (1 + x).cumprod())
        # Total return
        self.total_return = (self.nav.iloc[-1] / self.nav.iloc[0]) - 1

        # Annualized return
        self.annualized_return = ((1 + self.total_return) ** (252 / len(self.df))) - 1

        # Daily volatility
        self.daily_volatility = self.df['daily_return'].std()

        # Annualized volatility
        self.annualized_volatility = self.daily_volatility * np.sqrt(252)

        # Sharpe ratio
        self.sharpe_ratio = self.annualized_return / self.annualized_volatility

        # t-stat
        lambda_hat = self.long_short_df['long_short_rt_adj'].mean()
        se_lambda = np.std(self.long_short_df['long_short_rt_adj']) / np.sqrt(len(self.long_short_df))
        self.t_stat = lambda_hat / se_lambda if se_lambda != 0 else np.nan

        # Max drawdown
        cumulative_returns = (1 + self.df['daily_return']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        self.max_drawdown = drawdown.min()

        # Daily win rate
        self.df['daily_win_rate'] = np.where(self.df['daily_return'] > 0, 1, 0)
        self.daily_win_rate = self.df['daily_win_rate'].mean()

        # Max drawdown start and end dates
        max_dd_id = np.argmax(np.maximum.accumulate(self.df['daily_return']) - self.df['daily_return'])
        drawdown_end_id = np.argmax(self.df['daily_return'][:max_dd_id])
        self.drawdown_end_date = self.df['date'].iloc[drawdown_end_id]
        drawdown_start_id = np.argmax(self.df['daily_return'][:drawdown_end_id])
        self.drawdown_start_date = self.df['date'].iloc[drawdown_start_id]

    def _print_metrics(self):
        self._calculate_return_features()
        print(f"Total Return: {self.total_return:.2%}")
        print(f"Annualized Return: {self.annualized_return:.2%}")
        print(f"Annualized Volatility: {self.annualized_volatility:.2%}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Long short t-stat: {self.t_stat:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Daily Win Rate: {self.daily_win_rate:.2%}")
        print(f"Max Drawdown Start Date: {self.drawdown_start_date}")
        print(f"Max Drawdown End Date: {self.drawdown_end_date}")

    def print_long_short_metrics(self):
        print(f'Long-short portfolio return features of factor {self.factor}:')
        self._print_metrics()

        print(f'Decile factor value and returns IC_IR metrics of factor {self.factor}:')
        self._print_ic_metrics(df_type='decile')

    def print_icir_bystock(self):
        print(f'Stock factor value and returns IC_IR metrics of factor {self.factor}:')
        self._print_ic_metrics(df_type='stock')

if __name__ == '__main__':
    all_mkt_mom = pd.read_parquet('/Users/xinyuezhang/Desktop/中量投/Project/ZLT-Project Barra/Data/all_market_momentum.parquet')

    analysis = DecileAnalysis(all_mkt_mom, 5, 'RSTR', 'd')
    mom_decile_rt_df = analysis.calculate_decile_returns()
    long_short_df = analysis.long_short_NAV(mom_decile_rt_df)
    results = analysis.calculate_ic_metrics()
    print(results)


