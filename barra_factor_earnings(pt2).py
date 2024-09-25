# -*- coding = utf-8 -*-
# @Time: 2024/09/23
# @Author: Xinyue
# @File:xxx.py
# @Software: PyCharm

import pandas as pd
import statsmodels.api as sm
import numpy as np
from linearmodels.panel import PanelOLS
from scipy import stats

factor_df = pd.read_parquet('Data/style_factors_updated.parquet')
factor_df.dropna(subset = 'S_VAL_MV', inplace = True)

# 定义不包括 'WIND_PRI_IND' 的因子列
factor_columns = ['COUNTRY', 'BETA', 'RSTR', 'LNCAP', 'EARNYILD', 'GROWTH',
                  'LEVERAGE', 'LIQUIDITY', 'RESVOL', 'BTOP', 'NLSIZE']

# 将 'WIND_PRI_IND' 转换为虚拟变量
factor_df_with_dummies = pd.get_dummies(factor_df, columns=['WIND_PRI_IND'],
                                     drop_first=True, dtype = float)  # drop_first=True avoids multicollinearity

# Define a function to run weighted OLS regression for each day
def weighted_ols_for_day(df_with_dummies):
    # 获取行业虚拟变量列
    industry_columns = [col for col in df_with_dummies if col.startswith('WIND_PRI_IND_')]

    # 将行业虚拟变量列添加到因子列中
    X = df_with_dummies[factor_columns + industry_columns]

    # 因变量（股票回报率）
    y = df_with_dummies['RETURN_T1']

    # 使用市值作为权重
    df_with_dummies['S_VAL_MV'] = pd.to_numeric(df_with_dummies['S_VAL_MV'])
    weights = df_with_dummies['S_VAL_MV']

    # 执行加权最小二乘回归，不包括常数项
    try:
        model = sm.WLS(y, X, weights=weights)
        results = model.fit()

        # 返回因子回报（系数）
        return results.params

    except np.linalg.LinAlgError as e:
        print(f"SVD did not converge for {df_with_dummies['TRADE_DT'].iloc[0]}: {e}")
        return pd.Series([None] * len(X.columns), index=X.columns)

# 按 'TRADE_DT' 分组，并对每一天应用回归
factor_returns_by_day = factor_df_with_dummies.groupby('TRADE_DT').apply(weighted_ols_for_day, include_groups=False)

# Reset the index to make 'TRADE_DT' a column
factor_returns_by_day = factor_returns_by_day.reset_index()

# Test significance
# 设置面板数据的多重索引
panel_data = factor_df_with_dummies.set_index(['S_INFO_WINDCODE', 'TRADE_DT'])

# 选择自变量 (因子列和虚拟变量)
X = panel_data[factor_columns + [col for col in panel_data if col.startswith('WIND_PRI_IND_')]]

# 因变量 (股票收益)
y = panel_data['RETURN_T1']

# 市值作为权重
weights = panel_data['S_VAL_MV']

# 拟合 PanelOLS 模型，无时间效应，且不进行标准误的聚类调整
model1 = PanelOLS(y, X, weights=weights, entity_effects=False)
results1 = model1.fit()  # 使用默认的标准误拟合模型

# 输出回归结果摘要 (非聚类标准误模型)
print("----- 非聚类标准误模型回归结果摘要 -----")
print(results1.summary)

# 计算残差的 t 值和 p 值
resid1 = results1.resids
mean_resid1 = np.mean(resid1)
std_resid1 = np.std(resid1, ddof=1)  # 使用样本标准差
n1 = len(resid1)

# 计算 t 值
t_statistic1 = mean_resid1 / (std_resid1 / np.sqrt(n1))

# 计算 p 值 (双尾检验)
p_value1 = 2 * (1 - stats.t.cdf(np.abs(t_statistic1), df=n1 - 1))

# 输出残差的显著性
print("----- 无时间效应模型的残差显著性 -----")
print(f"残差 t值: {t_statistic1:.4f}")
print(f"残差 p值: {p_value1}")

# 使用聚类标准误进行拟合
results2 = model1.fit(cov_type='clustered', cluster_entity=True)

# 输出回归结果摘要 (实体聚类标准误模型)
print("----- 聚类标准误模型回归结果摘要 -----")
print(results2.summary)

# 使用聚类标准误进行拟合，按时间聚类
results3 = model1.fit(cov_type='clustered', cluster_time=True)

# 输出回归结果摘要 (时间聚类模型)
print("----- 时间聚类模型回归结果摘要 -----")
print(results3.summary)

# 使用聚类标准误进行拟合，按实体和时间聚类
results4 = model1.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

# 输出回归结果摘要 (实体和时间聚类模型)
print("----- 实体和时间聚类模型回归结果摘要 -----")
print(results4.summary)

# 使用时间固定效应模型
model2 = PanelOLS(y, X, weights=weights, entity_effects=False, time_effects=True)

# 1.
# 默认标准误拟合 (无聚类)
results5 = model2.fit()

# 输出回归结果摘要 (无聚类)
print("----- 默认标准误模型回归结果摘要 -----")
print(results5.summary)

# 计算残差的 t 值和 p 值
resid2 = results5.resids
mean_resid2 = np.mean(resid2)
std_resid2 = np.std(resid2, ddof=1)  # 使用样本标准差
n2 = len(resid2)

# 计算 t 值
t_statistic2 = mean_resid2 / (std_resid2 / np.sqrt(n2))

# 计算 p 值 (双尾检验)
p_value2 = 2 * (1 - stats.t.cdf(np.abs(t_statistic2), df=n1 - 1))

# 输出残差的显著性
print("----- 时间效应模型的残差显著性 -----")
print(f"残差 t值: {t_statistic2:.4f}")
print(f"残差 p值: {p_value2:.4f}")

'''# 2. 使用实体聚类标准误拟合
results6 = model2.fit(cov_type='clustered', cluster_entity=True)

# 输出回归结果摘要 (实体聚类标准误模型)
print("----- 实体聚类模型回归结果摘要 -----")
print(results6.summary)

# 检查 'COUNTRY' 参数的显著性 (实体聚类模型)
print("----- 实体聚类模型的 COUNTRY 参数显著性 -----")
print(f"COUNTRY 系数 (实体聚类): {results6.params['COUNTRY']:.4f}")
print(f"COUNTRY t值 (实体聚类): {results6.tstats['COUNTRY']:.4f}")
print(f"COUNTRY p值 (实体聚类): {results6.pvalues['COUNTRY']:.4f}")

# 3. 使用时间聚类标准误拟合
results7 = model2.fit(cov_type='clustered', cluster_time=True)

# 输出回归结果摘要 (时间聚类标准误模型)
print("----- 时间聚类模型回归结果摘要 -----")
print(results7.summary)

# 检查 'COUNTRY' 参数的显著性 (时间聚类模型)
print("----- 时间聚类模型的 COUNTRY 参数显著性 -----")
print(f"COUNTRY 系数 (时间聚类): {results7.params['COUNTRY']:.4f}")
print(f"COUNTRY t值 (时间聚类): {results7.tstats['COUNTRY']:.4f}")
print(f"COUNTRY p值 (时间聚类): {results7.pvalues['COUNTRY']:.4f}")

# 4. 使用实体和时间聚类标准误拟合
results8 = model2.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

# 输出回归结果摘要 (实体和时间聚类模型)
print("----- 实体和时间聚类模型回归结果摘要 -----")
print(results8.summary)

# 检查 'COUNTRY' 参数的显著性 (实体和时间聚类模型)
print("----- 实体和时间聚类模型的 COUNTRY 参数显著性 -----")
print(f"COUNTRY 系数 (实体和时间聚类): {results8.params['COUNTRY']:.4f}")
print(f"COUNTRY t值 (实体和时间聚类): {results8.tstats['COUNTRY']:.4f}")
print(f"COUNTRY p值 (实体和时间聚类): {results8.pvalues['COUNTRY']:.4f}")
'''