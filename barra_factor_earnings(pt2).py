# -*- coding = utf-8 -*-
# @Time: 2024/09/23
# @Author: Xinyue
# @File:barra_factor_earnings(pt2).py
# @Software: PyCharm

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
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

'''# 定义目标函数：加权最小二乘
def objective_function(beta, X, y, weights):
    residuals = y - X @ beta
    weighted_residuals = residuals * np.sqrt(weights)
    return np.sum(weighted_residuals**2)

# 定义约束条件：s_{I1} * f_{I1} + s_{I2} * f_{I2} + ... = 0
def industry_constraint(beta, industry_weights):
    industry_factors = beta[-len(industry_weights):]  # 最后几项是行业因子的系数
    return np.dot(industry_weights, industry_factors)

def weighted_ols_with_constraint(df_with_dummies, max_tries=10000000):
    # 获取行业虚拟变量列
    industry_columns = [col for col in df_with_dummies if col.startswith('WIND_PRI_IND_')]

    # 将行业虚拟变量列添加到因子列中
    X = df_with_dummies[factor_columns + industry_columns].values
    y = df_with_dummies['RETURN_T1'].values
    df_with_dummies['S_VAL_MV'] = pd.to_numeric(df_with_dummies['S_VAL_MV'])
    weights = df_with_dummies['S_VAL_MV'].values # 使用市值作为权重

    # 计算行业权重
    industry_weights = [df_with_dummies.loc[df_with_dummies[col] == 1, 'S_VAL_MV'].sum() for col in industry_columns]
    print(industry_weights)

    # 初始猜测的回归系数
    beta_init = np.zeros(X.shape[1])

    # 定义线性约束
    constraints = {'type': 'eq', 'fun': industry_constraint, 'args': (industry_weights,)}

    # 初始最大迭代次数
    maxiter = 100

    while maxiter <= max_tries:
        print(f"Trying optimization with maxiter {maxiter}")
        # 执行带约束的优化
        result = minimize(objective_function, beta_init, args=(X, y, weights), constraints=constraints,
                          options={'maxiter': maxiter})

        if result.success:
            print("Optimization successful.")
            return pd.Series(result.x, index=factor_columns + industry_columns)
        else:
            print(f"Optimization failed with maxiter {maxiter}: {result.message}")
            # 增加迭代次数
            maxiter *= 10

    # 如果超过 max_tries 还没有成功，打印错误信息并返回 NaN
    print(f"Optimization failed after trying maxiter up to {max_tries}")
    return pd.Series([None] * len(beta_init), index=factor_columns + industry_columns)

# 按 'TRADE_DT' 分组，并对每一天应用回归
#factor_returns_by_day = factor_df_with_dummies.groupby('TRADE_DT').apply(weighted_ols_with_constraint, include_groups=False)

# Reset the index to make 'TRADE_DT' a column
factor_returns_by_day = factor_returns_by_day.reset_index()

# 获取包涵缺失值的日期
missing_dates = factor_returns_by_day.loc[factor_returns_by_day.isna().any(axis=1), 'TRADE_DT']

# 对于每个缺失值日期，提取子集并尝试重新计算因子收益
for trade_date in missing_dates:
    subset_df = factor_df_with_dummies[factor_df_with_dummies['TRADE_DT'] == trade_date]
    new_factor_returns = weighted_ols_with_constraint(subset_df)
    factor_returns_by_day.loc[factor_returns_by_day['TRADE_DT'] == trade_date, new_factor_returns.index] = new_factor_returns.values

print(factor_returns_by_day.columns)
factor_returns_by_day.to_parquet('/Volumes/quanyi4g/factor/day_frequency/barra/factor_return.parquet')'''

factor_returns_by_day = pd.read_parquet('/Volumes/quanyi4g/factor/day_frequency/barra/factor_return.parquet')
# 提取所有因子收益列，排除 'TRADE_DT'
factor_columns = [col for col in factor_returns_by_day.columns if col != 'TRADE_DT']

# 初始化列表来存储结果
factors = []
t_values = []
p_values = []

# 遍历每个因子列
for factor in factor_columns:
    factor_data = factor_returns_by_day[factor].dropna()  # 去掉 NaN 值

    # 计算平均收益和标准差
    mean_return = factor_data.mean()  # 计算均值
    std_return = factor_data.std()  # 计算标准差

    # 计算 t 值（均值除以标准误）
    t_value = mean_return / (std_return / np.sqrt(len(factor_data)))  # t值计算公式

    # 从 t 值计算 p 值
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df=len(factor_data) - 1))  # 双尾 p值计算

    # 将结果添加到列表中
    factors.append(factor)
    t_values.append(t_value)
    p_values.append(p_value)

# 创建一个 DataFrame 来存储结果
significance_df = pd.DataFrame({
    'Factor': factors,  # 因子名称
    'T-Value': t_values,  # t 值
    'P-Value': p_values  # p 值
})

# 显示结果
print(significance_df)


'''# 计算残差
def calculate_residuals(factor_df, factor_returns):
    residuals_dict = {}

    for trade_date in factor_df['TRADE_DT'].unique():
        # 获取特定交易日的数据
        daily_data = factor_df[factor_df['TRADE_DT'] == trade_date]

        # 计算因子暴露与因子收益的乘积
        factor_exposure = daily_data[factor_columns].values
        factor_return = factor_returns[factor_returns['TRADE_DT'] == trade_date].drop(columns='TRADE_DT').values

        # 计算预测收益
        predicted_returns = factor_exposure @ factor_return.T  # 矩阵乘法，得到预测收益

        # 计算残差
        residuals = daily_data['RETURN_T1'].values - predicted_returns.flatten()

        # 存储残差
        residuals_dict[trade_date] = residuals

    return residuals_dict


# 计算所有交易日的残差
residuals_dict = calculate_residuals(factor_df_with_dummies, factor_returns_by_day)


# 计算 t 值和 p 值
def calculate_t_and_p(residuals_dict):
    t_values = []
    p_values = []

    for trade_date, residuals in residuals_dict.items():
        if len(residuals) > 1:  # 确保有足够的样本计算 t 值和 p 值
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals, ddof=1)  # 样本标准差
            n = len(residuals)  # 样本量

            # 计算 t 值
            t_value = mean_residual / (std_residual / np.sqrt(n))
            t_values.append(t_value)

            # 计算 p 值
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df=n - 1))  # 使用 t 分布的 CDF
            p_values.append(p_value)
        else:
            t_values.append(np.nan)
            p_values.append(np.nan)

    return t_values, p_values


# 计算 t 值和 p 值
t_values, p_values = calculate_t_and_p(residuals_dict)

# 创建结果 DataFrame
results_df = pd.DataFrame({
    'TRADE_DT': list(residuals_dict.keys()),
    't_value': t_values,
    'p_value': p_values
})

# 输出结果
print(results_df)
'''
'''# Test significance
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

# 2. 使用实体聚类标准误拟合
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