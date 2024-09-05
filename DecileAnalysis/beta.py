# -*- coding = utf-8 -*-
# @Time: 2024/09/03
# @Author: Xinyue
# @File:residual_vol.py
# @Software: PyCharm


import pandas as pd


from barra_cne5_factor import Beta, ResidualVolatility

# 全市场去除北交所
all_mkt_price_df = pd.read_parquet('../Data/all_market_data.parquet')
all_mkt_price_df = all_mkt_price_df[~all_mkt_price_df['S_INFO_WINDCODE'].str.endswith('BJ')].reset_index(drop=True)

# 创建计算Beta实例
calculate_beta = Beta(all_mkt_price_df)
beta_df = calculate_beta.beta_df
beta_df.to_parquet('../Data/alpha_beta_all_market.parquet', index=False)
