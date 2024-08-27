# -*- coding = utf-8 -*-
# @Time: 2024/07/19
# @Author: Xinyue
# @File:xxx.py
# @Software: PyCharm

from datetime import datetime
import pandas as pd

from Utils.get_wind_data import WindData

wind_primary_industry = pd.read_csv('../Data/wind_primary_industry.csv',
                                    dtype={'INDUSTRIESCODE_PREFIX': str,
                                               'INDUSTRIESCODE': str})

START_DATE = START_DATE = '20071112'
END_DATE = datetime.now().strftime('%Y%m%d')
data = WindData(START_DATE, END_DATE)
ashareind_df = data.get_stock_ind()
ashareind_df['REMOVE_DT'] = ashareind_df['REMOVE_DT'].astype(str).str.split('.').str[0]
ashareind_df['REMOVE_DT'] = pd.to_datetime(ashareind_df['REMOVE_DT'], format='mixed', errors='coerce')
ashareind_df['ENTRY_DT'] = pd.to_datetime(ashareind_df['ENTRY_DT'].astype(str))
ashareind_df['WIND_IND_CODE'] = ashareind_df['WIND_IND_CODE'].astype(str)

def map_primary_industry(industry_code, industry_classifier):

    for id, cls in zip(industry_classifier['INDUSTRIESCODE_PREFIX'], industry_classifier['WIND_NAME_ENG']):
        if industry_code.startswith(id):
            return cls
    return None

ashareind_df['WIND_PRI_IND'] = ashareind_df['WIND_IND_CODE'].apply(
    lambda x: map_primary_industry(x, wind_primary_industry))
ashareind_df = ashareind_df[['S_INFO_WINDCODE', 'WIND_PRI_IND', 'ENTRY_DT', 'REMOVE_DT', 'CUR_SIGN']]

ashareind_df.to_parquet('Data/wind_pri_industry_Ashare.parquet')

