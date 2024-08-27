# -*- coding = utf-8 -*-

from Utils.connect_wind import ConnectDatabase

class DataPerStock(ConnectDatabase):
    '''获取wind数据'''
    def __init__(self, start_date: str, end_date: str, stock_code: str):
        '''
        :param start_date: 起始时间 例：'20100101'
        :param end_date: 结束时间 例：datetime.datetime.now().strftime('%Y%m%d')
        :param stock_code: 股票WIND代码 例 '000001.SZ'
        '''
        self.start_date = start_date
        self.end_date = end_date
        self.stock_code = stock_code

    def price_data(self):
        sql = f'''SELECT
                    p.S_INFO_WINDCODE,
                    p.TRADE_DT,
                    p.S_DQ_PRECLOSE,
                    p.S_DQ_OPEN,
                    p.S_DQ_CLOSE,
                    p.S_DQ_TRADESTATUS,
                    p.S_DQ_LIMIT,
                    p.S_DQ_STOPPING,
                    p.S_DQ_AMOUNT,
                    d.NET_ASSETS_TODAY,
                    d.S_DQ_TURN,
                    d.S_VAL_MV
                FROM
                    ASHAREEODPRICES p
                JOIN
                    ASHAREEODDERIVATIVEINDICATOR d
                ON
                    p.S_INFO_WINDCODE = d.S_INFO_WINDCODE
                    AND p.TRADE_DT = d.TRADE_DT
                WHERE
                    p.S_INFO_WINDCODE = '{self.stock_code}'
                    AND d.S_INFO_WINDCODE = '{self.stock_code}'
                    AND p.TRADE_DT BETWEEN '{self.start_date}' AND '{self.end_date}';
                '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def list_info(self):
        sql = f'''
            SELECT S_INFO_WINDCODE, S_INFO_LISTDATE, S_INFO_DELISTDATE
            FROM ASHAREDESCRIPTION
            WHERE S_INFO_WINDCODE = '{self.stock_code}'
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        return df

    def suspend_info(self):
        sql = f'''
            SELECT S_INFO_WINDCODE, S_DQ_SUSPENDDATE, S_DQ_RESUMPDATE
            FROM ASHARETRADINGSUSPENSION
            WHERE S_INFO_WINDCODE = '{self.stock_code}'
                  AND S_DQ_SUSPENDDATE >= '{self.start_date}'
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        return df

    def st_info(self):
        sql = f'''SELECT S_INFO_WINDCODE, S_TYPE_ST, ENTRY_DT, REMOVE_DT, ANN_DT, REASON               
                  FROM ASHAREST
                  WHERE ENTRY_DT > {self.start_date} AND S_TYPE_ST != 'R'
                    AND S_INFO_WINDCODE = '{self.stock_code}'
                  '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        return df

    def major_event(self):
        sql = f'''
            SELECT S_INFO_WINDCODE, S_EVENT_CATEGORYCODE, S_EVENT_ANNCEDATE, S_EVENT_HAPDATE,
            S_EVENT_EXPDATE, S_EVENT_CONTENT
            FROM ASHAREMAJOREVENT
            WHERE (S_INFO_WINDCODE = '{self.stock_code}' AND S_EVENT_CATEGORYCODE = 204007001)
                OR (S_INFO_WINDCODE = '{self.stock_code}' AND S_EVENT_CATEGORYCODE = 204007005)
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        return df

    def bond_rt(self):
        sql = f'''
            SELECT TRADE_DT, S_DQ_CLOSE
            FROM CGBBENCHMARK
            WHERE S_INFO_WINDCODE = '{self.stock_code}' 
        '''

        connection = ConnectDatabase(sql)
        df = connection.get_data()
        df.sort_values(by='TRADE_DT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={'S_DQ_CLOSE': f'{self.stock_code}'}, inplace=True)
        return df

if __name__ == '__main__':
    single_data = DataPerStock('20071214', '2024722', 'GZHY.WI')
    brt = single_data.bond_rt()
    print(brt)
