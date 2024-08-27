# -*- coding = utf-8 -*-

import pandas as pd
import pymysql


class ConnectDatabase:
    """连接数据库组件"""
    def __init__(self, sql):
        """
        :param sql: 查询语句
        """
        self.db_config = {
                            'host': '192.168.7.93',
                            'port': 3306,
                            'username': 'quantchina',
                            'password': 'zMxq7VNYJljTFIQ8',
                            'database': 'wind'
                            }
        self.sql = sql

    def connect(self):
        """连接数据库"""
        host = self.db_config['host']
        port = self.db_config['port']
        username = self.db_config['username']
        password = self.db_config['password']
        database = self.db_config['database']

        try:
            conn = pymysql.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database
            )
            return conn
        except Exception as e:
            print(f'Error connecting to database:{e}')

    def get_data(self):
        """获取数据"""
        if self.connect() is not None:
            with self.connect().cursor() as cursor:
                cursor.execute(self.sql)
                data = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                data = pd.DataFrame(list(data), columns=columns)
                if data.empty:
                    return data
            return data
        else:
            print('connection failed')

