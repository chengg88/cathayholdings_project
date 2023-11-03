'''''''''
與資料庫相關程式撰寫位置，包括DB連結、SQL語法撰寫等...
'''''''''
import pymysql
import time


class DB_manger:
    def __init__(self, host, user, password, db):
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.conn = None
        self.cur = None

    def connect(self):
        self.conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            db=self.db,
            charset='utf8'
        )
        self.cur = self.conn.cursor()

    def close(self):
        self.cur.close()
        self.conn.close()

    def execute(self, sql):
        self.cur.execute(sql)
        self.conn.commit()

    def fetchall(self):
        return self.cur.fetchall()

    def fetchone(self):
        return self.cur.fetchone()

    def get_one(self, sql):
        self.execute(sql)
        return self.fetchone()