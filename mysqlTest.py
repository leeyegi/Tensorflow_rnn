'''
import MySQLdb
"""
db=MySQLdb.connect("localhost", "root", "123456", "testdb")
cursor=db.cursor()

cursor.execute("SELECT VERSION()")
data=cursor.fetchone()
print("Database version : %s" %(data))

db.close()
"""

#import pymysql

# MySQL Connection 연결
conn = pymysql.connect(host='localhost', user='root', password='123456',
                       db='testdb', charset='utf8')

# Connection 으로부터 Cursor 생성
curs = conn.cursor()

# SQL문 실행
sql = "select * from testtable"
curs.execute(sql)

# 데이타 Fetch
rows = curs.fetchall()
print(rows)  # 전체 rows

# Connection 닫기
conn.close()'''