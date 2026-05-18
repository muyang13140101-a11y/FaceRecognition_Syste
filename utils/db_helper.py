import sqlite3
import numpy as np
import os

class DBHelper:
    def __init__(self, db_path="data/faces.db"):
        """
        初始化数据库连接，如果库文件不存在会自动创建
        """
        self.db_path = db_path
        # 确保存放数据库的目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _create_table(self):
        """
        核心原理解释：使用 BLOB 类型来存储 512 维的二进制特征向量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                feature BLOB NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def insert_user(self, name, feature_vector):
        """
        将提取好的人脸特征存入数据库
        """
        # 第一性原理：将 numpy 数组转化为极其紧凑的连续 C 语言级别的字节流
        feature_blob = feature_vector.tobytes()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, feature) VALUES (?, ?)", (name, feature_blob))
        conn.commit()
        conn.close()
        print(f"[DB] 成功写入底层数据库: {name}")

    def get_all_users(self):
        """
        系统启动时，将底库中所有人的特征全部加载到内存中用于比对
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, feature FROM users")
        rows = cursor.fetchall()
        conn.close()

        users = []
        for row in rows:
            name = row[0]
            # 逆向操作：将 BLOB 字节流重新解析为 numpy 浮点数组 (float32)
            feature_vector = np.frombuffer(row[1], dtype=np.float32)
            users.append((name, feature_vector))
        return users