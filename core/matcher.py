import numpy as np
from utils.db_helper import DBHelper

class FaceMatcher:
    def __init__(self, threshold=0.6):
        """
        初始化特征比对器
        :param threshold: 认定为本人的及格线（0.0 到 1.0 之间，越大越严格）
        """
        self.threshold = threshold
        self.db = DBHelper()
        
        # 系统启动时，一次性把底库里所有人的特征加载到内存里，保证比对速度
        print("[INFO] 正在将数据库特征加载至内存...")
        self.known_users = self.db.get_all_users()
        print(f"[INFO] 成功加载 {len(self.known_users)} 个人员特征。")

    def match(self, unknown_feature):
        """
        传入一个未知的 512 维特征，去底库中寻找最匹配的人
        """
        if len(self.known_users) == 0:
            return "Unknown", 0.0

        best_match_name = "Unknown"
        highest_similarity = -1.0

        # 遍历底库中的每一个人
        for name, known_feature in self.known_users:
            # 第一性原理：计算余弦相似度
            # 公式: 向量点积 / (向量A的模长 * 向量B的模长)
            dot_product = np.dot(unknown_feature, known_feature)
            norm_a = np.linalg.norm(unknown_feature)
            norm_b = np.linalg.norm(known_feature)
            
            # 防止除以 0 的情况
            if norm_a == 0 or norm_b == 0:
                continue
                
            similarity = dot_product / (norm_a * norm_b)

            # 找出相似度最高的那个人
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_name = name

        # 检验是否过了及格线
        if highest_similarity >= self.threshold:
            return best_match_name, highest_similarity
        else:
            # 就算是最高分也没及格，说明是个陌生人
            return "Unknown", highest_similarity
        
        # 在 matcher.py 的 FaceMatcher 类最后加上这个方法
    def reload(self):
        """重新从数据库加载特征，用于动态录入后更新记忆"""
        print("[INFO] 正在刷新内存中的人员特征库...")
        self.known_users = self.db.get_all_users()
        print(f"[INFO] 刷新完成，当前特征库人数: {len(self.known_users)}")