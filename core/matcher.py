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
            # ==========================================
            # 核心数学优化：超球面特征的极速余弦计算
            # ==========================================
            # 由于在 extractor.py 中已经进行了严格的 L2 范数归一化 (||A||=1, ||B||=1)
            # 余弦相似度计算公式直接退化为向量的纯点积 (Dot Product)
            # 彻底省去了极其耗时的平方求和与开根号运算
            similarity = float(np.dot(unknown_feature, known_feature))

            # 找出相似度最高的那个人
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_name = name
            print(f"[DEBUG] 当前镜头人脸最高匹配度 -> {best_match_name}: {highest_similarity:.4f}")

        # 检验是否过了及格线
        if highest_similarity >= self.threshold:
            return best_match_name, highest_similarity
        else:
            # 就算是最高分也没及格，说明是个陌生人
            return "Unknown", highest_similarity
        
    def reload(self):
        """重新从数据库加载特征，用于动态录入后更新记忆"""
        print("[INFO] 正在刷新内存中的人员特征库...")
        self.known_users = self.db.get_all_users()
        print(f"[INFO] 刷新完成，当前特征库人数: {len(self.known_users)}")