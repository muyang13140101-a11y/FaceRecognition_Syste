import cv2
import numpy as np
from collections import deque

class QualityAssessor:
    def __init__(self, window_size=5, blur_threshold=60.0):
        """
        初始化质量评估器 (完美对应论文第 5 章)
        :param window_size: 时序滑动窗口的长度 (对应论文 5.3 节)
        :param blur_threshold: 拉普拉斯清晰度及格线 (对应论文 5.4 节劣质帧剔除)
        """
        self.window_size = window_size
        self.blur_threshold = blur_threshold
        # 使用双端队列模拟滑动窗口，存储最近 N 帧的质量分数
        self.score_history = deque(maxlen=window_size)

    def assess_frame(self, face_image):
        """
        对单张人脸图像进行物理质量评分
        """
        # 1. 转换为灰度图 (对应论文 5.2 节)
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # ==========================================
        # 核心算子 A：拉普拉斯二阶导数评估清晰度 (对应公式 5-1)
        # ==========================================
        # 运动模糊会导致边缘高频信息丢失，方差暴跌
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        laplacian_score = min(laplacian_var * 5.0, 100.0)

        # ==========================================
        # 核心算子 B：全局均值评估光照度 (对应公式 5-2)
        # ==========================================
        # 理想曝光应当在 128 左右
        mean_brightness = np.mean(gray)
        # 计算光照惩罚（太暗或太曝都会扣分，非线性惩罚）
        brightness_penalty = abs(mean_brightness - 128) / 128.0 
        
        # 综合单帧得分 (以拉普拉斯为主，扣除光照惩罚) (对应公式 5-3)
        single_frame_score = laplacian_var * (1.0 - 0.2 * brightness_penalty)

        # ==========================================
        # 核心算法 C：一维滑动窗口时序滤波 (对应公式 5-4, 5-5)
        # ==========================================
        self.score_history.append(single_frame_score)
        
        # 1D 卷积平滑：计算窗口内的加权平均值，消除瞬时高频毛刺
        smoothed_score = sum(self.score_history) / len(self.score_history)

        # ==========================================
        # 核心机制 D：劣质帧前置阻断 (对应 5.4 节)
        # ==========================================
        # 如果平滑后的分数低于阈值，判定为严重的运动模糊或光照遮挡
        is_valid = smoothed_score > self.blur_threshold

        return is_valid, smoothed_score