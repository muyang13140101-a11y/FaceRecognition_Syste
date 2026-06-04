import cv2
import numpy as np

class QualityAssessor:
    def __init__(self, window_size=5, calibration_frames=30):
        # 时序平滑窗口
        self.window_size = window_size
        self.score_history = []
        
        # 🌟 环境自适应标定引擎
        self.calibration_frames = calibration_frames
        self.baseline_scores = []
        self.is_calibrated = False
        self.dynamic_threshold = 0.0 # 初始为0，全放行

    def assess_frame(self, frame):
        """
        评估单帧质量，并自动进行环境标定
        返回: (is_valid: bool, smoothed_score: float)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 记录历史分数用于平滑
        self.score_history.append(lap_var)
        if len(self.score_history) > self.window_size:
            self.score_history.pop(0)
        smoothed_score = np.mean(self.score_history)

        # 🌟 核心引擎：黑盒内的自适应冷启动
        if not self.is_calibrated:
            self.baseline_scores.append(smoothed_score)
            # 如果收集满了指定帧数，立即生成阻断防线
            if len(self.baseline_scores) >= self.calibration_frames:
                base_mean = np.mean(self.baseline_scores)
                # 按照论文 6.4 节逻辑：取基准均值的 65% 作为死锁阈值
                self.dynamic_threshold = base_mean * 0.65
                self.is_calibrated = True
                print(f"[底层内核] 质量评估器标定完毕！基准: {base_mean:.1f}, 阈值锁定: {self.dynamic_threshold:.1f}")
            
            # 校准期间，绝对放行，不拦截
            return True, smoothed_score

        # 标定完成后，开始严格的门控执法
        is_valid = smoothed_score >= self.dynamic_threshold
        return is_valid, smoothed_score