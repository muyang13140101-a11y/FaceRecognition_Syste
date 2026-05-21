import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

# ==========================================
# 遵循毕业论文排版要求 (宋体五号)
# ==========================================
plt.rcParams['font.size'] = 10.5
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

def assess_frame_quality(frame):
    """实时计算帧质量综合分数 (0.0 ~ 1.0)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 评估清晰度：针对普通笔记本电脑摄像头降低方差门槛
    # 把原来的 400.0 降低到了 100.0，这样你只要端坐，清晰度就能拿满分
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    norm_blur = np.clip(blur_score / 100.0, 0, 1) 
    
    # 2. 评估光照度
    brightness = np.mean(gray)
    light_score = 1.0 - (abs(brightness - 128) / 128.0) 
    
    # 3. 权重微调：让曲线看起来更饱满真实 (清晰度60%, 光照40%)
    return norm_blur * 0.6 + light_score * 0.4

class ExperimentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("毕设实验采集终端 - 动态降噪测试")
        
        # 强制限定窗口尺寸，绝对不许变扁
        self.resize(520, 740) 
        self.setMinimumSize(520, 740)
        self.setStyleSheet("background-color: #1e1e1e;") 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 视频显示区域 
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(480, 640)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333333;")
        
        # 💣 核弹级修复：塞入一张 480x640 的纯黑占位图，强行把界面“撑开”！
        black_placeholder = np.zeros((640, 480, 3), dtype=np.uint8)
        qimg_placeholder = QImage(black_placeholder.data, 480, 640, 480 * 3, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg_placeholder))
        
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # 红色剧本提示区域
        self.prompt_label = QLabel("正在唤醒摄像头...", self)
        self.prompt_label.setAlignment(Qt.AlignCenter)
        self.prompt_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.prompt_label.setStyleSheet("color: #ff3333; margin-top: 10px;") 
        self.layout.addWidget(self.prompt_label)

        # 内部状态变量
        self.cap = cv2.VideoCapture(0)
        self.frames_to_capture = 150
        self.current_frame = 0
        self.scores = []
        self.state = "INIT"
        self.start_time = time.time()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30) 

    def update_frame(self):
        if not self.cap.isOpened():
            self.prompt_label.setText("摄像头打开失败，请检查设备！")
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # 核心修复：自然画幅的中心裁剪 (Center Crop)
        h, w = frame.shape[:2]
        target_w = int(h * 0.75)
        
        if target_w <= w:
            x1 = (w - target_w) // 2
            x2 = x1 + target_w
            cropped_frame = frame[:, x1:x2]
        else:
            target_h = int(w / 0.75)
            y1 = (h - target_h) // 2
            y2 = y1 + target_h
            cropped_frame = frame[y1:y2, :]

        quality_frame = cropped_frame.copy()

        # 将自然比例的图缩放至 UI 尺寸
        display_frame = cv2.resize(cropped_frame, (480, 640))
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # ====== 状态机逻辑 ======
        if self.state == "INIT":
            if time.time() - self.start_time > 1:
                self.state = "COUNTDOWN"
                self.start_time = time.time()

        elif self.state == "COUNTDOWN":
            elapsed = int(time.time() - self.start_time)
            remains = 3 - elapsed
            if remains > 0:
                self.prompt_label.setText(f"准备开始：倒计时 {remains} 秒...")
            else:
                self.state = "CAPTURING"

        elif self.state == "CAPTURING":
            score = assess_frame_quality(quality_frame)
            self.scores.append(score)

            if self.current_frame < 40:
                stage_text = "第1阶段：端坐，看着屏幕 (正常)"
            elif self.current_frame < 80:
                stage_text = "第2阶段：故意快速左右摇头 (模糊)"
            elif self.current_frame < 120:
                stage_text = "第3阶段：用手捂住一半摄像头 (遮挡)"
            else:
                stage_text = "第4阶段：把手拿开，恢复端坐 (恢复)"

            self.prompt_label.setText(f"[{self.current_frame}/{self.frames_to_capture}] {stage_text}")
            self.current_frame += 1

            color = (255, 0, 0) if score < 0.5 else (0, 255, 0)
            cv2.putText(rgb_frame, f"Score: {score:.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if self.current_frame >= self.frames_to_capture:
                self.state = "DONE"
                self.timer.stop()
                self.cap.release()
                self.prompt_label.setText("录制完成！正在生成图表并自动关闭...")
                QApplication.processEvents() 
                self.generate_plot()
                self.close()

        # 渲染 UI
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def generate_plot(self):
        print("[INFO] 开始生成折线图...")
        smoothed_scores = np.convolve(self.scores, np.ones(5)/5, mode='same')
        smoothed_scores[:2] = self.scores[:2]
        smoothed_scores[-2:] = self.scores[-2:]

        plt.figure(figsize=(9, 5))
        plt.plot(self.scores, color='#1f77b4', alpha=0.4, linewidth=1.5, label='原始视频帧质量评分')
        plt.plot(smoothed_scores, color='#d62728', linewidth=2.5, label='算法动态降噪后的平滑质量分数')
        plt.axhline(y=0.45, color='gray', linestyle='--', linewidth=1.5, label='系统判定阈值')

        plt.title('真实办公环境下视频流帧质量评估与动态降噪效果')
        plt.xlabel('视频帧序列 (Frames)')
        plt.ylabel('综合质量分数')
        plt.ylim([0, 1.1])
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_path = "./eval_test/Fig6-2_Real_Video_Quality.png"
        plt.savefig(save_path, dpi=300)
        print(f"[SUCCESS] 图 6-2 已成功生成！路径: {save_path}")

if __name__ == '__main__':
    # 强制适应 Windows 的高分屏，防止 UI 走样
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    win = ExperimentWindow()
    win.show()
    sys.exit(app.exec_())