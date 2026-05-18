import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSpacerItem, QSizePolicy

# 引入 Fluent 丰富的高级动效组件
from qfluentwidgets import (FluentWindow, SubtitleLabel, setTheme, Theme, 
                            PrimaryPushButton, PushButton, FluentIcon as FIF, 
                            setThemeColor, TextEdit, LineEdit, ProgressRing, InfoBar, InfoBarPosition)

from core.detector import FaceDetector
from core.extractor import FaceExtractor
from core.matcher import FaceMatcher
from utils.db_helper import DBHelper

# ==========================================
# 核心 AI 状态机线程
# ==========================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_log_signal = pyqtSignal(str)
    enroll_progress_signal = pyqtSignal(int)
    enroll_success_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.state = "STANDBY" # 三种状态: STANDBY (待机), MONITOR (监控), ENROLL (录入)
        self.detector = FaceDetector()
        self.extractor = FaceExtractor()
        self.matcher = FaceMatcher(threshold=0.6)
        self.db = DBHelper()
        
        # 动态录入专属变量
        self.enroll_name = ""
        self.enroll_features = []
        self.enroll_target_frames = 5 # 采集 5 帧清晰画面进行平均融合，提高精准度

    def set_state(self, state, name=""):
        self.state = state
        self.enroll_name = name
        self.enroll_features = []

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: continue

            if self.state == "STANDBY":
                # 待机模式：降低画质消耗，绘制待机界面
                frame = cv2.GaussianBlur(frame, (51, 51), 0)
                cv2.putText(frame, "SYSTEM STANDBY", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            elif self.state == "MONITOR":
                # 正常监控模式
                cropped_faces = self.detector.detect_and_crop(frame)
                for face_crop, box in cropped_faces:
                    x, y, w, h = box
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    feature = self.extractor.extract_feature(face_crop_rgb)
                    name, confidence = self.matcher.match(feature)

                    color = (255, 255, 255) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = f"{name} ({confidence*100:.1f}%)"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if name != "Unknown":
                        self.update_log_signal.emit(f"[已授权] 识别: {name} | 置信度: {confidence*100:.1f}%")
                    else:
                        self.update_log_signal.emit(f"[警告] 拦截未知人员！")

            elif self.state == "ENROLL":
                # 动态录入模式：寻找画面中最清晰的一张脸
                cropped_faces = self.detector.detect_and_crop(frame)
                if len(cropped_faces) > 0:
                    face_crop, box = cropped_faces[0] # 取第一张脸
                    x, y, w, h = box
                    
                    # 绘制扫描光效特效 (青蓝色框)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 204, 0), 2)
                    cv2.putText(frame, f"Scanning: {self.enroll_name}...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 204, 0), 2)

                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    feature = self.extractor.extract_feature(face_crop_rgb)
                    
                    # 收集有效特征
                    if np.sum(feature) != 0:
                        self.enroll_features.append(feature)
                        progress = int((len(self.enroll_features) / self.enroll_target_frames) * 100)
                        self.enroll_progress_signal.emit(progress)

                        # 采集完毕，融合并存入数据库
                        if len(self.enroll_features) >= self.enroll_target_frames:
                            # 多帧平均融合算法，大幅提升特征稳定性
                            final_feature = np.mean(self.enroll_features, axis=0)
                            self.db.insert_user(self.enroll_name, final_feature)
                            
                            self.matcher.reload() # 热更新记忆
                            self.state = "MONITOR" # 切回监控模式
                            self.enroll_success_signal.emit(self.enroll_name)
                else:
                    cv2.putText(frame, "Please face the camera", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            # 统一画面转码渲染
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 使用 SmoothTransformation 开启抗锯齿，大幅提升画质
            p = qimg.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.change_pixmap_signal.emit(p)

# ==========================================
# UI 模块：监控与录入整合面板
# ==========================================
class MonitorInterface(QWidget):
    def __init__(self, thread, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("MonitorInterface")
        self.thread = thread
        layout = QVBoxLayout(self)

        # 高画质视频显示区
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: rgba(0, 0, 0, 0.2); border-radius: 10px;")
        layout.addWidget(self.video_label, stretch=1)

        # 底部控制台：左右布局
        control_layout = QHBoxLayout()
        
        # 左侧：感知控制
        self.btn_start = PrimaryPushButton(FIF.PLAY, "启动感知网络", self)
        self.btn_stop = PushButton(FIF.PAUSE, "系统挂起", self)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        control_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # 右侧：动态无感录入台
        self.name_input = LineEdit(self)
        self.name_input.setPlaceholderText("输入新面孔姓名...")
        self.btn_enroll = PushButton(FIF.ADD, "开始无感扫描", self)
        
        # 科技感录入进度环 (初始隐藏)
        self.progress_ring = ProgressRing(self)
        self.progress_ring.setFixedSize(30, 30)
        self.progress_ring.hide()

        control_layout.addWidget(self.name_input)
        control_layout.addWidget(self.btn_enroll)
        control_layout.addWidget(self.progress_ring)

        layout.addLayout(control_layout)

        # 绑定点击动效与逻辑
        self.btn_start.clicked.connect(lambda: self.thread.set_state("MONITOR"))
        self.btn_stop.clicked.connect(lambda: self.thread.set_state("STANDBY"))
        self.btn_enroll.clicked.connect(self.start_enrollment)

    def start_enrollment(self):
        name = self.name_input.text().strip()
        if not name:
            InfoBar.error(title="错误", content="录入前必须填写对象代号 (姓名)", parent=self, position=InfoBarPosition.TOP_RIGHT)
            return
        
        self.progress_ring.setValue(0)
        self.progress_ring.show()
        self.name_input.setDisabled(True)
        self.btn_enroll.setDisabled(True)
        self.thread.set_state("ENROLL", name)

# ==========================================
# UI 模块：访问日志面板
# ==========================================
class LogInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("LogInterface")
        layout = QVBoxLayout(self)
        
        self.log_text = TextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas; font-size: 14px; background: transparent;")
        layout.addWidget(self.log_text)

# ==========================================
# 终极主窗口
# ==========================================
class FaceRecognitionApp(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEXUS 神经网络认证中枢")
        self.resize(1100, 750)
        
        # 深邃蓝毛玻璃风格设置
        setTheme(Theme.DARK)
        setThemeColor('#0055FF') 

        # 启动常驻底层引擎
        self.thread = VideoThread()
        self.thread.start() # 线程一开启就在跑，但默认是 STANDBY 待机模式，极其省电
        
        # 实例化页面
        self.monitor_interface = MonitorInterface(self.thread, self)
        self.log_interface = LogInterface(self)

        # 挂载导航栏 (带丰富图标)
        self.addSubInterface(self.monitor_interface, FIF.VIDEO, '感知阵列')
        self.addSubInterface(self.log_interface, FIF.HISTORY, '追踪日志')

        # 绑定底层信号更新 UI
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_log_signal.connect(self.log_interface.log_text.append)
        self.thread.enroll_progress_signal.connect(self.update_enroll_progress)
        self.thread.enroll_success_signal.connect(self.on_enroll_success)
        
    def update_image(self, q_img):
        # 核心修复：将后台的 QImage 转换为前端需要的 QPixmap
        self.monitor_interface.video_label.setPixmap(QPixmap.fromImage(q_img))
    def update_enroll_progress(self, val):
        self.monitor_interface.progress_ring.setValue(val)

    def on_enroll_success(self, name):
        self.monitor_interface.progress_ring.hide()
        self.monitor_interface.name_input.clear()
        self.monitor_interface.name_input.setDisabled(False)
        self.monitor_interface.btn_enroll.setDisabled(False)
        # 弹出顶级高级动画通知
        InfoBar.success(
            title="基因锁录入完成",
            content=f"对象 [{name}] 的生物特征已通过多帧融合加密存入底层数据库。",
            duration=3000,
            parent=self,
            position=InfoBarPosition.TOP_RIGHT
        )