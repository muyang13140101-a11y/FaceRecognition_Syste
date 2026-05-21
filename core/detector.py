import cv2
import math
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        """
        初始化人脸检测器
        本系统在 PC 端运行，算力充沛。采用 MTCNN 进行高精度感知，
        为后续的 2D 仿射对齐提供 5 点拓扑坐标。
        """
        self.detector = MTCNN() 
        print("[INFO] MTCNN 检测器初始化成功，准备执行高精度感知与仿射对齐。")

    def detect_and_crop(self, image_array):
        """
        传入图像，执行联合检测与仿射旋转对齐，返回端正的人脸切片列表。
        """
        # 1. 颜色通道转换
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # 2. 调用模型检测人脸
        results = self.detector.detect_faces(rgb_image)
        
        cropped_faces = []
        
        # 3. 遍历检测到的所有人脸
        for result in results:
            # 仅处理置信度极高的真切人脸
            if result['confidence'] > 0.95:
                x, y, width, height = result['box']
                x, y = abs(x), abs(y) 
                
                # ==========================================
                # 核心升级：基于双眼坐标的 2D 仿射变换与纠偏
                # ==========================================
                # 提取 MTCNN 同步输出的面部关键点
                keypoints = result['keypoints']
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']
                
                # A. 计算双眼在二维平面的倾斜角 (反正切解算)
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = math.degrees(math.atan2(dy, dx))
                
                # B. 确定旋转轴心 (以面部中心为轴，防止旋转后原边界框截断下巴)
                center_x = x + width // 2
                center_y = y + height // 2
                
                # C. 动态死区滤波器 (Deadband Filter)
                # 作用：PC 摄像头视频流中，关键点会有 1-2 像素的微弱抖动。
                # 如果角度极小（< 3度），强行旋转会导致画面高频抽搐。
                if abs(angle) > 3.0:
                    # 生成 2D 刚性旋转仿射矩阵
                    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale=1.0)
                    
                    # 获取原图尺寸并执行双三次插值重采样
                    h_img, w_img = image_array.shape[:2]
                    aligned_img = cv2.warpAffine(image_array, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
                    
                    # 在旋转纠偏后的端正图像上，执行正交裁剪
                    y_end = min(y + height, h_img)
                    x_end = min(x + width, w_img)
                    face_crop = aligned_img[y : y_end, x : x_end]
                else:
                    # 角度极小，视为纯净正脸或白噪声，跳过仿射计算直接裁剪
                    h_img, w_img = image_array.shape[:2]
                    y_end = min(y + height, h_img)
                    x_end = min(x + width, w_img)
                    face_crop = image_array[y : y_end, x : x_end]
                
                # 确保裁剪出的是有效图像防崩溃
                if face_crop.size > 0:
                    cropped_faces.append((face_crop, result['box']))
                
        return cropped_faces