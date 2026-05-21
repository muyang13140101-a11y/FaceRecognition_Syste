import cv2
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        """
        初始化人脸检测器
        本系统在 PC 端运行，算力充沛。采用 MTCNN 进行高精度感知，
        为后续的 5点相似变换对齐 (Similarity Transform) 提供拓扑坐标。
        """
        self.detector = MTCNN() 
        print("[INFO] MTCNN 检测器初始化成功，已启用标准 5点拓扑归一化。")

        # ==========================================
        # 核心机密：ArcFace 官方标准的 112x112 面部基准点物理坐标
        # 顺序: [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
        # ==========================================
        self.reference_pts = np.array([
            [38.2946, 51.6963], 
            [73.5318, 51.5014], 
            [56.0252, 71.7366], 
            [41.5493, 92.3655], 
            [70.7299, 92.2041]  
        ], dtype=np.float32)

    def detect_and_crop(self, image_array):
        """
        传入图像，执行联合检测与 5点相似变换对齐，返回端正的 112x112 人脸切片列表。
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
                # 提取原图中的边界框，保留下来用于给 UI 界面画框
                box = result['box'] 
                x, y, width, height = box
                x, y = abs(x), abs(y)
                # 防止MTCNN的负数越界Bug，修复返回的 box
                safe_box = [x, y, width, height]
                
                # ==========================================
                # 终极升级：基于 5 个关键点的相似变换对齐 (无缝对接 ArcFace 理论)
                # ==========================================
                keypoints = result['keypoints']
                
                # A. 按标准顺序组织当前人脸的真实 5 点坐标
                src_pts = np.array([
                    keypoints['left_eye'],
                    keypoints['right_eye'],
                    keypoints['nose'],
                    keypoints['mouth_left'],
                    keypoints['mouth_right']
                ], dtype=np.float32)
                
                # B. 求解 4 自由度仿射变换矩阵 (包含 旋转、平移、尺度缩放)
                # 底层数学原理：利用最小二乘法，算出如何把当前歪七扭八的脸，完美拉扯映射到基准点上
                M, _ = cv2.estimateAffinePartial2D(src_pts, self.reference_pts)
                
                if M is None:
                    continue # 极其罕见的矩阵求解失败，则跳过
                
                # C. 一步到位：利用矩阵对原图进行重采样，直接切出一张完美的 112x112 正脸图
                # 这一步取代了之前的“计算角度 -> 旋转全图 -> 按框抠图 -> 强行缩放”的冗余操作
                aligned_face = cv2.warpAffine(image_array, M, (112, 112), borderValue=0.0)
                
                if aligned_face.size > 0:
                    cropped_faces.append((aligned_face, safe_box))
                
        return cropped_faces