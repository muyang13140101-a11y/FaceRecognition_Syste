import cv2
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        """
        初始化人脸检测器
        原理解释：在这里实例化 MTCNN()，模型权重会在这一步加载到内存中。
        我们把它封装在 __init__ 里，是为了避免每次检测图片时都重复加载模型，浪费 CPU 算力。
        """
        # 如果你的 MTCNN 库支持 device 参数，可以加上 device='CPU:0'
        # 但标准的 ipazc/mtcnn 在没有 GPU 环境下会自动 fallback 到 CPU
        self.detector = MTCNN() 
        print("[INFO] MTCNN 检测器初始化成功，运行在 CPU 模式。")

    def detect_and_crop(self, image_array):
        """
        传入一张图片（OpenCV 格式的 numpy 数组），返回截取出来的人脸图片列表
        """
        # 1. MTCNN 要求输入 RGB 格式的图片，而 OpenCV 默认读取的是 BGR 格式
        # 原理解释：颜色通道颠倒会导致模型提取特征完全错误，所以必须先转换。
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # 2. 调用模型检测人脸
        # 结果是一个列表，包含图里找到的所有人脸的信息（坐标框和5个关键点）
        results = self.detector.detect_faces(rgb_image)
        
        cropped_faces = []
        
        # 3. 遍历检测到的所有人脸
        for result in results:
            # 置信度极高（>0.95）的我们才认为是真的人脸，滤除误判（比如把灯泡当成脸）
            if result['confidence'] > 0.95:
                # 获取人脸框的坐标：x(左上角水平坐标), y(左上角垂直坐标), width(宽), height(高)
                x, y, width, height = result['box']
                
                # 修复一个 MTCNN 的历史遗留 Bug：有时返回的 x 或 y 会是负数，导致数组越界报错
                x, y = abs(x), abs(y) 
                
                # 4. 关键步骤：利用 numpy 的切片功能，把人脸区域的像素“抠”出来
                # 原理解释：图像在 Python 里本质上是个多维数组，image[y_start:y_end, x_start:x_end]
                face_crop = image_array[y : y + height, x : x + width]
                
                cropped_faces.append((face_crop, result['box']))
                
        return cropped_faces