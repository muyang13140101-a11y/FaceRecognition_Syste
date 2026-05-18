import os
import cv2
from core.detector import FaceDetector
from core.extractor import FaceExtractor
from utils.db_helper import DBHelper

def main():
    print("=== 启动系统底库注册程序 ===")
    
    # 1. 实例化三大核心组件
    detector = FaceDetector()
    extractor = FaceExtractor()
    db = DBHelper()

    base_dir = "data/faces"
    if not os.path.exists(base_dir):
        print(f"[错误] 找不到图库目录 {base_dir}，请先建立文件夹并放入照片！")
        return

    print("\n--- 开始批量遍历并提取特征 ---")
    
    # 2. 遍历每个人名的文件夹
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        # 遍历该人名下的所有照片
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # 使用 OpenCV 读取照片
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] 无法读取图片 {img_path}，已跳过。")
                continue
                
            print(f"正在处理: {person_name} - {img_name}")
            
            # 3. 找脸（抠图）
            cropped_faces = detector.detect_and_crop(img_bgr)
            if len(cropped_faces) == 0:
                print(f"   -> [跳过] 未检测到人脸。")
                continue
                
            # 默认取画面里检测到的第一张脸
            face_crop, _ = cropped_faces[0]
            
            # 4. 颜色空间转换 (极其关键的一步)
            # OpenCV 截取出来的图是 BGR 格式，但 PyTorch 训练模型时用的是 RGB 格式。
            # 如果不转，特征提取会全错！
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # 5. 提取特征 (进入 MobileFaceNet 模型，吐出 512 维向量)
            feature_vector = extractor.extract_feature(face_rgb)
            
            # 6. 存入数据库
            db.insert_user(person_name, feature_vector)

    print("\n=== 所有图库照片注册完毕，特征已永久保存！ ===")

if __name__ == "__main__":
    main()