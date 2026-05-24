import os
import cv2
import numpy as np
import time

# 导入你自己的特征提取器！
from core.extractor import FaceExtractor 
from core.detector import FaceDetector

# 配置路径
LFW_DIR = "./eval_test/lfw/lfw-deepfunneled/lfw-deepfunneled"
PAIRS_FILE = "./eval_test/pairs.txt"
OUTPUT_FILE = "./eval_test/lfw_real_results.csv"

def compute_cosine_similarity(feat1, feat2):
    """
    计算余弦相似度 (极致优化版)
    由于 extractor 已对特征进行 L2 归一化，此处直接使用点积等价于余弦相似度
    """
    return float(np.dot(feat1, feat2))

def get_image_path(lfw_dir, name, num):
    """格式化 LFW 图片路径，并强制转换为绝对正斜杠路径解决 Windows Bug"""
    filename = f"{name}_{int(num):04d}.jpg"
    raw_path = os.path.join(lfw_dir, name, filename)
    abs_path = os.path.abspath(raw_path).replace('\\', '/')
    return abs_path

def run_evaluation():
    print("[INFO] 正在初始化你系统的 AI 模型 (只读模式)...")
    detector = FaceDetector()
    extractor = FaceExtractor()
    
    results = []
    print(f"[INFO] 开始读取 LFW 比汤清单: {PAIRS_FILE}")
    
    with open(PAIRS_FILE, 'r') as f:
        lines = f.readlines()[1:] 
        
    total_pairs = len(lines)
    print(f"[INFO] 共发现 {total_pairs} 对测试样本。开始提取特征...")
    
    start_time = time.time()
    
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        
        if len(parts) == 3:
            name, img1_idx, img2_idx = parts
            path1 = get_image_path(LFW_DIR, name, img1_idx)
            path2 = get_image_path(LFW_DIR, name, img2_idx)
            is_same = 1
        elif len(parts) == 4:
            name1, img1_idx, name2, img2_idx = parts
            path1 = get_image_path(LFW_DIR, name1, img1_idx)
            path2 = get_image_path(LFW_DIR, name2, img2_idx)
            is_same = 0
        else:
            continue
            
        try:
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            if img1 is None or img2 is None:
                continue
                
            face1_list = detector.detect_and_crop(img1)
            face2_list = detector.detect_and_crop(img2)
            
            if not face1_list or not face2_list:
                results.append(f"{is_same},0.000")
                continue
                
            face1_raw = face1_list[0]
            face2_raw = face2_list[0]
            
            face1 = face1_raw[0] if isinstance(face1_raw, tuple) else face1_raw
            face2 = face2_raw[0] if isinstance(face2_raw, tuple) else face2_raw
            
            if face1 is None or getattr(face1, 'size', 0) == 0 or face2 is None or getattr(face2, 'size', 0) == 0:
                results.append(f"{is_same},0.000")
                continue
                
            feat1 = extractor.extract_feature(face1)
            feat2 = extractor.extract_feature(face2)
            
            sim = compute_cosine_similarity(feat1, feat2)
            results.append(f"{is_same},{sim:.4f}")
            
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            
        if (i + 1) % 100 == 0:
            print(f"进度: {i + 1}/{total_pairs} 已完成...")
            
    with open(OUTPUT_FILE, 'w') as f:
        f.write("is_same,similarity\n")
        f.write("\n".join(results))
        
    print(f"\n[SUCCESS] 测试完成！耗时: {time.time()-start_time:.2f} 秒")
    print(f"[SUCCESS] 真实评估数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()