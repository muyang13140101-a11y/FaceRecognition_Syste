import cv2
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os

# ==========================================
# 严格遵循毕业论文排版要求 (宋体五号)
# ==========================================
plt.rcParams['font.size'] = 10.5
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("[INFO] 正在加载 MTCNN 级联感知网络...")
    mtcnn = MTCNN(keep_all=False, device='cpu')

    img_path = "./tilted_face.jpg" 
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"[ERROR] 找不到图片: {img_path}")
        print("请用手机拍一张歪着头的照片，命名为 tilted_face.jpg 放在项目目录下！")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("[INFO] 正在进行像素级感知与 5 点拓扑定位...")
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

    if boxes is None or landmarks is None:
        print("[ERROR] 未检测到人脸，请换一张脸部清晰的图片。")
        return

    box = boxes[0]
    points = landmarks[0] # 5个真实关键点

    # ==========================================
    # 绘制左半边图：原始图像 + 5个关键点标定
    # ==========================================
    img_raw_draw = img_rgb.copy()
    
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img_raw_draw, (x, y), 5, (0, 255, 0), -1) 
        cv2.circle(img_raw_draw, (x, y), 6, (255, 255, 255), 1)

    x1, y1, x2, y2 = [int(b) for b in box]
    cv2.rectangle(img_raw_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # ==========================================
    # 核心算法替换：使用深度学习界标准的 5点相似变换 (Similarity Transform)
    # ==========================================
    print("[INFO] 正在执行 5点相似变换拓扑对齐...")
    
    # 定义 ArcFace 官方 112x112 模板的 5 个黄金基准点
    reference_pts = np.array([
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041]   # 右嘴角
    ], dtype=np.float32)

    src_pts = np.array(points, dtype=np.float32)

    # 绝杀：直接调用底层算法求解包含平移、旋转、缩放的 4自由度仿射矩阵 M
    M, _ = cv2.estimateAffinePartial2D(src_pts, reference_pts)

    # 终极对齐：一步到位切出 112x112 的完美标准化正向脸！
    face_aligned_112 = cv2.warpAffine(img_rgb, M, (112, 112), flags=cv2.INTER_CUBIC)

    # 为了画图好看，把左边的原图局部抠出来
    margin = 30
    h, w = img_rgb.shape[:2]
    crop_x1 = max(0, x1 - margin)
    crop_y1 = max(0, y1 - margin)
    crop_x2 = min(w, x2 + margin)
    crop_y2 = min(h, y2 + margin)
    face_raw_cropped = img_raw_draw[crop_y1:crop_y2, crop_x1:crop_x2]

    # ==========================================
    # 拼图并保存：生成最终的学术图表 3-2
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    
    axes[0].imshow(face_raw_cropped)
    axes[0].set_title("原始面部空间几何畸变\n(MTCNN 提取 5 点真实坐标)", fontsize=11, pad=10)
    axes[0].axis('off')
    
    axes[1].imshow(face_aligned_112)
    axes[1].set_title("五点相似变换矩阵纠偏\n(直接输出 112x112 规范化表征)", fontsize=11, pad=10)
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs("./eval_test", exist_ok=True)
    save_path = "./eval_test/Fig3-2_Face_Alignment.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] 完美的图 3-2 已生成！底层逻辑已与核心检测器 100% 同步！路径: {save_path}")

if __name__ == "__main__":
    main()