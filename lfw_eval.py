import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==========================================
# 1. 导入你毕设中现有的提取器 (请确保路径正确)
# ==========================================
# 假设你的提取器有一个 extract 方法，传入 OpenCV 图像，返回 1D numpy 数组
from core.extractor import FaceExtractor 
extractor = FaceExtractor()

# ==========================================
# 2. 配置 LFW 数据集路径
# ==========================================
LFW_DIR = "./lfw_aligned_112"  # 替换为你的 LFW 图片所在文件夹路径
PAIRS_PATH = "./pairs.txt"     # 替换为你的 pairs.txt 路径

def get_image_path(lfw_dir, name, num):
    """根据名字和序号拼接图片路径 (具体根据你的LFW文件夹结构调整)"""
    filename = f"{name}_{str(num).zfill(4)}.jpg"
    return os.path.join(lfw_dir, name, filename)

def extract_feature(img_path):
    """读取图片并提取特征"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    # 调用你自己的系统代码提取特征
    feature = extractor.extract(img) 
    return feature

# ==========================================
# 3. 解析 pairs.txt 并计算真实相似度
# ==========================================
print("[INFO] 开始读取 pairs.txt 并提取特征...")
y_true = [] # 真实标签：1表示同一个人，0表示不同人
y_scores = [] # 真实的余弦相似度

with open(PAIRS_PATH, 'r') as f:
    lines = f.readlines()[1:] # 跳过第一行表头
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        
        if len(parts) == 3:
            # 正样本：同一个人
            name, img1, img2 = parts
            path1 = get_image_path(LFW_DIR, name, img1)
            path2 = get_image_path(LFW_DIR, name, img2)
            label = 1
        elif len(parts) == 4:
            # 负样本：不同人
            name1, img1, name2, img2 = parts
            path1 = get_image_path(LFW_DIR, name1, img1)
            path2 = get_image_path(LFW_DIR, name2, img2)
            label = 0
            
        feat1 = extract_feature(path1)
        feat2 = extract_feature(path2)
        
        if feat1 is not None and feat2 is not None:
            # 计算余弦相似度
            sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
            y_scores.append(sim)
            y_true.append(label)
            
        if (i+1) % 500 == 0:
            print(f"[INFO] 已处理 {i+1} 对...")

y_true = np.array(y_true)
y_scores = np.array(y_scores)

print("[INFO] 提取完成！开始生成学术图表...")

# ==========================================
# 4. 基于真实数据生成：图 6-1 (正负样本相似度分布)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

genuine_scores = y_scores[y_true == 1]
imposter_scores = y_scores[y_true == 0]

plt.figure(figsize=(8, 5))
plt.hist(imposter_scores, bins=50, alpha=0.6, color='#1f77b4', label='Imposter Pairs (Different Person)', density=True)
plt.hist(genuine_scores, bins=50, alpha=0.6, color='#ff7f0e', label='Genuine Pairs (Same Person)', density=True)
plt.title('Real Similarity Distribution on LFW', fontsize=14)
plt.xlabel('Cosine Similarity', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('Real_Fig6-1_Distribution.png', dpi=300)
plt.close()

# ==========================================
# 5. 基于真实数据生成：图 6-3 (ROC 曲线)
# ==========================================
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#d62728', linewidth=2.5, label=f'Model Performance (AUC={roc_auc:.4f})')
plt.xscale('log')
plt.xlim([10**-4, 10**0])
plt.ylim([0.6, 1.0])
plt.title('ROC Curve on LFW Dataset', fontsize=14)
plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Real_Fig6-3_ROC.png', dpi=300)
plt.close()

# ==========================================
# 6. 基于真实数据生成：图 6-4 (FAR/FRR 交叉求最优阈值)
# ==========================================
frr = 1 - tpr

plt.figure(figsize=(8, 5))
plt.plot(thresholds, fpr, color='red', linewidth=2, label='False Acceptance Rate (FAR)')
plt.plot(thresholds, frr, color='blue', linewidth=2, label='False Rejection Rate (FRR)')

# 寻找等错误率 EER 的最佳真实阈值
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
eer_threshold = interp1d(fpr, thresholds)(eer)

plt.plot(eer_threshold, eer, 'ko', markersize=8)
plt.annotate(f'Best Threshold $\\approx$ {eer_threshold:.2f}', 
             xy=(eer_threshold, eer), xytext=(eer_threshold-0.1, eer+0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

plt.title('FAR and FRR vs. Decision Threshold', fontsize=14)
plt.xlabel('Cosine Similarity Threshold', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)
plt.xlim([-0.2, 1.0])
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('Real_Fig6-4_FAR_FRR.png', dpi=300)
plt.close()

print(f"[SUCCESS] 所有真实图表已生成！系统的理论最佳阈值为: {eer_threshold:.3f}")