import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

CSV_FILE = "./eval_test/lfw_real_results.csv"

y_true = []
y_scores = []

print(f"[INFO] 正在读取真实评估数据: {CSV_FILE}...")
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        if len(row) == 2:
            y_true.append(int(row[0]))
            y_scores.append(float(row[1]))

y_true = np.array(y_true)
y_scores = np.array(y_scores)

# 全局绘图参数设置（解决中文乱码与数学符号冲突）
plt.rcParams['font.size'] = 10.5
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['mathtext.fontset'] = 'cm'

# ==========================================
# 2. 绘制图 6-1：正负样本相似度真实分布直方图
# ==========================================
genuine_scores = y_scores[y_true == 1]
imposter_scores = y_scores[y_true == 0]

plt.figure(figsize=(8, 5))
plt.hist(imposter_scores, bins=50, alpha=0.6, color='#1f77b4', label='不同人 (Imposter Pairs)', density=True)
plt.hist(genuine_scores, bins=50, alpha=0.6, color='#ff7f0e', label='同一个人 (Genuine Pairs)', density=True)

plt.axvline(x=0.6, color='red', linestyle='--', linewidth=1.5, label='系统阈值 (0.6)')

plt.title('LFW 数据集上的真实相似度分布')
plt.xlabel('余弦相似度 (Cosine Similarity)')
plt.ylabel('概率密度')
plt.legend(loc='upper center')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./eval_test/Fig6-1_Real_Distribution.png', dpi=300)
plt.close()

# ==========================================
# 3. 绘制图 6-3：真实的 ROC 曲线
# ==========================================
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#d62728', linewidth=2, label=f'模型性能 (AUC = {roc_auc:.4f})')

# X轴对数坐标及小数格式化
plt.xscale('log')
plt.xlim([0.0001, 1.0])
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x)))

plt.ylim([0.6, 1.0])
plt.title('LFW 数据集上的 ROC 曲线')
plt.xlabel('错误接受率 (FAR)')
plt.ylabel('真正率 (TPR)')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('./eval_test/Fig6-3_Real_ROC.png', dpi=300)
plt.close()

# ==========================================
# 4. 绘制图 6-4：FAR 与 FRR 交叉折线图 (带对数优化)
# ==========================================
frr = 1 - tpr

plt.figure(figsize=(8, 5))
plt.plot(thresholds, fpr, color='red', linewidth=2, label='错误接受率 (FAR)')
plt.plot(thresholds, frr, color='blue', linewidth=2, label='错误拒绝率 (FRR)')

eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
eer_threshold = interp1d(fpr, thresholds)(eer)

plt.plot(eer_threshold, eer, 'ko', markersize=6)
plt.annotate(f'EER 最佳阈值 $\\approx$ {eer_threshold:.3f}', 
             xy=(eer_threshold, eer), xytext=(eer_threshold-0.15, eer+0.2), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

# Y轴对数坐标及小数格式化
plt.yscale('log')
plt.ylim([0.0001, 1.0]) 
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:g}'.format(y)))

plt.title('FAR 与 FRR 随判定阈值变化曲线')
plt.xlabel('余弦相似度阈值')
plt.ylabel('错误率 (Log Scale)')
plt.xlim([-0.2, 1.0])
plt.legend(loc='upper center')
plt.grid(True, which="both", linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('./eval_test/Fig6-4_Real_FAR_FRR.png', dpi=300)
plt.close()

print(f"[SUCCESS] 所有的真实学术图表已全部生成！(坐标轴已强制转换为直观小数)")