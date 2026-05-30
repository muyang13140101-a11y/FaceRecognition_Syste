import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D  # 修复 NameError 的关键导入

# 确保输出目录存在
os.makedirs('./eval_test', exist_ok=True)

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
# 1. 核心数学指标动态计算 (含 d-prime 指数)
# ==========================================
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
frr = 1 - tpr

# 计算等错误率 (EER)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
eer_threshold = interp1d(fpr, thresholds)(eer)

# 计算高安全阈值 (约束 FAR <= 0.001)
target_far = 0.001
idx_strict = np.argmin(np.abs(fpr - target_far))
strict_threshold = thresholds[idx_strict]
strict_tpr = tpr[idx_strict]

# 计算 d-prime (类间可分性指数)
genuine_scores = y_scores[y_true == 1]
imposter_scores = y_scores[y_true == 0]
mu_gen, var_gen = np.mean(genuine_scores), np.var(genuine_scores)
mu_imp, var_imp = np.mean(imposter_scores), np.var(imposter_scores)
d_prime = np.sqrt(2) * abs(mu_gen - mu_imp) / np.sqrt(var_gen + var_imp)

# 打印论文可以直接抄的详细数据卡片
print("\n" + "="*50)
print("🎓 毕业论文 第6章 核心量化指标 (可以直接写进论文)")
print("="*50)
print(f"[全局判别力] ROC曲线下面积 (AUC) : {roc_auc:.4f}")
print(f"[分布分离度] d-prime (d') 指数    : {d_prime:.4f} (值越大越好，表明正负样本极度分离)")
print(f"[系统平衡点] 等错误率 (EER)       : {eer*100:.2f}% (对应理论最佳阈值: {eer_threshold:.3f})")
print(f"[工程落地站] 高安全阈值 (FAR=0.1%): {strict_threshold:.3f} (此时召回率 TPR: {strict_tpr*100:.2f}%)")
print("="*50 + "\n")


# ==========================================
# 2. 绘制图 6-1：正负样本相似度真实分布直方图
# ==========================================
plt.figure(figsize=(8, 5))
plt.hist(imposter_scores, bins=50, alpha=0.6, color='#1f77b4', density=True)
plt.hist(genuine_scores, bins=50, alpha=0.6, color='#ff7f0e', density=True)

# 仅画线，不带图例标签
plt.axvline(x=eer_threshold, color='green', linestyle=':', linewidth=2)
plt.axvline(x=strict_threshold, color='red', linestyle='--', linewidth=2)

# 自定义图例 (确保顺序且不挡文字)
custom_lines = [
    Line2D([0], [0], color='#1f77b4', lw=4, alpha=0.6),
    Line2D([0], [0], color='#ff7f0e', lw=4, alpha=0.6),
    Line2D([0], [0], color='green', lw=2, linestyle=':'),
    Line2D([0], [0], color='red', lw=2, linestyle='--')
]

ymin, ymax = plt.ylim()
text_y_pos = ymax * 0.85

# EER (绿线) 数字标在左侧空白处，用箭头指向绿线
plt.text(eer_threshold - 0.05, text_y_pos, f'{eer_threshold:.3f}', color='green', 
         fontsize=11, va='center', ha='right', fontweight='bold')
plt.annotate('', xy=(eer_threshold, text_y_pos), xytext=(eer_threshold - 0.045, text_y_pos),
            arrowprops=dict(arrowstyle="->", color='green', lw=1.5))
         
# 高安全阈值 (红线) 数字标在右侧空白处，用箭头指向红线
plt.text(strict_threshold + 0.05, text_y_pos, f'{strict_threshold:.3f}', color='red', 
         fontsize=11, va='center', ha='left', fontweight='bold')
plt.annotate('', xy=(strict_threshold, text_y_pos), xytext=(strict_threshold + 0.045, text_y_pos),
            arrowprops=dict(arrowstyle="->", color='red', lw=1.5))

plt.title('LFW 数据集上的真实相似度分布')
plt.xlabel('余弦相似度 (Cosine Similarity)')
plt.ylabel('概率密度')

# 核心修改：将图例位置 loc 改为 'upper right'，避免遮挡中部的分布线
plt.legend(custom_lines, ['不同人 (Imposter Pairs)', '同一个人 (Genuine Pairs)', 'EER 阈值', '高安全阈值'], loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./eval_test/Fig6-1_Real_Distribution.png', dpi=300)
plt.close()


# ==========================================
# 3. 绘制图 6-3：真实的 ROC 曲线 (带精确定位)
# ==========================================
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#d62728', linewidth=2, label=f'系统整体性能 (AUC = {roc_auc:.4f})')

plt.plot(eer, 1-eer, 'go', markersize=8, label='EER 平衡工作点')
plt.plot(fpr[idx_strict], tpr[idx_strict], 'ro', markersize=8, label='高安全工作点 (FAR=0.1%)')

plt.annotate(f'FAR: {eer:.4f}\nTPR: {1-eer:.4f}', 
             xy=(eer, 1-eer), xytext=(-35, -45), textcoords='offset points',
             color='green', fontweight='bold', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.2", color='green', lw=1.5))

plt.annotate(f'FAR: {fpr[idx_strict]:.4f}\nTPR: {tpr[idx_strict]:.4f}', 
             xy=(fpr[idx_strict], tpr[idx_strict]), xytext=(30, -35), textcoords='offset points',
             color='red', fontweight='bold', ha='left',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.2", color='red', lw=1.5))

plt.xscale('log')
plt.xlim([0.0001, 1.0])
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:g}'.format(x)))

plt.ylim([0.5, 1.0])
plt.title('LFW 数据集上的 ROC 曲线与系统工作点选择')
plt.xlabel('错误接受率 (FAR, Log Scale)')
plt.ylabel('真正例率 (TPR)')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('./eval_test/Fig6-3_Real_ROC.png', dpi=300)
plt.close()


# ==========================================
# 4. 绘制图 6-4：FAR 与 FRR 交叉折线图 (解决挡住对数坐标轴问题)
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(thresholds, fpr, color='red', linewidth=2, label='错误接受率 (FAR)')
plt.plot(thresholds, frr, color='blue', linewidth=2, label='错误拒绝率 (FRR)')

plt.plot(eer_threshold, eer, 'ko', markersize=6)

# 文字向右上角空白区域排版
plt.annotate(f'EER 最佳平衡阈值 $\\approx$ {eer_threshold:.3f}', 
             xy=(eer_threshold, eer), xytext=(eer_threshold+0.1, eer*10), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10.5, fontweight='bold', ha='left')

plt.yscale('log')
plt.ylim([0.0001, 1.0]) 
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:g}'.format(y)))

plt.title('FAR 与 FRR 随判定阈值变化曲线')
plt.xlabel('余弦相似度阈值')
plt.ylabel('错误率 (Log Scale)')
plt.xlim([-0.2, 1.0])

# 将图例挪到右上角，避开曲线交叉的核心区域
plt.legend(loc='upper right') 

plt.grid(True, which="both", linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('./eval_test/Fig6-4_Real_FAR_FRR.png', dpi=300)
plt.close()

print("[SUCCESS] 图 6-1 和 6-4 的防遮挡版学术排版已完成！")