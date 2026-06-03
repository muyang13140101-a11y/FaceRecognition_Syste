import csv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 强制接管全局渲染引擎 (绝对锁定五号字体 10.5pt)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False 

plt.rcParams['font.size'] = 10.5 
plt.rcParams['axes.titlesize'] = 10.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5
plt.rcParams['legend.fontsize'] = 10.5

def load_csv_data(file_path):
    frames, times, scores, cnn_flags, identities, lap_vars, bright_pens = [], [], [], [], [], [], []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            frames.append(int(row[0]))
            times.append(float(row[1]))
            scores.append(float(row[2]))
            cnn_flags.append(row[3] == 'True')
            identities.append(row[4])
            lap_vars.append(float(row[5]))
            bright_pens.append(float(row[6]))
    return np.array(frames), np.array(times), np.array(scores), np.array(cnn_flags), identities, np.array(lap_vars), np.array(bright_pens)

# 加载数据 
f_wrong, t_wrong, s_wrong, c_wrong, id_wrong, lap_wrong, bright_wrong = load_csv_data("result_without_filter.csv")
f_right, t_right, s_right, c_right, id_right, _, _ = load_csv_data("result_with_filter.csv")

# ==========================================
# 🌟 智能侦测逻辑：定义物理噪声门限
# ==========================================
# 设定 1: 均值偏离中心点超过 35%，判定为光照异常 (遮挡/强光)，画黄色
illumination_noise = bright_wrong > 0.35 
# 设定 2: 拉普拉斯方差低于 60，判定为运动模糊 (左右晃动)，画灰色
motion_noise = lap_wrong < 60.0 

# ==========================================
# 绘制图表 1：单帧处理耗时对比折线图 
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 5))

# 自动渲染彩色背景区间
ax1.fill_between(f_wrong, 0, 150, where=illumination_noise, color='#FFD700', alpha=0.3, label='光照畸变干扰区')
ax1.fill_between(f_wrong, 0, 150, where=motion_noise, color='#808080', alpha=0.3, label='左右晃动干扰区')

ax1.plot(f_wrong, t_wrong, color='red', label='传统模式 (无前置过滤)', linewidth=1.5)
ax1.plot(f_right, t_right, color='blue', label='创新模式 (开启时序滑窗)', linewidth=1.5, linestyle='--')

ax1.set_title('视频流单帧处理耗时时序对比消融实验 (Ablation Study)')
ax1.set_xlabel('视频流离散帧序号 (Frame Index)')
ax1.set_ylabel('单帧整体处理耗时开销 (Time Cost / ms)')
ax1.set_ylim(0, max(np.max(t_wrong) + 20, 120)) 
ax1.grid(True, linestyle=':')
ax1.legend(loc='upper right')
plt.tight_layout()
plt.savefig("ablation_time_comparison.png", dpi=300)
plt.close()

# ==========================================
# 绘制图表 2：系统识别状态震荡对比散点图 
# ==========================================
def convert_identity_to_status(identities, cnn_flags):
    status = []
    for identity, cnn in zip(identities, cnn_flags):
        if identity == "Blocked":
            status.append(-1)
        elif identity != "Unknown" and identity != "None":
            status.append(1)
        else:
            status.append(0)
    return status

status_wrong = convert_identity_to_status(id_wrong, c_wrong)
status_right = convert_identity_to_status(id_right, c_right)

fig, ax2 = plt.subplots(figsize=(10, 4))

# 自动渲染彩色背景区间
ax2.fill_between(f_wrong, -1.5, 1.5, where=illumination_noise, color='#FFD700', alpha=0.3, label='光照畸变干扰区')
ax2.fill_between(f_wrong, -1.5, 1.5, where=motion_noise, color='#808080', alpha=0.3, label='左右晃动干扰区')

ax2.scatter(f_wrong, status_wrong, color='red', alpha=0.6, label='传统模式 (频繁闪烁)', s=15)
ax2.scatter(f_right, status_right, color='blue', alpha=0.6, label='创新模式 (平滑受控)', s=15, marker='x')

ax2.set_title('视频流识别输出状态平滑度对比实验')
ax2.set_xlabel('视频流离散帧序号 (Frame Index)')
ax2.set_ylabel('系统工作状态决策编码')

ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['主动丢帧拦截 (-1)', '状态跳变/Unknown (0)', '有效特征识别 (1)'])
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, linestyle=':')
ax2.legend(loc='lower right')
plt.tight_layout()
plt.savefig("ablation_stability_comparison.png", dpi=300)
plt.close()

print("图表渲染完毕！全图已严格锁定为: 中文宋体, 英文Times New Roman, 绝对字号 10.5pt (五号)")
print("已成功实现物理信道干扰类型的自动化分色标记。")