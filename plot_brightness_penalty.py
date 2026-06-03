import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文和学术规范
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def plot_penalty_function():
    # 模拟从 0(纯黑) 到 255(纯白) 的所有可能的图像均值
    mean_brightness = np.linspace(0, 255, 500)
    
    # 严格代入你系统中的核心公式
    brightness_penalty = np.abs(mean_brightness - 128) / 128.0
    
    # 开始绘图
    plt.figure(figsize=(10, 6), dpi=300) # 300 dpi 保证论文打印高清
    
    # 绘制核心折线
    plt.plot(mean_brightness, brightness_penalty, color='#003366', linewidth=3, label='惩罚因子曲线')
    
    # 标出最优点 (128, 0)
    plt.scatter([128], [0], color='red', s=100, zorder=5)
    plt.annotate('理想曝光点 (均值=128, 惩罚=0)', 
                 xy=(128, 0), xytext=(135, 0.15),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, color='red')
    
    # 标出两个极端点
    plt.scatter([0, 255], [1, 1], color='orange', s=100, zorder=5)
    plt.text(5, 0.95, '纯黑 (过暗)', fontsize=12, color='orange')
    plt.text(220, 0.95, '纯白 (过曝)', fontsize=12, color='orange')
    
    # 填充安全区域 (假设惩罚因子 < 0.4 为可接受区域)
    safe_zone_mask = brightness_penalty < 0.4
    plt.fill_between(mean_brightness, 0, brightness_penalty, where=safe_zone_mask, 
                     color='#daffde', alpha=0.5, label='高质量成片安全区')

    # 图表装饰
    plt.title('图像均值亮度与光照惩罚因子映射关系', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('图像全局灰度均值 (Mean Brightness)', fontsize=14)
    plt.ylabel('光照惩罚因子 (Brightness Penalty)', fontsize=14)
    plt.xlim(0, 255)
    plt.ylim(-0.05, 1.1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper center')
    
    # 保存为高清图片
    save_path = "Fig5_Brightness_Penalty.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 学术图表已生成: {save_path}，可直接插入论文！")
    
if __name__ == "__main__":
    plot_penalty_function()