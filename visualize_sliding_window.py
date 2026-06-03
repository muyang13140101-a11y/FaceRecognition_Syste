import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def simulate_sliding_window():
    np.random.seed(42)
    frames = 50
    
    # 1. 模拟真实的单帧得分
    raw_scores = np.random.uniform(70, 85, frames)
    
    # 2. 强行注入“突发噪声” (模拟摄像头失焦、用户突然转头或眨眼)
    raw_scores[10] = 30.0  # 瞬间模糊
    raw_scores[11] = 45.0  # 仍在恢复
    raw_scores[30] = 25.0  # 逆光闪烁
    raw_scores[42] = 35.0  # 运动模糊
    
    # 3. 实现滑动窗口滤波 (完全对应你 quality.py 中的算法逻辑)
    window_size = 5
    smoothed_scores = []
    
    # 模拟队列填充过程
    history_queue = []
    for score in raw_scores:
        history_queue.append(score)
        if len(history_queue) > window_size:
            history_queue.pop(0) # 剔除最老的数据，保持窗口长度
        
        # 计算当前窗口的平均值
        smoothed_scores.append(sum(history_queue) / len(history_queue))
        
    # 4. 可视化绘图
    plt.figure(figsize=(12, 6), dpi=300)
    
    # 画原始分数（充满毛刺）
    plt.plot(raw_scores, color='gray', linestyle='--', alpha=0.7, marker='x', 
             label='单帧原始质量得分 (Raw Score)')
    
    # 画滤波后的分数（平滑顺滑）
    plt.plot(smoothed_scores, color='#003366', linewidth=3, marker='o', 
             label=f'滑动窗口滤波后得分 (Window={window_size})')
    
    # 画及格线 (对应你 quality.py 中的 blur_threshold=60.0)
    threshold = 60.0
    plt.axhline(y=threshold, color='red', linestyle='-.', linewidth=2, 
                label=f'质量拦截阈值 (Threshold={threshold})')
    
    # 突出显示被挽救的“假阴性”点
    plt.annotate('突发噪点被成功抹平\n系统未产生误判', 
                 xy=(10, 30), xytext=(12, 40),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=12, color='green')
    
    plt.title('动态视频流质量评估：滑动窗口时序滤波效果对比', fontsize=16, pad=15)
    plt.xlabel('视频时间轴 (Frame Index)', fontsize=14)
    plt.ylabel('图像质量综合评分', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 100)
    
    # 保存图片
    plt.savefig('Fig5_Sliding_Window.png', bbox_inches='tight')
    print("✅ 滤波效果对比图已生成: Fig5_Sliding_Window.png")

if __name__ == "__main__":
    simulate_sliding_window()