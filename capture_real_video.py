import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 严格遵循毕业论文排版要求 (宋体五号)
# ==========================================
plt.rcParams['font.size'] = 10.5
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

def assess_frame_quality(frame):
    """
    实时计算并返回：综合分数, 拉普拉斯得分, 光照惩罚
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 评估清晰度 (拉普拉斯方差)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    laplacian_score = min(laplacian_var * 5.0, 100.0)
    
    # 2. 评估光照度惩罚 (均值偏离 128 的程度)
    mean_brightness = np.mean(gray)
    brightness_penalty = abs(mean_brightness - 128) / 128.0 
    
    # 综合得分
    score = laplacian_score * (1.0 - 0.2 * brightness_penalty)
    return score, laplacian_score, brightness_penalty

def main():
    print("[INFO] 正在唤醒电脑摄像头...")
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头！")
        return

    frames_to_capture = 150
    scores = []
    lap_scores = []
    light_penalties = []
    
    print("\n[INFO] 🔴 开始录制！请按照剧本表演 (端坐 -> 疯狂摇头 -> 手捂镜头 -> 恢复端坐)")

    cv2.namedWindow("Desktop Real-time Quality Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Desktop Real-time Quality Capture", 640, 480) 

    for i in range(frames_to_capture):
        ret, frame = cap.read()
        if not ret: break
            
        score, lap, light = assess_frame_quality(frame)
        scores.append(score)
        lap_scores.append(lap)
        light_penalties.append(light)
        
        display_frame = cv2.resize(frame, (640, 480))
        
        color = (0, 0, 255) if score < 60.0 else (0, 255, 0)
        cv2.putText(display_frame, f"Frame: {i+1}/{frames_to_capture}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Quality Score: {score:.1f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Desktop Real-time Quality Capture", display_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 录制完成！正在生成带自动标记的最终学术图表...")

    # ==========================================
    # 模拟系统的时间序列降噪算法
    # ==========================================
    window_size = 5
    kernel = np.ones(window_size) / window_size
    smoothed_scores = np.convolve(scores, kernel, mode='same')
    smoothed_scores[:2] = scores[:2]
    smoothed_scores[-2:] = scores[-2:]

    # ==========================================
    # 绘制 图6-2：带自动识别区间标记的图表
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    ax.plot(scores, color='#1f77b4', alpha=0.3, linewidth=1.5, 
             label='原始视频帧物理评分')
    
    ax.plot(smoothed_scores, color='#d62728', linewidth=2.5, 
             label='一维卷积平滑后的质量特征曲线')
             
    ax.axhline(y=60.0, color='gray', linestyle='--', linewidth=1.5, 
                label='系统动态防抖阻断阈值 (T_blur = 60.0)')

    # 🚀 绝杀亮点：自动寻找并标记“异常区间”
    blur_marked = False
    light_marked = False
    
    for i in range(len(scores)):
        # 如果光照惩罚极大（比如大于 0.4，说明极暗或极曝），标记为【光照突变】
        if light_penalties[i] > 0.4:
            ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.15)
            if not light_marked:
                ax.text(i, max(scores) + 5, '检测到局部遮挡/光线突变', color='darkorange', fontsize=9, fontweight='bold')
                light_marked = True
        
        # 如果光线正常，但拉普拉斯得分极低（且最终得分低于60），标记为【运动模糊】
        elif lap_scores[i] < 60.0 and light_penalties[i] <= 0.4:
            ax.axvspan(i-0.5, i+0.5, color='purple', alpha=0.15)
            if not blur_marked:
                ax.text(i, 10, '检测到剧烈运动模糊', color='purple', fontsize=9, fontweight='bold')
                blur_marked = True

    ax.set_title('非受控桌面环境下视频流时序质量感知与过滤效果 (含系统自动归因)')
    ax.set_xlabel('时间序列 (视频帧数 / Frames)')
    ax.set_ylabel('物理综合质量分数 (Score)')
    
    max_y = max(max(scores) + 20, 110)
    ax.set_ylim([0, max_y])
    
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs("./eval_test", exist_ok=True)
    save_path = "./eval_test/Fig6-2_Real_Video_Quality.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] 带智能标记的图 6-2 已生成！快去文件夹里查看吧！路径: {save_path}")

if __name__ == "__main__":
    main()