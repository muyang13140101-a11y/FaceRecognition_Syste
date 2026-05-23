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
    实时计算帧质量综合分数
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # ==========================================
    # 核心修复 1：传感器增益放大 (将分数映射到 0-100)
    # 假设原相机峰值为 20，乘以 5.0 后峰值可达 100
    # ==========================================
    laplacian_score = np.clip(laplacian_var * 5.0, 0, 100)
    
    mean_brightness = np.mean(gray)
    brightness_penalty = abs(mean_brightness - 128) / 128.0 
    
    score = laplacian_score * (1.0 - 0.2 * brightness_penalty)
    return score

def main():
    print("[INFO] 正在唤醒电脑摄像头...")
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        return

    frames_to_capture = 150
    scores = []
    
    print("\n[INFO] 🔴 开始录制！请开始你的表演！")

    cv2.namedWindow("Desktop Real-time Quality Capture", cv2.WINDOW_NORMAL)
    # 核心修复 2：恢复正常的横屏比例 (宽度 640，高度 480)
    cv2.resizeWindow("Desktop Real-time Quality Capture", 640, 480) 

    for i in range(frames_to_capture):
        ret, frame = cap.read()
        if not ret: break
            
        score = assess_frame_quality(frame)
        scores.append(score)
        
        # 核心修复 2：正确缩放为横屏
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

    # 一维滑动窗口
    window_size = 5
    kernel = np.ones(window_size) / window_size
    smoothed_scores = np.convolve(scores, kernel, mode='same')
    smoothed_scores[:2] = scores[:2]
    smoothed_scores[-2:] = scores[-2:]

    # 绘图
    plt.figure(figsize=(9, 5))
    plt.plot(scores, color='#1f77b4', alpha=0.4, linewidth=1.5, label='原始视频帧物理评分')
    plt.plot(smoothed_scores, color='#d62728', linewidth=2.5, label='一维卷积平滑后的质量特征曲线')
    plt.axhline(y=60.0, color='gray', linestyle='--', linewidth=1.5, label='系统动态防抖阻断阈值 (T_blur = 60.0)')

    plt.title('非受控桌面环境下视频流时序质量感知与过滤效果')
    plt.xlabel('时间序列 (视频帧数 / Frames)')
    plt.ylabel('物理综合质量分数 (Score)')
    plt.ylim([0, 110]) # 强制 Y 轴最高显示到 110，完美契合百分制
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs("./eval_test", exist_ok=True)
    plt.savefig("./eval_test/Fig6-2_Real_Video_Quality.png", dpi=300)
    plt.close()
    print("[SUCCESS] 图表已生成！")

if __name__ == "__main__":
    main()