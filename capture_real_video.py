import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 严格遵循毕业论文排版要求 (宋体五号)
# ==========================================
plt.rcParams['font.size'] = 10.5
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

def assess_frame_quality(frame):
    """
    实时计算帧质量综合分数 (0.0 ~ 1.0)
    结合拉普拉斯方差(反运动模糊)与灰度均值(光照度)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 评估清晰度 (数值越小越模糊)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    norm_blur = np.clip(blur_score / 400.0, 0, 1) 
    
    # 2. 评估光照度 (越接近128越完美)
    brightness = np.mean(gray)
    light_score = 1.0 - (abs(brightness - 128) / 128.0) 
    
    # 综合权重：清晰度占70%，光照占30%
    return norm_blur * 0.7 + light_score * 0.3

def main():
    print("[INFO] 正在唤醒电脑摄像头...")
    # 如果 Windows 系统仍然报错，请将其修改为 cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头！请检查电脑摄像头权限。")
        return

    frames_to_capture = 150
    scores = []
    
    print("\n===========================================")
    print(f"[INFO] 摄像头已就绪！即将开始采集 {frames_to_capture} 帧真实数据。")
    print("🎬 【剧本提示】：")
    print("   第1阶段 (正常)：端坐，看着屏幕。")
    print("   第2阶段 (模糊)：故意快速左右摇头 (制造运动模糊)。")
    print("   第3阶段 (遮挡)：用手捂住一半摄像头 (制造光线突变)。")
    print("   第4阶段 (恢复)：把手拿开，恢复端坐。")
    print("===========================================\n")
    
    for i in range(3, 0, -1):
        print(f"倒计时 {i} 秒后开始...")
        time.sleep(1)

    print("[INFO] 🔴 开始录制！请开始你的表演！")

    # ==========================================
    # 核心修正：在唯一的一个录制循环前，初始化 3:4 比例窗口
    # ==========================================
    cv2.namedWindow("IoT Real-time Quality Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("IoT Real-time Quality Capture", 480, 640) # 强制给 Windows 一个初试尺寸

    for i in range(frames_to_capture):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 计算每一帧的真实质量分数
        score = assess_frame_quality(frame)
        scores.append(score)
        
        # 核心修正：将画面强制裁剪或调整为 3:4 的 480x640 竖屏小画面
        display_frame = cv2.resize(frame, (480, 640))
        
        # 在变形后的标准 3:4 画面上绘制学术文本
        color = (0, 0, 255) if score < 0.5 else (0, 255, 0)
        cv2.putText(display_frame, f"Frame: {i+1}/{frames_to_capture}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Quality Score: {score:.2f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 展现干净紧凑的小窗口
        cv2.imshow("IoT Real-time Quality Capture", display_frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # 录制结束后，一次性释放并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 录制完成！正在生成最终学术图表...")

    # ==========================================
    # 模拟系统的时间序列降噪算法 (滑动平均滤波)
    # ==========================================
    smoothed_scores = np.convolve(scores, np.ones(5)/5, mode='same')
    smoothed_scores[:2] = scores[:2]
    smoothed_scores[-2:] = scores[-2:]

    # ==========================================
    # 绘制 图6-2：视频帧质量评估与降噪效果折线图
    # ==========================================
    plt.figure(figsize=(9, 5))
    
    plt.plot(scores, color='#1f77b4', alpha=0.4, linewidth=1.5, 
             label='原始视频帧质量评分 (含运动模糊与光线突变)')
    
    plt.plot(smoothed_scores, color='#d62728', linewidth=2.5, 
             label='算法动态降噪后的平滑质量分数')
             
    plt.axhline(y=0.45, color='gray', linestyle='--', linewidth=1.5, 
                label='系统判定阈值 (低于此分数的帧将被剔除)')

    plt.title('真实办公环境下视频流帧质量评估与动态降噪效果')
    plt.xlabel('视频帧序列 (Frames)')
    plt.ylabel('综合质量分数')
    plt.ylim([0, 1.1])
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    save_path = "./eval_test/Fig6-2_Real_Video_Quality.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] 极其真实的图 6-2 已成功生成！路径: {save_path}")

if __name__ == "__main__":
    main()