import cv2
import time
import csv
import numpy as np
from core.quality import QualityAssessor
from core.detector import FaceDetector 
from core.extractor import FaceExtractor
from core.matcher import FaceMatcher

def run_ablation_experiment(video_path, enable_filter, output_csv_name):
    print(f"\n==============================================")
    print(f"正在运行测试，前置过滤状态：{enable_filter}...")
    print(f"==============================================")
    
    # 彻底交由底层接管：初始化时告诉它需要 30 帧来校准环境
    quality_assessor = QualityAssessor(window_size=5, calibration_frames=30)
    detector = FaceDetector()
    extractor = FaceExtractor()
    matcher = FaceMatcher()
    
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    
    with open(output_csv_name, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame_ID', 'Time_Cost_ms', 'Smoothed_Score', 'CNN_Activated', 'Identity_Result', 'Lap_Var', 'Bright_Penalty'])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_id += 1
            if frame_id % 30 == 0: 
                print(f"正在疯狂处理第 {frame_id} 帧...")
            
            # 计时开始
            start_time = time.time()
            
            # 1. 光照畸变惩罚计算 (非常轻量，保留用于遥测制图)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            bright_penalty = abs(np.mean(gray) - 128) / 128.0
            
            # 2. 🌟 核心引擎调用：所有复杂的时序平滑、环境标定、动态阈值生成，全在这一句代码内自动完成！
            is_valid, smoothed_score = quality_assessor.assess_frame(frame)
            
            # (为了绘图需要，提取当前平滑分作为 lap_var 记录，避免双重计算卷积)
            lap_var_proxy = smoothed_score 

            cnn_activated = False
            identity = "Unknown"
            
            # 3. 前置拦截门控 (直接听从底层评估器的判决)
            if enable_filter and not is_valid:
                time_cost_ms = (time.time() - start_time) * 1000
                # 记录阻断状态，并直接进入下一帧 (真正做到了 0 毫秒级卸载)
                writer.writerow([frame_id, time_cost_ms, smoothed_score, False, "Blocked", lap_var_proxy, bright_penalty])
                continue
            
            # 4. 重度网络管线
            cnn_activated = True
            
            faces = detector.detect_and_crop(frame)
            if len(faces) > 0:
                face_crop = faces[0][0] 
                features = extractor.extract_feature(face_crop)
                match_name, match_score = matcher.match(features)
                identity = match_name
            
            time_cost_ms = (time.time() - start_time) * 1000
            writer.writerow([frame_id, time_cost_ms, smoothed_score, cnn_activated, identity, lap_var_proxy, bright_penalty])

    cap.release()
    print(f"\n>>> 自动化数据采集完成！已保存至: {output_csv_name} <<<")

if __name__ == "__main__":
    # 请确保视频在同一目录下
    test_video_file = "test_noise_video.mp4" 
    run_ablation_experiment(test_video_file, enable_filter=False, output_csv_name="result_without_filter.csv")
    run_ablation_experiment(test_video_file, enable_filter=True, output_csv_name="result_with_filter.csv")