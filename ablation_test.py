import cv2
import time
import csv
import numpy as np
from core.quality import QualityAssessor
from core.detector import FaceDetector 
# 1. 修改了这里的导入名称，与你的 extractor.py 保持绝对一致
from core.extractor import FaceExtractor
from core.matcher import FaceMatcher

def run_ablation_experiment(video_path, enable_filter, output_csv_name):
    print(f"正在运行测试，前置过滤状态：{enable_filter}...")
    quality_assessor = QualityAssessor(window_size=5, blur_threshold=54.80)
    detector = FaceDetector()
    
    # 2. 修改了这里的实例化名称
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
            start_time = time.time()
            
            # 自动化信道噪声物理指纹提取
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright_penalty = abs(np.mean(gray) - 128) / 128.0
            
            smoothed_score = 100.0
            cnn_activated = False
            identity = "Unknown"
            
            # 前置拦截门控
            if enable_filter:
                is_valid, smoothed_score = quality_assessor.assess_frame(frame)
                if not is_valid:
                    time_cost_ms = (time.time() - start_time) * 1000
                    writer.writerow([frame_id, time_cost_ms, smoothed_score, cnn_activated, "Blocked", lap_var, bright_penalty])
                    continue
            else:
                _, smoothed_score = quality_assessor.assess_frame(frame)
            
            # 重度网络管线
            cnn_activated = True
            faces = detector.detect(frame)
            if len(faces) > 0:
                face_crop = faces[0] 
                
                # 3. 修改了这里的函数调用，与你的 extract_feature 保持绝对一致
                features = extractor.extract_feature(face_crop)
                identity = matcher.match(features)
            
            time_cost_ms = (time.time() - start_time) * 1000
            writer.writerow([frame_id, time_cost_ms, smoothed_score, cnn_activated, identity, lap_var, bright_penalty])

    cap.release()
    print("自动化数据采集完成！已保存至: " + output_csv_name)

if __name__ == "__main__":
    # 请确保视频在同一目录下，或者填写完整的绝对路径
    test_video_file = "test_noise_video.mp4" 
    run_ablation_experiment(test_video_file, enable_filter=False, output_csv_name="result_without_filter.csv")
    run_ablation_experiment(test_video_file, enable_filter=True, output_csv_name="result_with_filter.csv")