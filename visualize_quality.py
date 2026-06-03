import cv2
import numpy as np
import os

def visualize_quality_process(image_path):
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}")
        return

    # 读取原图
    original_img = cv2.imread(image_path)
    
    # ==========================================
    # 步骤 1：灰度转换与光照评估可视化
    # ==========================================
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_img)
    brightness_penalty = abs(mean_brightness - 128) / 128.0
    
    # 将灰度图复制一份用于绘制文字
    gray_vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(gray_vis, f"Mean Brightness: {mean_brightness:.1f} (Ideal: 128)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(gray_vis, f"Brightness Penalty: {brightness_penalty:.3f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite("Quality_Step1_Grayscale.png", gray_vis)
    print("✅ 已生成: Quality_Step1_Grayscale.png (展示光照分布)")

    # ==========================================
    # 步骤 2：拉普拉斯算子 (高频边缘提取) 可视化
    # ==========================================
    # cv2.CV_64F 可以保留负数和大于255的值，确保计算方差的绝对精准
    laplacian_64f = cv2.Laplacian(gray_img, cv2.CV_64F)
    laplacian_var = laplacian_64f.var()
    
    # 为了让人眼能看到(可视化)，我们需要将 64位浮点数 取绝对值并压缩回 0-255(uint8)
    laplacian_vis = cv2.convertScaleAbs(laplacian_64f)
    
    # 转换回BGR空间以便画彩色文字
    laplacian_vis_color = cv2.cvtColor(laplacian_vis, cv2.COLOR_GRAY2BGR)
    
    # 判断是否达到你代码中的阈值 (blur_threshold=60.0)
    threshold = 60.0
    status = "PASS (Clear)" if laplacian_var > threshold else "FAIL (Blurry)"
    color = (0, 255, 0) if laplacian_var > threshold else (0, 0, 255)
    
    cv2.putText(laplacian_vis_color, f"Laplacian Variance: {laplacian_var:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(laplacian_vis_color, f"Status: {status} (Thresh: {threshold})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite("Quality_Step2_Laplacian.png", laplacian_vis_color)
    print("✅ 已生成: Quality_Step2_Laplacian.png (展示高频边缘特征)")

    # 打印最终的综合得分 (对应你 quality.py 中的单帧计算公式)
    single_frame_score = laplacian_var * (1.0 - 0.2 * brightness_penalty)
    print(f"\n📊 最终单帧综合质量得分: {single_frame_score:.2f}")

if __name__ == "__main__":
    target_image = "five_point.png"
    visualize_quality_process(target_image)