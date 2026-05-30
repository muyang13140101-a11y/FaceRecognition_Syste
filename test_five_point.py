import cv2
import matplotlib.pyplot as plt
from core.detector import FaceDetector

def main():
    # 1. 实例化你自己的高级检测器类
    print("[INFO] 正在加载检测器模型...")
    detector = FaceDetector()

    # 2. 读取原始图片
    img_path = r"D:\FaceRecognition_System\five_point.png"
    image = cv2.imread(img_path)

    if image is None:
        print(f"[ERROR] 无法读取图片，请检查路径是否正确: {img_path}")
        return

    # 3. 颜色转换，转为 RGB 给 MTCNN 使用
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 4. 直接调用 MTCNN 底层的检测方法获取框和关键点数据
    results = detector.detector.detect_faces(rgb_image)
    
    if not results:
        print("[WARNING] 未能在图片中检测到人脸！")
        return
        
    print(f"[INFO] 成功检测到 {len(results)} 张人脸。正在绘制边界框和关键点...")

    # 5. 在原图副本上进行绘制
    draw_image = image.copy()
    
    for result in results:
        # 【修改点 1】：严格对齐你论文的默认阈值 0.60
        if result['confidence'] >= 0.60:
            
            # 【修改点 2 & 3】：人脸框粗一点（线宽设为5），不要绿色，改为醒目的纯蓝色 (255, 0, 0)
            x, y, w, h = result['box']
            cv2.rectangle(draw_image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            
            # 【修改点 4】：关键点大一点，半径从 4 提升到 8，保持纯红色 (0, 0, 255)
            keypoints = result['keypoints']
            cv2.circle(draw_image, keypoints['left_eye'], 8, (0, 0, 255), -1)
            cv2.circle(draw_image, keypoints['right_eye'], 8, (0, 0, 255), -1)
            cv2.circle(draw_image, keypoints['nose'], 8, (0, 0, 255), -1)
            cv2.circle(draw_image, keypoints['mouth_left'], 8, (0, 0, 255), -1)
            cv2.circle(draw_image, keypoints['mouth_right'], 8, (0, 0, 255), -1)
            
    # 6. 保存最终生成的图片
    output_path = r"D:\FaceRecognition_System\five_point_result.png"
    cv2.imwrite(output_path, draw_image)
    print(f"[SUCCESS] 处理完成！带框和关键点的图片已保存至: {output_path}")
    
    # 7. 用 matplotlib 直接在屏幕上弹出展示
    plt.figure(figsize=(8, 8))
    plt.title("Face Detection & 5-Point Alignment (Threshold: 0.60)")
    plt.imshow(cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()