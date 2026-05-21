import cv2

import numpy as np

import math

import matplotlib.pyplot as plt

from facenet_pytorch import MTCNN



# ==========================================

# 严格遵循毕业论文排版要求 (宋体五号)

# ==========================================

plt.rcParams['font.size'] = 10.5

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['SimSun']

plt.rcParams['axes.unicode_minus'] = False



def main():

    print("[INFO] 正在加载 MTCNN 级联感知网络...")

    # 初始化 MTCNN

    mtcnn = MTCNN(keep_all=False, device='cpu')



    # 读取你刚才拍的歪头照片

    img_path = "./tilted_face.jpg" 

    img = cv2.imread(img_path)

    

    if img is None:

        print(f"[ERROR] 找不到图片: {img_path}")

        print("请用手机拍一张歪着头的照片，命名为 tilted_face.jpg 放在项目目录下！")

        return

        

    # OpenCV 默认是 BGR 格式，转换为 RGB

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    print("[INFO] 正在进行像素级感知与 5 点拓扑定位...")

    # 核心算法：MTCNN 检测边界框与 5 个关键点

    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)



    if boxes is None or landmarks is None:

        print("[ERROR] 未检测到人脸，请换一张脸部清晰的图片。")

        return



    box = boxes[0]

    points = landmarks[0] # 5个关键点: [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]



    # ==========================================

    # 绘制左半边图：原始图像 + 5个关键点标定

    # ==========================================

    img_raw_draw = img_rgb.copy()

    

    # 遍历 5 个关键点并画图

    for point in points:

        x, y = int(point[0]), int(point[1])

        # 画个显眼的绿点，外加一个白圈，学术感拉满

        cv2.circle(img_raw_draw, (x, y), 5, (0, 255, 0), -1) 

        cv2.circle(img_raw_draw, (x, y), 6, (255, 255, 255), 1)



    # 画人脸感知候选框 (红色虚线感)

    x1, y1, x2, y2 = [int(b) for b in box]

    cv2.rectangle(img_raw_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)



    # ==========================================

    # 算法原理：基于双眼坐标的 2D 仿射变换纠偏

    # ==========================================

    left_eye = points[0]

    right_eye = points[1]

    

    # 计算双眼中心点坐标

    eye_center = (int((left_eye[0] + right_eye[0]) / 2), 

                  int((left_eye[1] + right_eye[1]) / 2))

    

    # 利用坐标差计算出头部的物理倾斜角度

    dy = right_eye[1] - left_eye[1]

    dx = right_eye[0] - left_eye[0]

    angle = math.degrees(math.atan2(dy, dx))

    

    print(f"[INFO] 计算出面部倾斜角度为: {angle:.2f} 度，正在生成仿射变换矩阵...")



    # 利用 OpenCV 获取 2D 仿射变换矩阵 (旋转并将脸部摆正)

    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)

    

    # 对整张图片执行仿射变换

    h, w = img_rgb.shape[:2]

    img_aligned = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC)



    # 重新裁剪出端正的人脸区域 (扩大30像素边缘，保证下巴不被切掉)

    margin = 30

    crop_x1 = max(0, x1 - margin)

    crop_y1 = max(0, y1 - margin)

    crop_x2 = min(w, x2 + margin)

    crop_y2 = min(h, y2 + margin)

    

    face_aligned_cropped = img_aligned[crop_y1:crop_y2, crop_x1:crop_x2]

    face_raw_cropped = img_raw_draw[crop_y1:crop_y2, crop_x1:crop_x2]



    # ==========================================

    # 拼图并保存：生成最终的学术图表 3-2

    # ==========================================

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    

    axes[0].imshow(face_raw_cropped)

    axes[0].set_title("原始面部畸变图像\n(附带 MTCNN 5点拓扑坐标)", fontsize=11, pad=10)

    axes[0].axis('off')

    

    axes[1].imshow(face_aligned_cropped)

    axes[1].set_title("2D 仿射变换与对齐后图像\n(消除空间倾斜，输出端正表征)", fontsize=11, pad=10)

    axes[1].axis('off')



    plt.tight_layout()

    save_path = "./eval_test/Fig3-2_Face_Alignment.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    

    print(f"[SUCCESS] 极其直观的 图 3-2 已经成功生成！快去 {save_path} 看看吧！")



if __name__ == "__main__":

    main()