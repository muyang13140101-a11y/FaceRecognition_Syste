import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image, ImageDraw

# 自动将 mtcnn-pytorch 目录加入 Python 寻址路径，以导入底层网络
current_dir = os.path.dirname(os.path.abspath(__file__))
mtcnn_dir = os.path.join(current_dir, 'mtcnn-pytorch')
sys.path.append(mtcnn_dir)

from src.get_nets import PNet, RNet, ONet
from src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from src.first_stage import run_first_stage

def draw_and_save_bboxes(image, bounding_boxes, filename, landmarks=None):
    """
    通用绘图函数：将当前阶段的边界框和关键点画在图像上并保存
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # 绘制矩形框
    for b in bounding_boxes:
        # b[0:4] 对应的就是 [xmin, ymin, xmax, ymax]
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="red", width=2)
    
    # 如果有关键点（O-Net阶段），则绘制关键点
    if landmarks is not None and len(landmarks) > 0:
        for l in landmarks:
            # l 包含了 10 个值: [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
            for i in range(5):
                x, y = l[i], l[i+5]
                # 用蓝色小圆点标出面部特征
                draw.ellipse([(x-2, y-2), (x+2, y+2)], fill="blue", outline="blue")
                
    img_draw.save(filename)
    print(f"✅ 已保存 {filename} | 当前剩余候选框数量: {len(bounding_boxes)}")

def main():
    # 你的目标图片路径
    img_path = r"D:\FaceRecognition_System\five_point.png"
    if not os.path.exists(img_path):
        print(f"❌ 找不到图片: {img_path}")
        return

    # 读取图像
    image = Image.open(img_path).convert('RGB')
    
    # 1. 实例化三个级联网络
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # 2. 设置 MTCNN 的核心超参数
    min_face_size = 20.0
    thresholds = [0.6, 0.7, 0.8]      # P, R, O 网络的置信度阈值
    nms_thresholds = [0.7, 0.7, 0.7]  # 非极大值抑制阈值

    # 3. 构建图像金字塔 (为 P-Net 准备不同尺度的图片)
    width, height = image.size
    min_length = min(height, width)
    min_detection_size = 12
    factor = 0.707
    scales = []
    m = min_detection_size / min_face_size
    min_length *= m
    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor**factor_count)
        min_length *= factor
        factor_count += 1

    # ==========================================
    # STAGE 1: P-Net 阶段
    # ==========================================
    print("\n🚀 开始 STAGE 1: P-Net (全卷积区域提议)...")
    bounding_boxes = []
    for s in scales:
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    bounding_boxes = [i for i in bounding_boxes if i is not None]
    if len(bounding_boxes) > 0:
        bounding_boxes = np.vstack(bounding_boxes)
        # 执行 NMS 剔除高度重叠框
        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    else:
        bounding_boxes = np.empty((0, 5))

    # 保存 P-Net 的输出照片
    draw_and_save_bboxes(image, bounding_boxes, "1_pnet_output.png")

    if len(bounding_boxes) == 0:
        return

    # ==========================================
    # STAGE 2: R-Net 阶段
    # ==========================================
    print("\n🚀 开始 STAGE 2: R-Net (精细过滤阶段)...")
    # 将 P-Net 的框截取并缩放为 24x24 输入 R-Net
    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    
    with torch.no_grad(): # 兼容新版 PyTorch，取消了 volatile
        img_boxes_tensor = Variable(torch.FloatTensor(img_boxes))
        output = rnet(img_boxes_tensor)
        offsets = output[0].data.numpy()
        probs = output[1].data.numpy()

    # 剔除置信度低于 0.7 的伪阳性框
    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # 保存 R-Net 的输出照片
    draw_and_save_bboxes(image, bounding_boxes, "2_rnet_output.png")

    if len(bounding_boxes) == 0:
        return

    # ==========================================
    # STAGE 3: O-Net 阶段
    # ==========================================
    print("\n🚀 开始 STAGE 3: O-Net (终极判决与关键点输出)...")
    # 将 R-Net 剩下的极少数框截取并缩放为 48x48 输入 O-Net
    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    
    with torch.no_grad():
        img_boxes_tensor = Variable(torch.FloatTensor(img_boxes))
        output = onet(img_boxes_tensor)
        landmarks = output[0].data.numpy()
        offsets = output[1].data.numpy()
        probs = output[2].data.numpy()

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # 将 0~1 的相对坐标转换为原图上的绝对像素坐标
    b_width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    b_height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(b_width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(b_height, 1) * landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    # 保存 O-Net 的输出照片 (包含边界框 + 蓝色关键点)
    draw_and_save_bboxes(image, bounding_boxes, "3_onet_output.png", landmarks)
    print("\n🎉 三个阶段拆解完毕，请查看当前目录下的输出图片！")

if __name__ == '__main__':
    main()