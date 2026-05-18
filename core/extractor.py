import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

class FaceExtractor:
    def __init__(self):
        """
        使用极其稳定的 facenet-pytorch 库。
        它会自动加载 InceptionResnetV1 网络骨架，并在首次运行时自动下载合法的预训练权重。
        """
        self.device = torch.device('cpu')
        print(f"[INFO] 正在初始化特征提取器 (Facenet PyTorch)，运行在 {self.device}...")
        
        # 实例化模型并自动下载匹配的权重，切入推理模式
        # 这里的 vggface2 是业界公认最权威的人脸识别开源权重之一
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # 定义预处理流水线 (将像素值标准化到模型期望的范围)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_feature(self, face_image_rgb):
        """
        传入单张人脸的 RGB 数组，返回一个极其稳定的 512 维特征向量 (numpy array)
        """
        # 1. Facenet 模型要求输入的标准尺寸为 160x160
        resized_face = cv2.resize(face_image_rgb, (160, 160))
        
        # 2. 预处理转为 Tensor，并增加 Batch 维度
        tensor = self.transform(resized_face).unsqueeze(0).to(self.device)
        
        # 3. 前向传播提取特征 (关闭梯度计算以提速)
        with torch.no_grad():
            feature = self.model(tensor)
            
        # 返回脱离了张量计算图的 numpy 数组
        return feature.numpy()[0].astype(np.float32)