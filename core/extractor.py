import os
import torch
import cv2
import numpy as np
from torchvision import transforms

# 导入你本地定义的 MobileFaceNet 网络骨架 
# (对应你项目里的 models/mobilefacenet.py)
from models.mobilefacenet import MobileFaceNet

class FaceExtractor:
    def __init__(self):
        """
        初始化特征提取器
        核心升级：彻底切入 MobileFaceNet 架构，对接 ArcFace 顶级预训练权重。
        在 CPU 上实现毫秒级特征提取。
        """
        self.device = torch.device('cpu')
        print(f"[INFO] 正在初始化特征提取器 (MobileFaceNet)，运行在 {self.device}...")
        
        # 1. 实例化你本地的 MobileFaceNet 架构 (输出 512 维特征)
        self.model = MobileFaceNet(512).to(self.device)
        
        # 2. 加载你下载好的顶级预训练权重
        # 【注意】请确保这个路径指向你真实的 model_mobilefacenet.pth 文件！
        weight_path = "./model_mobilefacenet.pth" 
        
        if not os.path.exists(weight_path):
            print(f"[ERROR] 找不到权重文件: {weight_path}，请检查路径！")
        else:
            # 加载字典并解决可能的 module. 前缀问题
            state_dict = torch.load(weight_path, map_location=self.device)
            # 如果是 DataParallel 保存的权重，可能会有 'module.' 前缀，需要过滤
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
            print("[SUCCESS] 成功加载 model_mobilefacenet.pth 顶级预训练权重！")
        
        # 3. 切入推理模式 (极其重要：冻结 BatchNorm 和 Dropout)
        self.model.eval()
        
        # 4. 定义预处理流水线
        # ArcFace 官方数据分布要求：将 [0, 255] 的像素映射到 [-1, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 转为张量并除以 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_feature(self, face_image_rgb):
        """
        传入 112x112 的端正人脸 RGB 数组，返回 L2 范数归一化后的 512 维特征向量。
        """
        # A. 尺寸防呆机制：对接 detector.py 传过来的 112x112 标准人脸
        if face_image_rgb.shape[:2] != (112, 112):
            face_image_rgb = cv2.resize(face_image_rgb, (112, 112))
        
        # B. 预处理转为 Tensor，并增加 Batch 维度
        tensor = self.transform(face_image_rgb).unsqueeze(0).to(self.device)
        
        # C. 前向传播提取 512 维特征
        with torch.no_grad():
            feature = self.model(tensor)
            
        # ==========================================
        # 核心学术逻辑：特征的 L2 范数归一化 (L2 Normalization)
        # ==========================================
        # 严丝合缝地对应了你论文【4.1节】的式 (4-2)！
        # 强行剥离光照剧变带来的模长噪声，将特征强制投影到单位超球面上！
        feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            
        # 返回 numpy 数组供外层 SQLite 或余弦相似度计算使用
        return feature.numpy()[0].astype(np.float32)