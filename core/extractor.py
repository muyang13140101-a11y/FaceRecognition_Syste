import os
import torch
import cv2
import numpy as np
from torchvision import transforms

# 导入刚才写好的旗舰级骨干网络
from models.resnet import IR_SE_50

class FaceExtractor:
    def __init__(self):
        """
        初始化特征提取器
        采用 ArcFace 官方的旗舰级架构：IR-SE50 (ResNet50)
        提供顶级的人脸识别精度。
        """
        self.device = torch.device('cpu')
        print(f"[INFO] 正在初始化特征提取器 (IR-SE50 旗舰版), 运行在 {self.device}...")
        
        # 1. 实例化真正的 IR-SE50 骨架
        self.model = IR_SE_50().to(self.device)
        
        # 2. 加载你的权重文件
        weight_path = "./models/model.pth" 
        
        if not os.path.exists(weight_path):
            print(f"[ERROR] 找不到权重文件: {weight_path}，请检查路径！")
        else:
            state_dict = torch.load(weight_path, map_location=self.device)
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
            print("[SUCCESS] 成功加载 IR-SE50 顶级预训练权重！(识别精度大幅提升)")
        
        # 3. 切入推理模式
        self.model.eval()
        
        # 4. 预处理流水线
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_feature(self, face_image_rgb):
        """
        传入 112x112 的端正人脸，返回 512 维特征向量。
        """
        if face_image_rgb.shape[:2] != (112, 112):
            face_image_rgb = cv2.resize(face_image_rgb, (112, 112))
        
        tensor = self.transform(face_image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model(tensor)
            
        return feature.numpy()[0].astype(np.float32)