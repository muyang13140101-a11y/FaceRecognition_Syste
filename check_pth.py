import torch

# 把这里的路径换成你实际 model.pth 的路径
file_path = "D:\FaceRecognition_System\models\model.pth" 

print(f"正在透视文件: {file_path}")
data = torch.load(file_path, map_location="cpu", weights_only=False)

print("\n=== 透视结果 ===")
print("文件类型:", type(data))

if isinstance(data, dict):
    print("结论：这是一个【纯权重文件】（格式一）。")
    print("里面只包含了参数字典，必须要有完全匹配的 .py 网络代码才能运行。")
    # 随便打印几个键名看看
    print("\n前三个键名是:", list(data.keys())[:3])
else:
    print("结论：这是一个【完整模型文件】（格式二）！")
    print("太棒了！你不需要任何额外的 .py 网络代码，直接用它就可以提取特征！")