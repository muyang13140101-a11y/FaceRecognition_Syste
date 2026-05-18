import urllib.request
import os

def download_model():
    # 业界公认的 MobileFaceNet PyTorch 开源权重安全下载链接
    url = "https://github.com/TreB1eN/InsightFace_Pytorch/raw/master/work_space/save/model_mobilefacenet.pth"
    save_dir = "models"
    file_name = "model_mobilefacenet.pth"
    save_path = os.path.join(save_dir, file_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"开始从 GitHub 下载官方开源权重文件，大概需要 4MB，请稍候...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"下载成功！已保存至：{save_path}")
    except Exception as e:
        print(f"下载失败。如果因为网络问题无法连接 GitHub，请查阅报错：{e}")
        print("如果是网络原因，你可以尝试手动用浏览器打开上方 URL 下载，并改名为 model_mobilefacenet.pth 放入 models 文件夹。")

if __name__ == "__main__":
    download_model()