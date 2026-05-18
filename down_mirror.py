import urllib.request
import os

# 确保文件夹存在
os.makedirs("./eval_test", exist_ok=True)

# 使用国内稳定的 GitHub 加速代理，下载 Facenet 仓库里备份的 pairs.txt
url = "https://ghproxy.net/https://raw.githubusercontent.com/davidsandberg/facenet/master/data/pairs.txt"
save_path = "./eval_test/pairs.txt"

print("[INFO] 正在通过国内加速节点获取 pairs.txt...")

try:
    # 设置 15 秒超时
    urllib.request.urlretrieve(url, save_path)
    print(f"\n[SUCCESS] 下载完美成功！文件已保存至: {save_path}")
except Exception as e:
    print(f"\n[ERROR] 下载失败，报错为: {e}")