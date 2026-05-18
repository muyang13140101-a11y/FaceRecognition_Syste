import urllib.request
import os

# 确保文件夹存在
os.makedirs("./eval_test", exist_ok=True)

url = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
save_path = "./eval_test/pairs.txt"

print("[INFO] 正在伪装成 Chrome 浏览器向 LFW 服务器请求下载...")

# 伪装头部，假装我们是一个正常的人类在用 Chrome 浏览器访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

req = urllib.request.Request(url, headers=headers)

try:
    # 设置 30 秒超时，防止卡死
    with urllib.request.urlopen(req, timeout=30) as response:
        with open(save_path, 'wb') as f:
            f.write(response.read())
    print(f"\n[SUCCESS] 下载完美成功！文件已保存至: {save_path}")
except Exception as e:
    print(f"\n[ERROR] 下载依然失败，网络报错为: {e}")
    print("👉 备用方案：请直接复制网址 http://vis-www.cs.umass.edu/lfw/pairs.txt 到电脑浏览器里打开。")
    print("👉 网页加载出来后，按下键盘上的 Ctrl + S，将网页另存为 pairs.txt 并放到 eval_test 目录下即可。")