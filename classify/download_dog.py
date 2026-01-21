import urllib.request
import json
import os

def download_dog_images(count=1):
    """
    使用 Dog CEO API 下载指定数量的随机狗狗图片并保存到 data/dogs 目录下。
    """
    api_url = "https://dog.ceo/api/breeds/image/random"
    
    for i in range(count):
        try:
            # 1. 调用 API 获取图片 URL
            print(f"[{i+1}/{count}] 正在请求 API: {api_url}")
            with urllib.request.urlopen(api_url) as response:
                data = json.loads(response.read().decode())
                
            if data.get("status") == "success":
                image_url = data.get("message")
                print(f"成功获取图片 URL: {image_url}")
                
                # 2. 确保保存目录存在
                base_dir = os.path.dirname(os.path.abspath(__file__))
                save_dir = os.path.join(base_dir, "data", "dogs")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                # 3. 提取文件名
                file_name = image_url.split("/")[-1]
                save_path = os.path.join(save_dir, file_name)
                
                # 4. 下载图片
                print(f"正在下载图片到: {save_path}")
                urllib.request.urlretrieve(image_url, save_path)
                print(f"[{i+1}/{count}] 下载成功！")
                
            else:
                print(f"API 请求失败，状态码: {data.get('status')}")
                
        except Exception as e:
            print(f"程序运行出错: {e}")

if __name__ == "__main__":
    # 默认下载 1 张，可以修改这里的数字下载更多
    download_dog_images(50)
