import urllib.request
import json
import os

def download_cat_images(limit=30):
    """
    使用 The Cat API 下载指定数量的随机猫咪图片并保存到 data/cats 目录下。
    """
    api_url = f"https://api.thecatapi.com/v1/images/search?limit={limit}"
    
    try:
        # 1. 调用 API 获取图片数据列表
        print(f"正在请求 API: {api_url}")
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            
        if isinstance(data, list):
            # 2. 确保保存目录存在
            # 使用相对于当前脚本所在目录的路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_dir, "data", "cats")
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"创建目录: {save_dir}")
            
            # 3. 遍历下载每一张图片
            for i, item in enumerate(data):
                image_url = item.get("url")
                if not image_url:
                    continue
                
                # 提取文件名
                file_name = image_url.split("/")[-1]
                save_path = os.path.join(save_dir, file_name)
                
                print(f"[{i+1}/{len(data)}] 正在下载图片: {image_url}")
                try:
                    urllib.request.urlretrieve(image_url, save_path)
                    print(f"[{i+1}/{len(data)}] 下载成功并保存到: {save_path}")
                except Exception as e:
                    print(f"[{i+1}/{len(data)}] 下载失败: {e}")
                    
        else:
            print(f"API 返回数据格式不正确: {data}")
            
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    # 没有key, api 实际好像只能返回10条数据
    download_cat_images(50)
