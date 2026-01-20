from classify_cats_dogs import train, predict
import os

# 这是一个演示脚本
def run_demo():
    # 获取脚本所在目录下的 data 路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cats_dir = os.path.join(base_dir, 'data', 'cats')
    
    # 检查是否有数据
    if not os.path.exists(cats_dir) or len(os.listdir(cats_dir)) == 0:
        print(f"提示：'{cats_dir}' 目录下没有图片。")
        print("请运行 download_cat.py 和 download_dog.py 下载图片。")
        return

    # 开始训练
    print("--- 开始运行训练流程 ---")
    train()
    
    # 如果训练成功，可以尝试预测一张图片
    # 这里只是举例，需要你有一张图片路径
    # predict('data/cats/cat_001.jpg')

if __name__ == "__main__":
    run_demo()
