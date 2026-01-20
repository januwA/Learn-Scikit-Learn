from classify_cats_dogs import predict
import sys
import os

def run_predict():
    # 检查是否提供了脚本参数
    if len(sys.argv) < 2:
        print("用法: python run_predict.py <图片路径>")
        print("示例: python run_predict.py data/cats/my_cat.jpg")
        return

    image_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return

    # 调用预测函数
    predict(image_path)

if __name__ == "__main__":
    run_predict()
