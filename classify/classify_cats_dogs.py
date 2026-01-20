import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 图像大小，统一缩放到 64x64
IMG_SIZE = 64

# 定义基础目录为脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cat_dog_model.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, 'data')

def load_data(data_dir):
    """
    加载图像数据并进行预处理
    数据目录结构建议:
    data/
        cats/
            cat1.jpg
            ...
        dogs/
            dog1.jpg
            ...
    """
    data = []
    labels = []
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在。")
        return None, None

    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        print(f"正在加载类别: {category}")
        for img_name in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, img_name)
                # 使用 OpenCV 读取图像并转换为灰度图以减少计算量（也可以用彩色）
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # 缩放图像
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # 将图像展平为一维向量
                data.append(img.flatten())
                labels.append(category)
            except Exception as e:
                print(f"无法加载图像 {img_name}: {e}")
                
    return np.array(data), np.array(labels)

def train():
    # 数据路径，默认为脚本所在目录下的 data 文件夹
    DATA_PATH = DEFAULT_DATA_PATH
    
    print("正在加载数据...")
    X, y = load_data(DATA_PATH)
    
    if X is None or len(X) == 0:
        print("未找到数据。请确保 'data' 目录下有 'cats' 和 'dogs' 文件夹，并放入图片。")
        return

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 归一化像素值 (0-255 -> 0-1)
    X = X / 255.0
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"训练集规模: {X_train.shape[0]}, 测试集规模: {X_test.shape[0]}")
    
    # 初始化支持向量机 (SVM) 分类器
    # C 是正则化参数，gamma 是核函数系数
    print("正在开始训练 SVM 模型...")
    clf = SVC(kernel='linear', C=1.0, probability=True) # 启用概率估算以获取置信度
    clf.fit(X_train, y_train)
    
    # 预测并评估
    y_pred = clf.predict(X_test)
    print("\n模型评估:")
    print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
    print("\n详细报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 保存模型和标签编码器
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\n模型已保存为 '{MODEL_PATH}'")

def predict(image_path):
    """
    使用训练好的模型预测新图片
    """
    if not os.path.exists(MODEL_PATH):
        print("未找到训练好的模型，请先运行训练函数。")
        return
        
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    
    # 处理单张图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("无法读取图片。")
        return
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flat = img.flatten().reshape(1, -1) / 255.0
    
    # 获取预测概率
    probabilities = clf.predict_proba(img_flat)[0]
    print(f"预测概率: {probabilities}")
    prediction_index = np.argmax(probabilities)
    label = le.inverse_transform([prediction_index])[0]
    
    print(f"预测结果: 这是一只 {label} (置信度: {probabilities[prediction_index]:.2%})")

if __name__ == "__main__":
    # 如果你想训练，请取消下面注释并确保有数据
    # train()
    
    # 如果你想预测单张图片，请使用 predict 函数
    # predict('test_image.jpg')
    print("脚本已就绪。请将猫和狗的图片分别放在 'data/cats' 和 'data/dogs' 目录下，然后调用 train() 函数。")
