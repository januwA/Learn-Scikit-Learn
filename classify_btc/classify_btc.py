import pandas as pd
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 路径配置
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "regression", "btc_1h.json")

def train_classification():
    if not os.path.exists(DATA_FILE):
        print("错误: 找不到数据文件，请先确保 regression 目录下有 btc_1h.json")
        return

    # 1. 加载数据
    with open(DATA_FILE, "r") as f:
        df = pd.DataFrame(json.load(f))

    # 2. 特征工程 (复用之前的指标)
    df['ma_14'] = df['close'].rolling(window=14).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain/loss))
    
    for i in range(1, 4):
        df[f'price_lag_{i}'] = df['close'].shift(i)
    
    # 3. 关键一步：制作分类标签 (Target)
    # 我们预测下一小时是否比当前小时高
    # 1 = 涨, 0 = 跌
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df = df.dropna()

    features = ['ma_14', 'rsi_14', 'price_lag_1', 'price_lag_2', 'price_lag_3']
    X = df[features]
    y = df['target']

    # 4. 训练/测试集切分 (由于是分类，我们看看整体准确率)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 归一化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 模型对比
    models = {
        "逻辑回归 (Logistic)": LogisticRegression(),
        "随机森林分类 (RandomForest)": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    print("\n--- BTC 涨跌分类预测结果 ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name}] 准确率: {acc:.2%}")
        
    print("\n提示：50% 准确率相当于抛硬币。在高手如云的金融市场，能稳定在 55% 以上就是大神了。")

if __name__ == "__main__":
    train_classification()
