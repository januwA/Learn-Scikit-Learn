import pandas as pd
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "btc_1h.json")

def train_and_predict():
    if not os.path.exists(DATA_FILE):
        print(f"错误: 找不到数据文件 {DATA_FILE}，请先运行 fetch_data.py")
        return

    # 1. 加载数据
    with open(DATA_FILE, "r") as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    print(f"成功加载数据，共 {len(df)} 行")

    # 2. 特征工程 (指标增强版)
    # 我们加入 RSI (相对强弱指标) 和 MA (移动平均线)
    
    # 计算 14 小时移动平均线 (SMA_14)
    df['ma_14'] = df['close'].rolling(window=14).mean()
    
    # 计算 RSI (相对强弱指标, 14周期)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # 保留传统的滞后特征
    lags = 5
    features = ['ma_14', 'rsi_14'] # 加入新特征
    
    for i in range(1, lags + 1):
        df[f'price_lag_{i}'] = df['close'].shift(i)
        df[f'vol_lag_{i}'] = df['volume'].shift(i)
        features.extend([f'price_lag_{i}', f'vol_lag_{i}'])
    
    # 剔除因为计算指标产生的前面几行空值
    df = df.dropna()

    # 特征字段：指标 + 过去5小时量价
    X = df[features]
    # 目标字段：当前的收盘价
    y = df['close']

    # 3. 划分数据集
    # 金融数据应按时间排序划分，不应打乱
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 4. 初始化并训练模型
    models = {
        "线性回归 (Linear)": LinearRegression(),
        "随机森林回归 (RandomForest)": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    print("\n--- 训练评估结果 (加上了成交量特征) ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"[{name}]")
        print(f"  均方误差 (MSE): {mse:.2f}")
        print(f"  确定系数 (R²): {r2:.4f}")

    # 5. 进行预测 (修复 Warning)
    # 构造一个和训练集 X 一模一样结构的 DataFrame
    last_row = df.iloc[-1]
    # 我们模拟“当前时刻”，其实预测的是“下一个时刻”
    # 实际上由于我们要预测未来，特征应该是当前及过去的数据
    current_features = df[features].iloc[-1:].copy()
    
    print("\n--- 预测下一小时 ---")
    for name, model in models.items():
        next_price = model.predict(current_features)
        print(f"[{name}] 预测价: {next_price[0]:.2f}")

if __name__ == "__main__":
    train_and_predict()
