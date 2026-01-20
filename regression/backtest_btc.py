import pandas as pd
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "btc_1h.json")

def prepare_features(df):
    """提取特征的逻辑提取出来，方便复用"""
    # 1. 技术指标
    df['ma_14'] = df['close'].rolling(window=14).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # 2. 滞后特征 (过去5小时)
    lags = 5
    features = ['ma_14', 'rsi_14']
    for i in range(1, lags + 1):
        df[f'price_lag_{i}'] = df['close'].shift(i)
        df[f'vol_lag_{i}'] = df['volume'].shift(i)
        features.extend([f'price_lag_{i}', f'vol_lag_{i}'])
    
    df = df.dropna()
    return df, features

def run_backtest():
    if not os.path.exists(DATA_FILE):
        print(f"错误: 找不到数据文件 {DATA_FILE}")
        return

    # 加载并处理特征
    with open(DATA_FILE, "r") as f:
        raw_data = json.load(f)
    df_raw = pd.DataFrame(raw_data)
    df, features = prepare_features(df_raw.copy())

    # 只要最后 100 条进行对比验证，起始位置大约在索引 400
    # 由于删除了前14个空值，我们要确保索引是对的
    total_len = len(df)
    test_size = 100
    start_point = total_len - test_size

    predictions = []
    actuals = []
    timestamps = []

    print(f"开始回测：从第 {start_point} 条数据开始，逐条预测未来 100 小时...")

    # 核心递归/递推逻辑
    for i in range(start_point, total_len):
        # 1. 划分训练集 (过去所有数据) 和 测试样本 (当前这一条)
        train_df = df.iloc[:i]
        test_sample = df.iloc[i:i+1] # 预测这一条

        X_train = train_df[features]
        y_train = train_df['close']
        X_test = test_sample[features]
        y_actual = test_sample['close'].values[0]
        # 获取当前时间戳并转换格式 (转换为北京时间 Asia/Shanghai)
        ts_ms = test_sample['timestamp'].values[0]
        readable_ts = pd.to_datetime(ts_ms, unit='ms').tz_localize('UTC').tz_convert('Asia/Shanghai').strftime('%Y-%m-%d %H:%M')

        # 2. 归一化 (Scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. 训练与预测
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)[0]

        predictions.append(y_pred)
        actuals.append(y_actual)
        timestamps.append(readable_ts)

        # 打印进度 (每20条打印一次)
        if (i - start_point + 1) % 20 == 0:
            print(f"进度: 已完成 {i - start_point + 1} / {test_size} 条预测")

    # 4. 计算最后 100 条的整体准确性
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print("\n--- 最终回测报告 (最后100条) ---")
    print(f"平均每小时预测误差 (RMSE): {rmse:.2f} USDT")
    print(f"预测与实际相关度 (R²): {r2:.4f}")

    # 看看最后 5 条的对比
    print("\n最后 5 条数据对比 (时间 | 预测 vs 实际):")
    for t, p, a in zip(timestamps[-5:], predictions[-5:], actuals[-5:]):
        diff = p - a
        print(f"[{t}] 预测: {p:.2f} | 实际: {a:.2f} | 偏差: {diff:+.2f}")

if __name__ == "__main__":
    run_backtest()
