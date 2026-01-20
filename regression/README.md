# Regression 学习：BTC 价格预测

本项目使用 Scikit-Learn 的回归算法来预测 BTC 的价格走势。

## 实验流程
1. **获取数据** (`fetch_data.py`): 使用 `ccxt` 从 Binance 下载最近 500 条 BTC/USDT 1h K线数据并存为 `btc_1h.json`。
2. **特征处理**: 将 K线数据转换为模型可理解的特征（如前期价格）。
3. **训练模型** (`predict_btc.py`): 使用回归模型（如线性回归、随机森林等）进行预测。

## 运行
1. 安装依赖: `uv sync` (或 `pip install ccxt pandas scikit-learn`)
2. 下载数据: `python fetch_data.py`
3. 执行预测: `python predict_btc.py`
