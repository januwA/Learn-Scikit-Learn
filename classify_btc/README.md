# Classification 学习：BTC 涨跌预测

本项目使用 Scikit-Learn 的**分类 (Classification)** 算法来预测下一小时 BTC 是涨还是跌。

## 什么是分类？
不同于回归预测具体的“价格数字”，分类预测的是“类别”。
- **类别 1 (涨)**: 下一小时收盘价 > 当前小时收盘价
- **类别 0 (跌)**: 下一小时收盘价 <= 当前小时收盘价

## 实验流程
1. **数据准备**: 复用 `regression/btc_1h.json` 中的数据。
2. **标签制作**: 创建 `target` 列，计算 `close.shift(-1) > close`。
3. **训练模型** (`classify_btc.py`): 使用 `LogisticRegression` 或 `RandomForestClassifier`。
4. **评估**: 查看“准确率 (Accuracy)”——预测 100 次涨跌，能对多少次。

## 运行
`python classify_btc.py`
