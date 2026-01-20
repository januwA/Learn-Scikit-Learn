# Cat vs Dog Image Classifier (Scikit-Learn)

这是一个使用 Scikit-Learn 实现的简单猫狗图像分类器。它通过将图像缩放并展平为像素向量，使用支持向量机 (SVM) 进行训练和预测。

## 项目结构

- `classify_cats_dogs.py`: 核心逻辑库，包含图像预处理、模型训练和预测函数。
- `run_training.py`: 训练脚本，加载数据并生成模型文件。
- `run_predict.py`: 预测脚本，支持从命令行传入图片路径进行识别。
- `download_cat.py`: 从 The Cat API 批量下载猫咪图片。
- `download_dog.py`: 从 Dog CEO API 批量下载狗狗图片。
- `data/`: 存放训练数据的目录（`cats/` 和 `dogs/`）。

## 快速开始

### 1. 准备数据
首先需要下载一些图片用于训练。你可以运行下载脚本：

```bash
# 从 API 下载图片
uv run python ./classify/download_cat.py
uv run python ./classify/download_dog.py
```

### 2. 训练模型
运行训练脚本来生成模型文件 (`.pkl`)：

```bash
uv run python ./classify/run_training.py
```

训练完成后，会在 `classify/` 目录下生成 `cat_dog_model.pkl` 和 `label_encoder.pkl`。

### 3. 进行预测
使用训练好的模型来识别一张新的图片。预测结果会包含**各个类别的概率**以及**最终结果的置信度**：

```bash
uv run python ./classify/run_predict.py "你的图片路径"
```

**输出示例：**
```
预测概率: [0.229 0.771]
预测结果: 这是一只 dogs (置信度: 77.05%)
```

## 技术细节
- **图像预处理**: 统一缩放为 64x64 像素，转换为灰度图，并进行归一化（0-1）。
- **算法**: 使用支持向量机 (SVC)，核心配置为 `kernel='linear'`。
- **概率估算**: 启用了 `probability=True` 以便获取预测时的置信度。
- **持久化**: 使用 `joblib` 实现高效的对象序列化。

## 局限性与改进方向
目前的模型准确率可能在 50%~70% 之间波动，这在图像识别中属于较低水平，原因如下：
1. **特征提取原始**：直接展平像素对位置、光照和背景极其敏感。建议引入 **HOG (方向梯度直方图)** 等特征。
2. **算法简单**：线性核处理非线性图像数据能力有限。可以尝试 `kernel='rbf'`。
3. **数据量不足**：几十张图片难以覆盖猫狗的所有形态。建议下载数百张以上的图片进行训练。

## 注意事项
- 模型文件 (`*.pkl`) 和 `data/` 目录下的图片已被加入 `.gitignore`。
- 脚本内的路径均已处理为相对脚本所在位置的绝对路径，可在任何工作目录下放心调用。
