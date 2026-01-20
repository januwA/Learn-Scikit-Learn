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
# 下载 30 张猫的图片
uv run python ./classify/download_cat.py

# 下载 30 张狗的图片
uv run python ./classify/download_dog.py
```

### 2. 训练模型
运行训练脚本来生成模型文件 (`.pkl`)：

```bash
uv run python ./classify/run_training.py
```

训练完成后，会在当前目录下生成 `cat_dog_model.pkl` 和 `label_encoder.pkl`。

### 3. 进行预测
使用训练好的模型来识别一张新的图片：

```bash
uv run python ./classify/run_predict.py "你的图片路径"
```

## 技术细节
- **图像处理**: 统一缩放为 64x64 像素，并转换为灰度图。
- **算法**: 使用线性核的支持向量机 (Linear SVC)。
- **持久化**: 使用 `joblib` 保存模型和标签编码器。

## 注意事项
- 本项目为一个基础示例，准确率受限于数据集大小和简单的像素特征提取方案。
- 模型文件 (`*.pkl`) 和下载的图片已被加入 `.gitignore`，不会提交到仓库。
