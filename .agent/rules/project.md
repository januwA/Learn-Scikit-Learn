# 项目结构与工作流规则

## 目录组织
- **根目录 (`/`)**: 主工作区。包含 Python 虚拟环境、`pyproject.toml` 以及总体的学习文档。
- **`classify/`**: 专注于 **监督学习 (Supervised Learning)**。处理带有标签的数据任务，如猫狗图像识别。
- **`clustering_issues/`**: 专注于 **无监督学习 (Unsupervised Learning)**。处理未打标签的数据模式发现，如 GitHub Issue 聚类。

## 环境管理
- 始终使用位于根目录的 Python 环境。
- 所有依赖项应通过根目录的 `pyproject.toml` 并使用 `uv` 工具进行统一管理。

## 编码规范
- 所有子目录中的脚本必须使用稳健的路径处理方式（例如 `os.path.dirname(os.path.abspath(__file__))`），以确保无论是在根目录还是在其自身目录下运行都能正常执行。
- 每个子目录应维护独立的 `README.md` 文件，用于记录该模块特定的学习目标和实践结果。
