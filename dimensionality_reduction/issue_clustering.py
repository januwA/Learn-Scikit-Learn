import json
import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISSUES_FILE = os.path.join(BASE_DIR, "issues.json")

import re

def clean_text(text):
    if not text:
        return ""
    # 1. 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 2. 移除 Markdown 链接和图片
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # 3. 移除特殊的 Markdown 语法
    text = re.sub(r'[`#*_\->+=|]', ' ', text)
    # 4. 只保留中文、英文和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 5. 合并多余空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chinese_tokenizer(text):
    """
    使用 jieba 进行中文分词
    """
    return jieba.lcut(text)

def cluster_issues(n_clusters=5):
    # 1. 读取数据
    if not os.path.exists(ISSUES_FILE):
        print(f"错误: 找不到文件 {ISSUES_FILE}")
        return

    with open(ISSUES_FILE, "r", encoding="utf-8") as f:
        issues = json.load(f)

    if not issues:
        print("没有可处理的 Issue")
        return

    # 2. 准备文本数据 (标题 + 内容)
    texts = []
    for issue in issues:
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        # 为了更好地聚类，我们将标题权重加大（例如重复两次）或者只用标题
        # 这里选择 标题 + 内容
        combined_text = f"{title} {body}"
        combined_text = clean_text(combined_text)
        texts.append(combined_text)

    print(f"共加载 {len(texts)} 条 Issue，准备开始聚类 (目标簇数: {n_clusters})...")

    # 定义中文停用词 (根据你的项目内容增加噪音词)
    STOP_WORDS = [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "怎么", "这个", "那个",
        "页面", "修改", "建议", "样式", "原型图", "增加", "显示", "功能", "点击", "内容", "统一", "位置", "出现", "需要", "已经", "还是", "以后", "之前", "大家", "可以", "接口", "返回", "提示", "应该", "图片", "zxt841614467", "jjjjjjjjp"
    ]

    # 3. 文本向量化 (TF-IDF)
    # 使用自定义分词器处理中文
    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        stop_words=STOP_WORDS, 
        max_df=0.4,      # 如果一个词在 40% 以上的文档都出现，说明它是通用废话，剔除
        min_df=2,        # 过滤掉冷门词
        token_pattern=None
    )
    
    X = vectorizer.fit_transform(texts)
    print(f"原始特征矩阵形状: {X.shape}")

    # --- 新增: 降维步骤 (LSA) ---
    # 使用 TruncatedSVD 处理稀疏矩阵，将成千上万个词维度压缩到 100 个语义维度
    from sklearn.decomposition import TruncatedSVD
    n_components = min(100, X.shape[1] - 1) # 确保维度不超过特征数
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    print(f"降维后特征矩阵形状: {X_reduced.shape} (保留方差: {svd.explained_variance_ratio_.sum():.2%})")

    # 4. 运行 K-Means 聚类 (使用降维后的数据)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X_reduced)
    
    labels = kmeans.labels_

    # 5. 整理结果为 JSON 格式
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(issues[idx])

    # 如果降维了，聚类中心也在降维后的空间
    # 我们需要将其投影回原始词空间，才能获取关键词
    original_space_centroids = np.dot(kmeans.cluster_centers_, svd.components_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    result_json = []

    for i in range(n_clusters):
        cluster_data = {
            "cluster_id": i,
            "count": len(clusters[i]),
            "keywords": [terms[ind] for ind in order_centroids[i, :8] if terms[ind].strip()],
            "issues": [
                {
                    "number": issue.get("number"),
                    "title": issue.get("title")
                } for issue in clusters[i]
            ]
        }
        result_json.append(cluster_data)

    # 6. 保存结果到 JSON 文件
    output_file = os.path.join(BASE_DIR, "clustering_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"\n聚类完成！结果已保存至: {output_file}")
    return result_json

if __name__ == "__main__":
    cluster_issues(n_clusters=4)
