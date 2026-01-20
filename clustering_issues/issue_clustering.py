import json
import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISSUES_FILE = os.path.join(BASE_DIR, "issues.json")

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
        texts.append(combined_text)

    print(f"共加载 {len(texts)} 条 Issue，准备开始聚类 (目标簇数: {n_clusters})...")

    # 3. 文本向量化 (TF-IDF)
    # 使用自定义分词器处理中文
    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        stop_words=None, # 如果有停用词表可以加上
        max_df=0.8,      # 过滤掉在 80% 以上文档中出现的词
        min_df=2,        # 过滤掉只出现 1 次的冷门词
        token_pattern=None
    )
    
    X = vectorizer.fit_transform(texts)
    print(f"特征矩阵形状: {X.shape}")

    # 4. 运行 K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)
    
    labels = kmeans.labels_

    # 5. 整理结果为 JSON 格式
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(issues[idx])

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
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
    cluster_issues(n_clusters=6)
