import json
import os
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISSUES_FILE = os.path.join(BASE_DIR, "issues.json")

# 预定义的停用词
STOP_WORDS = {
    'discussed', 'originally', 'posted', 'by', 'in', 'ref', 'github', 'attachments', 'assets', 'image',
    'the', 'a', 'an', 'and', 'or', 'of', 'for', 'with', 'on', 'at', 'to', 'from', 'this', 'that', 'it',
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '为', '与', '及', '等',
    '这里', '这些', '这个', '这样', '之后', '之后', '点击', '跳转', '已经', '目前'
}

def clean_text(text):
    """
    加强版预处理：
    1. 移除 Markdown/HTML 噪音
    2. 只保留中文字符和英文字母（彻底干掉 ###, *, , 等符号）
    3. 统一转小写
    """
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    
    # 只保留中文字符 \u4e00-\u9fa5 和 英文字符 a-zA-Z
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
    
    return text.strip().lower()

def chinese_tokenizer(text):
    """
    使用 jieba 进行中文分词，并过滤掉停用词和单字符词
    """
    words = jieba.lcut(text)
    # 过滤掉停用词，以及长度小于等于 1 的词（如：的、了、过）
    return [w for w in words if w.strip() and w not in STOP_WORDS and len(w) > 1]

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

    # 2. 准备并预处理文本数据
    print("正在进行深度文本预处理 (Cleaning & Filtering)...")
    texts = []
    for issue in issues:
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        # 给标题增加权重：将标题重复 3 次，使关键词更贴近标题
        raw_text = f"{title} {title} {title} {body}"
        cleaned_text = clean_text(raw_text)
        texts.append(cleaned_text)

    print(f"共加载 {len(texts)} 条 Issue，准备开始聚类 (目标簇数: {n_clusters})...")

    # 3. 文本向量化 (TF-IDF)
    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        stop_words=None, # 我们在分词器里手动过滤了
        max_df=0.8,
        min_df=2,
        token_pattern=None
    )
    
    X = vectorizer.fit_transform(texts)
    
    # 显式归一化
    normalizer = Normalizer(copy=False)
    X = normalizer.fit_transform(X)
    
    print(f"特征矩阵形状 (预处理后): {X.shape}")

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
    cluster_issues(n_clusters=3)
