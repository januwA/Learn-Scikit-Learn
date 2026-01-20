import json
import os
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# 设置中文字体（如果是 Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISSUES_FILE = os.path.join(BASE_DIR, "issues.json")
RESULTS_FILE = os.path.join(BASE_DIR, "clustering_results.json")

import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[`#*_\->+=|]', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chinese_tokenizer(text):
    return jieba.lcut(text)

def visualize_issues_pca():
    if not os.path.exists(ISSUES_FILE) or not os.path.exists(RESULTS_FILE):
        print("错误: 缺少 issues.json 或 clustering_results.json")
        return

    # 1. 加载原始数据
    with open(ISSUES_FILE, "r", encoding="utf-8") as f:
        issues = json.load(f)
    
    # 2. 加载聚类结果以获取标签
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 构建 issue_number -> cluster_id 的映射
    number_to_cluster = {}
    for cluster in results:
        cid = cluster['cluster_id']
        for issue in cluster['issues']:
            number_to_cluster[issue['number']] = cid

    # 准备特征数据和标签
    texts = []
    labels = []
    for issue in issues:
        texts.append(clean_text(f"{issue.get('title', '')} {issue.get('body', '') or ''}"))
        labels.append(number_to_cluster.get(issue['number'], -1))

    # 3. TF-IDF 向量化
    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None, max_df=0.8, min_df=2)
    X = vectorizer.fit_transform(texts)
    
    # 4. PCA 降维到 2 维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # 5. 可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("GitHub Issues 聚类结果可视化 (PCA 降维)")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")

    # 标注一些点的标题（可选）
    # for i, issue in enumerate(issues[:10]):
    #     plt.annotate(issue['title'][:10], (X_pca[i, 0], X_pca[i, 1]), fontsize=8)

    output_path = os.path.join(BASE_DIR, "issues_pca.png")
    plt.savefig(output_path)
    print(f"PCA 可视化完成，已保存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_issues_pca()
