import json
import os
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISSUES_FILE = os.path.join(BASE_DIR, "issues.json")
RESULTS_FILE = os.path.join(BASE_DIR, "clustering_results.json")

def chinese_tokenizer(text):
    return jieba.lcut(text)

def visualize_issues_tsne():
    if not os.path.exists(ISSUES_FILE) or not os.path.exists(RESULTS_FILE):
        print("错误: 缺少 issues.json 或 clustering_results.json")
        return

    # 1. 加载数据
    with open(ISSUES_FILE, "r", encoding="utf-8") as f:
        issues = json.load(f)
    
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    number_to_cluster = {}
    for cluster in results:
        cid = cluster['cluster_id']
        for issue in cluster['issues']:
            number_to_cluster[issue['number']] = cid

    texts = []
    labels = []
    for issue in issues:
        texts.append(f"{issue.get('title', '')} {issue.get('body', '') or ''}")
        labels.append(number_to_cluster.get(issue['number'], -1))

    # 2. TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None, max_df=0.8, min_df=2)
    X = vectorizer.fit_transform(texts)
    
    # 3. t-SNE (非线性降维)
    # 对于文本数据，t-SNE 通常能比 PCA 更好地展现簇的边界
    print("正在进行 t-SNE 降维计算，请稍候...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X.toarray())

    # 4. 可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='Set1', alpha=0.8, edgecolors='none')
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("GitHub Issues 聚类结果可视化 (t-SNE 降维)")
    plt.xlabel("t-SNE 特征 1")
    plt.ylabel("t-SNE 特征 2")

    output_path = os.path.join(BASE_DIR, "issues_tsne.png")
    plt.savefig(output_path)
    print(f"t-SNE 可视化完成，已保存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_issues_tsne()
