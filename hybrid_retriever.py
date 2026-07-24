# Day 14: 混合检索 — 向量检索 + BM25 加权融合 (α=0.5)，互补语义与关键词匹配
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from query_rewriter import rewrite_query

# ==================== 1. 加载 Chroma 向量库（复用 Day 10） ====================
print("正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_chunks")
print(f"Chroma 集合中有 {collection.count()} 条数据")

# 获取所有文本块（用于构建 BM25）
all_docs = collection.get()['documents']  # 列表，顺序与向量一一对应

# ==================== 2. 构建 BM25 检索器 ====================
def tokenize(text):
    return list(jieba.cut(text))

tokenized_docs = [tokenize(doc) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

# ==================== 3. 向量检索函数 ====================
def vector_search(query, k=5):
    """返回 [(文本块, 相似度), ...]，相似度已归一化到 [0, 1]"""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    docs = results['documents'][0]
    distances = results['distances'][0]
    # Chroma 返回的是余弦距离，转换为相似度：similarity = 1 - distance
    similarities = [1 - d for d in distances]
    return list(zip(docs, similarities))

# ==================== 4. BM25 检索函数 ====================
def bm25_search(query, k=5):
    """返回 [(文本块, 得分), ...]，得分已归一化到 [0, 1]"""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    # 取 Top-K 索引
    top_indices = np.argsort(scores)[::-1][:k]
    # 归一化：除以当前最大值（如果最大值>0）
    max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1
    return [(all_docs[i], scores[i] / max_score) for i in top_indices]

# ==================== 5. 混合检索（加权平均融合） ====================
def hybrid_search(query, alpha=0.5, k=5):
    """
    alpha: 向量检索权重 (0~1)，1=纯向量，0=纯BM25
    返回融合后的 Top-K 文本块
    """
    # 分别获取两路结果
    vector_results = vector_search(query, k=k)
    bm25_results = bm25_search(query, k=k)
    
    # 构建 文档 → 融合分数 的映射
    fused_scores = {}
    
    # 加权向量结果
    for doc, sim in vector_results:
        fused_scores[doc] = fused_scores.get(doc, 0) + alpha * sim
    
    # 加权 BM25 结果
    for doc, score in bm25_results:
        fused_scores[doc] = fused_scores.get(doc, 0) + (1 - alpha) * score
    
    # 按融合分数降序排序
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 返回 Top-K 文本块
    return [doc for doc, score in sorted_docs[:k]]

# ==================== 6. 对比测试 ====================
if __name__ == "__main__":
    test_queries = [
        "停车管理系统有哪些核心功能？",
        "Oracle 数据库设计中采用了什么技术？"
    ]
    
    for query in test_queries:
        rewritten = rewrite_query(query)
        print(f"📝 改写后查询: {rewritten}")
    
        # 然后所有检索都用 rewritten 而不是 query
        ec_results = vector_search(rewritten, k=3)
        bm_results = bm25_search(rewritten, k=3)
        hybrid_results = hybrid_search(rewritten, alpha=0.5, k=3)
        
        # 纯向量
        vec_results = vector_search(rewritten, k=3)
        print("\n📌 纯向量检索 Top-3:")
        for i, (doc, sim) in enumerate(vec_results, 1):
            print(f"  {i}. (相似度 {sim:.3f}) {doc[:80]}...")
        
        # 纯 BM25
        bm_results = bm25_search(rewritten, k=3)
        print("\n📌 纯 BM25 检索 Top-3:")
        for i, (doc, score) in enumerate(bm_results, 1):
            print(f"  {i}. (得分 {score:.3f}) {doc[:80]}...")
        
        # 混合检索 (alpha=0.5)
        hybrid_results = hybrid_search(rewritten, alpha=0.5, k=3)
        print(f"\n📌 混合检索 (α=0.5) Top-3:")
        for i, doc in enumerate(hybrid_results, 1):
            print(f"  {i}. {doc[:80]}...")