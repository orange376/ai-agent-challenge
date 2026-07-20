import os
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv
from llm_api import call_llm
from query_rewriter import rewrite_query

load_dotenv()

# ========= 配置 =========
COLLECTION_NAME = "strategy_512_50"
ALPHA = 0.5  # 混合检索权重

# ========= 初始化 Embedding =========
print("正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ========= 连接 Chroma =========
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(COLLECTION_NAME)
all_docs = collection.get()['documents']

# ========= 构建 BM25 =========
def tokenize(text):
    return list(jieba.cut(text))

tokenized_docs = [tokenize(doc) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

# ========= 检索函数 =========
def vector_search(query, k=5):
    query_emb = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    docs = results['documents'][0]
    distances = results['distances'][0]
    similarities = [1 - d for d in distances]
    return list(zip(docs, similarities))

def bm25_search_local(query, k=5):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1
    return [(all_docs[i], scores[i] / max_score) for i in top_indices]

def hybrid_search(query, alpha=ALPHA, k=5):
    vec_results = vector_search(query, k=k)
    bm_results = bm25_search_local(query, k=k)
    fused = {}
    for doc, sim in vec_results:
        fused[doc] = fused.get(doc, 0) + alpha * sim
    for doc, score in bm_results:
        fused[doc] = fused.get(doc, 0) + (1 - alpha) * score
    sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]

# ========= RAG 问答接口 =========
def ask_rag(question):
    """完整的 RAG 流程：查询改写 → 混合检索 → 生成答案"""
    # 1. 查询改写
    rewritten = rewrite_query(question)
    print(f"📝 改写后查询: {rewritten}")
    
    # 2. 混合检索
    contexts = hybrid_search(rewritten, alpha=ALPHA, k=3)
    
    # 3. 生成答案
    prompt = "你是一个专业的研究助手。请根据以下提供的文档片段回答用户的问题。\n如果文档片段中没有足够的信息，请如实说明“根据文档，无法找到相关信息”，不要编造。\n\n"
    for i, ctx in enumerate(contexts, 1):
        prompt += f"【参考片段 {i}】\n{ctx}\n\n"
    prompt += f"用户问题：{question}\n回答："
    answer = call_llm(prompt)
    return answer, contexts

if __name__ == "__main__":
    # 简单交互测试
    while True:
        q = input("\n请输入问题（输入 q 退出）: ")
        if q.lower() == 'q':
            break
        answer, sources = ask_rag(q)
        print(f"\n💬 回答: {answer}")