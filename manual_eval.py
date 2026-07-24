# Day 19: RAG 手动评估 — Faithfulness（忠实度）+ Answer Relevancy（切题度）双指标 LLM 评判
import json
import os
from dotenv import load_dotenv
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from llm_api import call_llm
from query_rewriter import rewrite_query

load_dotenv()

# ========= 可切换的集合名 =========
COLLECTION_NAME = "strategy_512_50"  # 改成"strategy_256_30" "strategy_512_50" 或 "strategy_1024_100"
RETRIEVAL_MODE = "hybrid" 

# ========= 初始化 Embedding =========
print("正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ========= 连接 Chroma 并获取文档列表 =========
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(COLLECTION_NAME)
all_docs = collection.get()['documents']

# 构建 BM25
def tokenize(text):
    return list(jieba.cut(text))

tokenized_docs = [tokenize(doc) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

# ========= 检索函数（复用混合检索逻辑） =========
def vector_search(query, k=5):
    """向量检索：余弦相似度 Top-K。

    Args:
        query: 查询字符串
        k: 返回数量

    Returns:
        list[tuple[str, float]]: [(文本块, 相似度), ...]
    """
    query_emb = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    docs = results['documents'][0]
    distances = results['distances'][0]
    similarities = [1 - d for d in distances]
    return list(zip(docs, similarities))

def bm25_search_local(query, k=5):
    """BM25 关键词检索：jieba 分词 + 得分归一化。

    Args:
        query: 查询字符串
        k: 返回数量

    Returns:
        list[tuple[str, float]]: [(文本块, 归一化得分), ...]
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1
    return [(all_docs[i], scores[i] / max_score) for i in top_indices]

def hybrid_search_local(query, alpha=0.5, k=5):
    """混合检索：向量 + BM25 加权融合，返回 Top-K 文本块。

    Args:
        query: 查询字符串
        alpha: 向量权重 (0~1)，默认 0.5
        k: 返回数量

    Returns:
        list[str]: 融合后的 Top-K 文本块
    """
    vec_results = vector_search(query, k=k)
    bm_results = bm25_search_local(query, k=k)
    fused = {}
    for doc, sim in vec_results:
        fused[doc] = fused.get(doc, 0) + alpha * sim
    for doc, score in bm_results:
        fused[doc] = fused.get(doc, 0) + (1 - alpha) * score
    sorted_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]

# ------------------ 1. 评估逻辑 ------------------
def evaluate_faithfulness(answer, contexts):
    """评估答案忠实度：判断 answer 中的每一条陈述是否都能在 contexts 中找到依据。

    通过 LLM 评判，逐条核对回答中的信息是否在参考上下文中可验证。
    若回答诚实声明"无法找到相关信息"，且上下文确实不包含答案，也算忠实。

    Args:
        answer (str): LLM 生成的回答
        contexts (list[str]): 检索到的参考文本块列表

    Returns:
        float: 忠实度分数，范围 0.0 ~ 1.0（解析失败返回 0.5）
    """
    prompt = f"""你是一个严格的评估助手。请判断以下“回答”中的每一条信息，是否都能在“参考上下文”中找到依据。
如果回答中的信息在上下文中找不到，就说“无法验证”。如果回答说“根据文档无法找到相关信息”但上下文确实没有，也算忠实。

参考上下文：
{chr(10).join([f"片段{i+1}: {ctx[:500]}" for i, ctx in enumerate(contexts)])}

回答：
{answer}

请只输出一个 JSON，格式为：
{{"score": 0.0~1.0, "reason": "简要说明扣分原因"}}
"""
    result = call_llm(prompt)
    try:
        # 解析返回的 JSON
        result = result.strip().replace("```json", "").replace("```", "")
        data = json.loads(result)
        return data.get("score", 0.0)
    except:
        return 0.5  # 解析失败给默认值

def evaluate_relevancy(question, answer):
    """评估答案切题度：判断 answer 是否直接、切题地回答了 question。

    检测回答是否跑题、答非所问或包含无关内容。

    Args:
        question (str): 用户原始问题
        answer (str): LLM 生成的回答

    Returns:
        float: 切题度分数，范围 0.0 ~ 1.0（解析失败返回 0.5）
    """
    prompt = f"""你是一个评估助手。请判断以下回答是否直接、切题地回答了用户的问题。
如果回答跑题、答非所问，或者包含无关内容，请扣分。

用户问题：{question}
回答：{answer}

请只输出一个 JSON，格式为：
{{"score": 0.0~1.0, "reason": "简要说明"}}
"""
    result = call_llm(prompt)
    try:
        result = result.strip().replace("```json", "").replace("```", "")
        data = json.loads(result)
        return data.get("score", 0.0)
    except:
        return 0.5

# ------------------ 2. 加载测试集 ------------------
with open("testset_v2.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# ------------------ 3. 运行评估 ------------------
# ------------------ 3. 运行评估 ------------------
print("正在手动评估 RAG 系统...")
faith_scores = []
relev_scores = []

for item in test_data:
    question = item["question"]
    ground_truth = item["ground_truth"]

    # ---- 整合所有优化：查询改写 + 混合检索 + 生成 ----
    # 1. 查询改写
    rewritten = rewrite_query(question)
    print(f"📝 改写后: {rewritten}")
    
    # 2. 混合检索（使用我们已经定义好的 hybrid_search_local 函数）
    contexts = hybrid_search_local(rewritten, alpha=0.5, k=3)
    
    # 3. 构建 prompt 并生成答案
    prompt = "你是一个专业的研究助手。请根据以下提供的文档片段回答用户的问题。\n如果文档片段中没有足够的信息，请如实说明“根据文档，无法找到相关信息”，不要编造。\n\n"
    for i, ctx in enumerate(contexts, 1):
        prompt += f"【参考片段 {i}】\n{ctx}\n\n"
    prompt += f"用户问题：{question}\n回答："
    
    try:
        answer = call_llm(prompt)
    except Exception as e:
        print(f"生成回答失败: {e}")
        answer = "根据文档，无法找到相关信息。"

    # 计算指标
    faith = evaluate_faithfulness(answer, contexts)
    relev = evaluate_relevancy(question, answer)
    faith_scores.append(faith)
    relev_scores.append(relev)
    print(f"Q: {question[:30]}... | Faith: {faith:.2f} | Relev: {relev:.2f}")

# ------------------ 4. 输出结果 ------------------
print("\n===== 评估结果（手动实现） =====")
print(f"平均 Faithfulness: {sum(faith_scores)/len(faith_scores):.3f}")
print(f"平均 Answer Relevancy: {sum(relev_scores)/len(relev_scores):.3f}")

# 保存结果
with open("manual_eval_results.json", "w", encoding="utf-8") as f:
    json.dump({"faithfulness": faith_scores, "relevancy": relev_scores}, f, ensure_ascii=False, indent=2)
print("结果已保存到 manual_eval_results.json")