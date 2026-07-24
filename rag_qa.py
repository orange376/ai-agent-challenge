# Day 11: 初版 RAG 问答 — Chroma 检索 → 拼接上下文 Prompt → DeepSeek 生成答案
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# 加载环境变量
load_dotenv()

# ==================== 1. 初始化生成模型（DeepSeek） ====================
llm = ChatOpenAI(
    model="deepseek-v4-flash",       
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3,                # RAG 需要稳定，不宜太高
    max_tokens=800
)

# ==================== 2. 加载同一个 Embedding 模型 ====================
print("正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ==================== 3. 连接已有的 Chroma 集合 ====================
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("pdf_chunks")
print(f"已连接 Chroma，集合中有 {collection.count()} 条数据")

# ==================== 4. 定义 RAG 问答函数 ====================
def ask_question(query, k=3):
    """
    1. 将用户问题向量化
    2. 从 Chroma 检索 k 个最相关的文本块
    3. 将检索结果拼接成上下文
    4. 构造 Prompt 发给 LLM 生成答案
    """
    # 检索
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    docs = results['documents'][0]  # k 个文本块
    
    # 拼接上下文
    context = "\n\n".join([f"【参考片段 {i+1}】\n{doc}" for i, doc in enumerate(docs)])
    
    # 构造 Prompt
    prompt = f"""你是一个专业的研究助手。请根据以下提供的文档片段回答用户的问题。
如果文档片段中没有足够的信息，请如实说明“根据文档，无法找到相关信息”，不要编造。

{context}

用户问题：{query}
回答："""
    
    # 调用 LLM 生成答案（这里直接传入 prompt 作为用户消息）
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, docs

# ==================== 5. 交互式问答测试 ====================
if __name__ == "__main__":
    test_questions = [
        "停车管理系统有哪些核心功能？",
        "数据库设计中采用了什么技术？",
        "论文提到了哪些国内外研究成果？",
        "系统可行性如何？" ,
        "系统用什么编程语言开发的？"   # 可能不在文档中，测试“不编造”能力
    ]
    
    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"❓ 用户问题：{q}")
        answer, sources = ask_question(q, k=3)
        print(f"\n💬 回答：\n{answer}")
        print(f"\n📚 参考片段数量：{len(sources)}")