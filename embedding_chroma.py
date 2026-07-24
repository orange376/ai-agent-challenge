# Day 10: 文本向量化与 Chroma 向量库 — BGE embedding + 持久化存储 + 余弦相似度检索
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import os

# ==================== 1. 加载 PDF 并切分 ====================
pdf_path = "sample.pdf"
print(f"正在加载 PDF: {pdf_path}")

reader = PdfReader(pdf_path)
documents = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        documents.append(Document(page_content=text))

full_text = "".join([doc.page_content for doc in documents])
print(f"PDF 总页数: {len(documents)}, 总字符数: {len(full_text)}")

# 使用上次实验效果较好的 chunk_size=256, overlap=50
splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"分块数量: {len(chunks)}")

# ==================== 2. 初始化 Embedding 模型 ====================
print("正在加载 Embedding 模型（首次会下载，约 400MB）...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 中文效果好的轻量模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ==================== 3. 创建 Chroma 客户端并构建向量库 ====================
# 使用持久化存储，数据保存在 ./chroma_db 目录
client = chromadb.PersistentClient(path="./chroma_db")

# 获取或创建集合（类似数据库的表）
collection = client.get_or_create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 检查集合是否已有数据，避免重复插入
if collection.count() == 0:
    print("正在向量化并存入 Chroma...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)
    
    # 为每个文本块生成唯一 ID
    ids = [f"chunk_{i}" for i in range(len(texts))]
    
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=ids
    )
    print(f"已存入 {collection.count()} 条向量数据")
else:
    print(f"集合已存在 {collection.count()} 条数据，跳过插入")

# ==================== 4. 实现检索函数 ====================
def search(query, k=3):
    """输入问题，返回最相关的 k 个文本块"""
    # 将问题转为向量
    query_embedding = embedding_model.embed_query(query)
    
    # 在 Chroma 中检索
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # 解析返回结果
    if results['documents'] and results['documents'][0]:
        return results['documents'][0]  # 返回文本块列表
    return []

if __name__ == "__main__":
    # 测试检索
    test_queries = [
        "停车管理系统的研究现状？",
        "数据库设计用到了什么技术栈？",
        "报告引用了哪些文献？"
    ]
    
    for query in test_queries:
        print(f"\n🔍 问题: {query}")
        results = search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"  结果{i} (前80字): {doc[:80]}...")