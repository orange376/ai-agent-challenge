# Day 17: 批量构建 Chroma 索引 — 三种切片策略 (256/512/1024) 独立集合，供评估切换
import sys
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

def build_index(chunk_size, chunk_overlap, collection_name):
    # 加载 PDF
    pdf_path = "sample.pdf"
    reader = PdfReader(pdf_path)
    documents = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            documents.append(Document(page_content=text))

    # 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    print(f"chunk_size={chunk_size}, overlap={chunk_overlap} → {len(chunks)} 块")

    # Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    embeddings = embedding_model.embed_documents(texts)

    # 存入指定集合
    client = chromadb.PersistentClient(path="./chroma_db")
    # 如果集合已存在，先删除再重建（确保数据是新的）
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(collection_name)
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=[f"chunk_{i}" for i in range(len(texts))]
    )
    print(f"集合 '{collection_name}' 构建完成，共 {collection.count()} 条数据\n")

if __name__ == "__main__":
    # 构建三种策略的索引
    build_index(256, 30, "strategy_256_30")
    build_index(512, 50, "strategy_512_50")
    build_index(1024, 100, "strategy_1024_100")