# Day 11: LlamaIndex 框架对比实验 — 高封装 API vs LangChain 组件化，代码量对比
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from pypdf import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()

load_dotenv()

# ---- 1. 加载 PDF 文本（自己解析，不依赖任何额外导入）----
pdf_path = "./data/sample.pdf"
print(f"正在手动解析 PDF: {pdf_path}")
reader = PdfReader(pdf_path)
texts = []
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        texts.append(page_text)

full_text = "\n".join(texts)
print(f"提取总字符数: {len(full_text)}")

# 创建 LlamaIndex 的 Document 对象
documents = [Document(text=full_text)]
print(f"✅ 文档创建完毕，内容前200字符：{full_text[:200]}")

# ---- 2. 配置 LLM 和 Embedding ----
llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    is_chat_model=True,
    temperature=0.3,
    max_tokens=800
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)

Settings.llm = llm
Settings.embed_model = embed_model

# ---- 3. 构建索引 ----
index = VectorStoreIndex.from_documents(documents)

# ---- 4. 查询引擎 ----
query_engine = index.as_query_engine(similarity_top_k=5)

# ---- 5. 调试检索片段 ----
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("停车管理系统有哪些核心功能？")
print("\n🔍 检索到的片段：")
for i, node in enumerate(nodes, 1):
    print(f"  {i}.（相似度 {node.score:.3f}）: {node.text[:100]}...")

# ---- 6. 最终提问 ----
response = query_engine.query("停车管理系统有哪些核心功能？")
print(f"\n💬 最终回答：{response}")