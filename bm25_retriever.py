from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
import jieba

# ==================== 1. 加载并切分 PDF（与 pdf_splitter.py 一致） ====================
pdf_path = "sample.pdf"
reader = PdfReader(pdf_path)
documents = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        documents.append(Document(page_content=text))

full_text = "".join([doc.page_content for doc in documents])
print(f"总字符数: {len(full_text)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]
print(f"分块数量: {len(texts)}")

# ==================== 2. 构建 BM25 检索器 ====================
# 对中文文本进行简易分词（按字符级切分，或使用 jieba 分词）
# 这里先按单字符切分，后续可升级为 jieba 分词

def tokenize(text):
    return list(jieba.cut(text))

tokenized_texts = [tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized_texts)

def bm25_search(query, k=3):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    # 取得分最高的 k 个索引
    top_indices = np.argsort(scores)[::-1][:k]
    return [(texts[i], scores[i]) for i in top_indices]

# ==================== 3. 对比测试 ====================

# 重新构建 BM25
tokenized_test = [tokenize(t) for t in texts]
bm25_test = BM25Okapi(tokenized_test)

def bm25_test_search(query, k=3):
    tokenized_query = tokenize(query)
    scores = bm25_test.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(texts[i], scores[i]) for i in top_indices]

# 从 PDF 中找两个真实存在的、相近但不同的查询词
print("\n===== 测试1：查询 '停车管理' =====")
results1 = bm25_test_search("停车管理", k=2)
for i, (text, score) in enumerate(results1, 1):
    print(f"结果{i} (得分{score:.2f}): {text[:200]}...")

print("\n===== 测试2：查询 '停车管理系统' =====")
results2 = bm25_test_search("停车管理系统", k=2)
for i, (text, score) in enumerate(results2, 1):
    print(f"结果{i} (得分{score:.2f}): {text[:200]}...")

print("\n===== 测试3：查询 '数据库' =====")
results3 = bm25_test_search("数据库", k=2)
for i, (text, score) in enumerate(results3, 1):
    print(f"结果{i} (得分{score:.2f}): {text[:200]}...")

print("\n===== 测试4：查询 'Oracle 数据库设计' =====")
results4 = bm25_test_search("Oracle 数据库设计", k=2)
for i, (text, score) in enumerate(results4, 1):
    print(f"结果{i} (得分{score:.2f}): {text[:200]}...")