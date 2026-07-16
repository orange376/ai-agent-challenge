from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 指定 PDF 路径（请替换为你自己的文件路径）
pdf_path = "sample.pdf"

# 2. 加载 PDF
df_path = "sample.pdf"
print(f"正在加载 PDF: {pdf_path}")

reader = PdfReader(pdf_path)
documents = []
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text
        documents.append(Document(page_content=text))

print(f"PDF 总页数: {len(documents)}")
print(f"总字符数: {len(full_text)}\n")

# 3. 定义三种不同的分块策略
chunk_sizes = [256, 512, 1024]
chunk_overlap = 50  # 块与块之间重叠 50 个字符，保持上下文连贯

for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]  # 中文友好的分割符
    )
    chunks = splitter.split_documents(documents)
    print(f"chunk_size={size}, chunk_overlap={chunk_overlap} → 分块数量: {len(chunks)}")
    
    # 可选：打印第一个块的前100字符，观察切分效果
    if chunks:
        print(f"  第一个块预览: {chunks[0].page_content[:100]}...\n")