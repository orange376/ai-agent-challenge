# Day 19: RAG API 服务化 — FastAPI /rag/chat 接口，返回答案+来源+置信度+改写查询
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_final import ask_rag  # 复用最优版RAG

# ------------------ 1. 日志配置 ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("rag_api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ 2. FastAPI 应用 ------------------
app = FastAPI(title="最优版RAG问答服务", version="1.0")

# 请求体模型
class ChatRequest(BaseModel):
    question: str

# 响应体模型
class SourceItem(BaseModel):
    content: str
    similarity: float = None  # 如果检索函数返回了相似度可填入，否则留空

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]        # 返回前3个文本块的前200字符摘要
    confidence: float         # 0~1 之间的置信度
    rewritten_query: str      # 查询改写后的短语（方便调试）

# ------------------ 3. 核心接口 ------------------
@app.post("/rag/chat", response_model=ChatResponse)
def rag_chat(request: ChatRequest):
    """RAG 问答接口：接收用户问题，返回答案、参考来源、置信度和改写后的查询。

    Args:
        request (ChatRequest): 包含 question 字段的请求体

    Returns:
        ChatResponse: 包含 answer（答案）、sources（参考文本块摘要）、
                      confidence（置信度 0~1）、rewritten_query（改写后查询）

    Raises:
        HTTPException 400: 问题为空
        HTTPException 500: RAG 管线内部异常
    """
    start_time = time.time()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    logger.info(f"收到问题: {question}")

    try:
        answer, contexts, rewritten = ask_rag(question)
    except Exception as e:
        logger.error(f"RAG处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail="RAG服务内部错误")

    # 构建来源列表（截取前200字符，避免太长）
    sources = [ctx[:200] + ("..." if len(ctx) > 200 else "") for ctx in contexts[:3]]

    # 置信度：目前基于是否有检索结果，后续可改为相似度均值方案
    confidence = 1.0 if contexts else 0.0

    elapsed = time.time() - start_time
    logger.info(f"完成 | 耗时: {elapsed:.2f}s | 答案长度: {len(answer)} | 改写: {rewritten}")

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        rewritten_query=rewritten
    )

# ------------------ 4. 健康检查 ------------------
@app.get("/health")
def health():
    """健康检查接口：返回服务运行状态。

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}