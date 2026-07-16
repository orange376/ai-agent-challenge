from fastapi import FastAPI
from pydantic import BaseModel
from llm_api import call_llm
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="我的第一个聊天接口")

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):   ##接受用户问题，调用大模型返回答案
    result = call_llm(request.question)  ##调用封装的函数
    if result is None:  ##调用失败则返回信息
        return ChatResponse(answer="抱歉，暂时无法获取回答，请稍后再试。")
    return ChatResponse(answer=result)