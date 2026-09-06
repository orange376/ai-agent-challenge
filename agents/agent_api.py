import os
import time
import threading
from typing import Optional, List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

# ========= 1. 初始化 LLM =========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    max_tokens=500
)

# ========= 2. 定义工具（与 Day 29 相同） =========
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。输入城市名称（如'北京'），返回天气描述。"""
    weather_db = {
        "北京": "晴天，温度 25°C，湿度 40%，适合户外活动",
        "上海": "多云，温度 28°C，湿度 70%，可能有小雨",
        "深圳": "阵雨，温度 30°C，湿度 80%，闷热",
        "成都": "小雨，温度 22°C，湿度 65%，注意带伞"
    }
    return weather_db.get(city, f"未找到 {city} 的天气信息，请确认城市名称")

@tool
def check_inventory(product_name: str) -> str:
    """查询指定商品的库存数量。输入商品名称，返回库存状态。"""
    inventory_db = {
        "iPhone 15": "库存充足，剩余 156 台，支持当日发货",
        "MacBook Pro": "库存紧张，仅剩 8 台，建议尽快下单",
        "AirPods Pro": "暂时缺货，预计下周补货",
        "充电器": "库存充足，剩余 500+ 个"
    }
    return inventory_db.get(product_name, f"未找到 {product_name} 的库存信息，请检查商品名称")

@tool
def get_user_info(user_id: str) -> str:
    """根据用户ID查询用户的基本信息。输入用户ID（如'U1001'），返回姓名、会员等级和积分。"""
    user_db = {
        "U1001": "姓名：张三，会员等级：金牌会员，积分：8,500，注册时间：2022年3月",
        "U1002": "姓名：李四，会员等级：银牌会员，积分：1,200，注册时间：2023年11月",
        "U1003": "姓名：王五，会员等级：普通用户，积分：300，注册时间：2024年6月"
    }
    return user_db.get(user_id, f"未找到用户ID {user_id} 的信息")

tools = [get_weather, check_inventory, get_user_info]
tools_map = {tool.name: tool for tool in tools}

# ========= 3. 会话管理（并发安全） =========
# 全局会话字典：session_id -> messages列表
session_histories: Dict[str, list] = {}
# 线程锁，保护字典的并发读写
lock = threading.Lock()

# ========= 4. Agent 执行函数（带日志） =========
def run_agent_with_session(session_id: str, user_input: str, return_log: bool = False):
    """
    执行 Agent，返回 (answer, log_list)
    - 如果 session_id 不存在，创建新会话；否则继续已有历史
    - 会话历史被更新，以支持多轮对话
    """
    with lock:
        # 获取或创建会话历史
        if session_id not in session_histories:
            session_histories[session_id] = []
        history = session_histories[session_id]

    # 将用户消息追加到历史
    history.append(HumanMessage(content=user_input))

    # 执行循环
    log = []
    llm_with_tools = llm.bind_tools(tools)
    max_steps = 8

    for step in range(max_steps):
        response = llm_with_tools.invoke(history)
        history.append(response)

        # 记录思考（如果有内容）
        if response.content and return_log:
            log.append(f"Thought: {response.content[:100]}")

        # 如果没有工具调用，结束
        if not response.tool_calls:
            answer = response.content
            break

        # 处理工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if return_log:
                log.append(f"Action: {tool_name}({tool_args})")

            if tool_name in tools_map:
                try:
                    tool_result = tools_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"工具执行错误: {str(e)}"
            else:
                tool_result = f"错误：未找到工具 {tool_name}"

            if return_log:
                log.append(f"Observation: {tool_result}")

            history.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    else:
        answer = "达到最大步数限制，无法完成任务。"

    return answer, log

# ========= 5. FastAPI 应用 =========
app = FastAPI(title="多会话 Agent 服务", version="1.0")

class ChatRequest(BaseModel):
    session_id: str
    question: str
    return_log: bool = False

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    log: Optional[List[str]] = None

@app.post("/agent/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    start = time.time()
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    # 执行 Agent
    answer, log = run_agent_with_session(req.session_id, req.question, req.return_log)
    elapsed = time.time() - start
    print(f"[{req.session_id}] 问题: {req.question[:30]}... 耗时: {elapsed:.2f}s")

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        log=log if req.return_log else None
    )

@app.get("/sessions")
def list_sessions():
    with lock:
        return {"active_sessions": list(session_histories.keys())}