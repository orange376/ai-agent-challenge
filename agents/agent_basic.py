import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

# ========= 1. 初始化 LLM =========
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,           # 工具调用需要精确，温度设 0
    max_tokens=500
)

# ========= 2. 定义工具 =========
@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。输入一个数学表达式（如 '123 * 456'），返回计算结果。"""
    try:
        # 安全评估：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"

# 工具列表
tools = [calculator]

# ========= 3. 创建 Agent =========
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个有用的助手，可以使用计算器工具来解决数学问题。当用户问数学计算时，请使用 calculator 工具。",
)

# ========= 4. 测试 =========
if __name__ == "__main__":
    test_questions = [
        "123 * 456 等于多少？",
        "计算 (15 + 27) * 3 - 100",
        "10 除以 3 等于多少？"
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"❓ 用户问题：{q}")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        # 提取最后一条消息作为回答
        last_msg = result["messages"][-1]
        print(f"\n💬 最终回答：{last_msg.content}")
