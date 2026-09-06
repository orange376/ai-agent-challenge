import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ========= 1. 初始化 LLM =========
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    max_tokens=500
)

# ========= 2. 定义工具 =========
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入如 '123 * 456'，返回结果。"""
    try:
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算出错：{str(e)}"

tools = [calculator]
# ToolNode 是 LangGraph 内置节点，会自动执行工具调用并生成 ToolMessage
tool_node = ToolNode(tools)

# ========= 3. 定义状态 (State) =========
class AgentState(TypedDict):
    # add_messages 是一个reducer，会自动将新消息追加到消息列表中
    messages: Annotated[list, add_messages]

# ========= 4. 定义 LLM 节点 =========
def chatbot_node(state: AgentState):
    """调用 LLM，并返回模型响应"""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    # 返回字典，LangGraph 会自动用 add_messages reducer 合并到状态中
    return {"messages": [response]}

# ========= 5. 定义条件边 (路由逻辑) =========
def should_continue(state: AgentState):
    """判断是否需要调用工具，还是直接结束"""
    last_message = state["messages"][-1]
    # 如果最后一条消息包含 tool_calls，则路由到 "tools" 节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则直接结束
    return END

# ========= 6. 构建图 (Graph) =========
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", tool_node)

# 设置入口：从 chatbot 节点开始
workflow.set_entry_point("chatbot")

# 添加条件边：chatbot -> (有工具调用？ tools : END)
workflow.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# 添加普通边：tools 节点执行完 -> 回到 chatbot 节点
workflow.add_edge("tools", "chatbot")

# 编译成可执行的应用
app = workflow.compile()

# ========= 7. 测试 =========
if __name__ == "__main__":
    print("LangGraph Agent 测试\n")
    
    # 1. 需要计算的简单问题
    initial_state = {"messages": [HumanMessage(content="帮我算一下 123 * 456 等于多少？")]}
    final_state = app.invoke(initial_state)
    print(f"💬 最终回答：{final_state['messages'][-1].content}")
    
    # 2. 不需要工具的问题
    print("\n--- 第二轮 ---")
    initial_state_2 = {"messages": [HumanMessage(content="你好，请介绍一下你自己")]}
    final_state_2 = app.invoke(initial_state_2)
    print(f"💬 最终回答：{final_state_2['messages'][-1].content}")