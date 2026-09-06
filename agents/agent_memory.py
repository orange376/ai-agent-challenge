import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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

# ========= 3. 持久化对话历史 =========
# 这就是 Agent 的“记忆”
conversation_history = []

def run_agent_with_memory(user_input: str, max_steps=8):
    """带记忆的 Agent 执行器。每次调用都会保留之前的对话历史。"""
    global conversation_history
    
    # 将用户新消息追加到历史
    conversation_history.append(HumanMessage(content=user_input))
    
    llm_with_tools = llm.bind_tools(tools)
    
    for step in range(max_steps):
        # 用完整历史调用模型
        response = llm_with_tools.invoke(conversation_history)
        conversation_history.append(response)
        
        if response.content:
            print(f"💭 思考: {response.content[:80]}...")
        
        # 如果没有工具调用，返回最终答案
        if not response.tool_calls:
            return response.content
        
        # 处理工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"🔧 调用: {tool_name}({tool_args})")
            
            if tool_name in tools_map:
                tool_result = tools_map[tool_name].invoke(tool_args)
            else:
                tool_result = f"错误：未找到工具 {tool_name}"
            
            print(f"📊 返回: {tool_result}")
            conversation_history.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    
    return "达到最大步数限制，无法完成任务。"

def clear_memory():
    """清空对话记忆"""
    global conversation_history
    conversation_history = []
    print("🧹 记忆已清空")

# ========= 4. 多轮对话测试 =========
if __name__ == "__main__":
    print("=" * 60)
    print("多轮对话测试：Agent 是否能记住上下文？")
    print("=" * 60)
    
    # 第一轮
    q1 = "我要订一个 5000 块的酒店，查一下北京的天气适不适合出行"
    print(f"\n❓ 用户(第1轮): {q1}")
    ans1 = run_agent_with_memory(q1)
    print(f"💬 Agent: {ans1}")
    
    # 第二轮：关键测试——Agent 是否理解“换成便宜的”指什么？
    q2 = "太贵了，换成便宜点的"
    print(f"\n❓ 用户(第2轮): {q2}")
    ans2 = run_agent_with_memory(q2)
    print(f"💬 Agent: {ans2}")
    
    # 第三轮：Agent 是否记得上次查询了哪个城市？
    q3 = "那个城市的天气怎么样？"
    print(f"\n❓ 用户(第3轮): {q3}")
    ans3 = run_agent_with_memory(q3)
    print(f"💬 Agent: {ans3}")