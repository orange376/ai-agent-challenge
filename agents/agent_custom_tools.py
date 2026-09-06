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

# ========= 2. 定义三个自定义工具 =========

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。输入城市名称（如'北京'），返回天气描述。"""
    # 模拟天气数据库
    weather_db = {
        "北京": "晴天，温度 25°C，湿度 40%，适合户外活动",
        "上海": "多云，温度 28°C，湿度 70%，可能有小雨",
        "深圳": "阵雨，温度 30°C，湿度 80%，闷热",
        "成都": "小雨，温度 22°C，湿度 65%，注意带伞"
    }
    return weather_db.get(city, f"未找到 {city} 的天气信息，请确认城市名称")

@tool
def check_inventory(product_name: str) -> str:
    """查询指定商品的库存数量。输入商品名称（如'iPhone 15'），返回库存状态。"""
    inventory_db = {
        "iPhone 15": "库存充足，剩余 156 台，支持当日发货",
        "MacBook Pro": "库存紧张，仅剩 8 台，建议尽快下单",
        "AirPods Pro": "暂时缺货，预计下周补货",
        "充电器": "库存充足，剩余 500+ 个",
        "雨伞": "库存一般，剩余 45 把，1000积分可兑换一把"
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

# ========= 3. 手写 Agent 循环（与 Day 28 一致） =========
def run_agent(user_input: str, max_steps=8):
    llm_with_tools = llm.bind_tools(tools)
    messages = [HumanMessage(content=user_input)]
    
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # ----- 新增：输出模型的完整思考 -----
        if response.content:
            print(f"💭 Thought: {response.content}")
        
        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"🔧 Action: {tool_name}({tool_args})")
            
            if tool_name in tools_map:
                tool_result = tools_map[tool_name].invoke(tool_args)
            else:
                tool_result = f"错误：未找到工具 {tool_name}"
            
            print(f"📊 Observation: {tool_result}")
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    
    return "达到最大步数限制。"

# ========= 4. 测试 =========
if __name__ == "__main__":
    test_questions = [
        "查一下北京的天气怎么样？",
        "看看 iPhone 15 还有货吗？",
        "帮我查一下用户 U1001 的信息",
        "上海天气如何？顺便看看 MacBook Pro 的库存",
        "深圳天气和用户 U1002 的信息一起查一下",
        "深圳明天会不会下雨？如果有雨，我想查一下我的会员积分够不够买一把雨伞，用户ID是U1003"
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ 用户问题：{q}")
        answer = run_agent(q)
        print(f"\n💬 最终回答：{answer}")