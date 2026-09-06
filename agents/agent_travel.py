import os
import re
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ========= 1. 初始化 LLM =========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3,
    max_tokens=800
)

# ========= 2. 模拟数据 =========
ATTRACTIONS = {
    "北京": ["故宫", "长城", "颐和园", "天坛", "南锣鼓巷"],
    "上海": ["外滩", "迪士尼乐园", "东方明珠", "豫园", "田子坊"],
    "成都": ["大熊猫基地", "宽窄巷子", "锦里", "武侯祠", "青城山"],
}

WEATHER = {
    "北京": "晴朗，适宜户外活动",
    "上海": "多云，可能有小雨",
    "成都": "阴天，凉爽舒适",
}

# ========= 3. 定义状态 =========
class TravelState(TypedDict):
    messages: Annotated[list, add_messages]
    destination: str
    days: int
    budget: int
    attractions: list
    weather: str
    hotel_level: str
    itinerary: str

# ========= 4. 节点函数 =========

def intent_node(state: TravelState):
    """意图识别：从用户输入中提取目的地、天数、预算"""
    user_input = state["messages"][-1].content
    print(f"\n🔍 意图识别：{user_input}")

    # 提取目的地（支持北京/上海/成都）
    dest_match = re.search(r'(北京|上海|成都)', user_input)
    destination = dest_match.group(1) if dest_match else "北京"  # 默认北京

    # 提取天数
    days_match = re.search(r'(\d+)\s*天', user_input)
    days = int(days_match.group(1)) if days_match else 3  # 默认3天

    # 提取预算
    budget_match = re.search(r'预算\s*(\d+)', user_input) or re.search(r'(\d+)\s*元', user_input)
    budget = int(budget_match.group(1)) if budget_match else 5000  # 默认5000

    print(f"   解析结果：目的地={destination}, 天数={days}, 预算={budget}")
    return {
        "destination": destination,
        "days": days,
        "budget": budget
    }

def attractions_node(state: TravelState):
    """查景点：返回目的地热门景点"""
    dest = state["destination"]
    attractions = ATTRACTIONS.get(dest, ["景点A", "景点B", "景点C"])
    print(f"🏞️ 查询景点：{dest} → {attractions}")
    return {"attractions": attractions}

def weather_node(state: TravelState):
    """查天气：返回目的地天气"""
    dest = state["destination"]
    weather = WEATHER.get(dest, "未知")
    print(f"🌤️ 查询天气：{dest} → {weather}")
    return {"weather": weather}

def budget_check(state: TravelState):
    """条件判断：预算是否 ≥ 5000"""
    if state["budget"] >= 5000:
        return "high"
    return "low"

def high_budget_node(state: TravelState):
    """高预算：选豪华酒店"""
    print("🏨 预算充足，选择五星级豪华酒店")
    return {"hotel_level": "五星级豪华酒店"}

def low_budget_node(state: TravelState):
    """低预算：选经济型酒店"""
    print("🏨 预算有限，选择经济型连锁酒店")
    return {"hotel_level": "经济型连锁酒店"}

def itinerary_node(state: TravelState):
    """行程生成：调用 LLM 生成完整行程方案"""
    prompt = f"""请根据以下信息生成一份详细的{state['days']}天旅游行程：
目的地：{state['destination']}
天数：{state['days']} 天
预算：{state['budget']} 元
景点列表：{', '.join(state['attractions'])}
天气情况：{state['weather']}
酒店档次：{state['hotel_level']}

行程应包括每天的具体安排、推荐景点、餐饮建议，并控制在预算内。请用 Markdown 格式输出。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    itinerary = response.content
    print(f"📝 行程已生成（长度 {len(itinerary)} 字符）")
    return {"itinerary": itinerary}

def human_confirmation_node(state: TravelState):
    """人工确认：展示行程并询问用户是否确认"""
    print("\n" + "="*60)
    print("请确认以下行程：")
    print(state["itinerary"])
    print("="*60)
    
    user_input = input("是否确认？(输入 y 确认，其他任意键取消): ").strip().lower()
    if user_input == "y":
        print("✅ 用户已确认行程")
        return {"messages": [AIMessage(content="行程已确认，祝您旅途愉快！")]}
    else:
        print("❌ 用户取消行程")
        return {"messages": [AIMessage(content="行程已取消。如有需要，请重新规划。")]}

# ========= 5. 构建图 =========
workflow = StateGraph(TravelState)

# 添加节点
workflow.add_node("intent", intent_node)
workflow.add_node("attractions", attractions_node)
workflow.add_node("weather", weather_node)
workflow.add_node("high_budget", high_budget_node)
workflow.add_node("low_budget", low_budget_node)
workflow.add_node("itinerary", itinerary_node)
workflow.add_node("human_confirm", human_confirmation_node)

# 入口
workflow.set_entry_point("intent")

# 普通边
workflow.add_edge("intent", "attractions")
workflow.add_edge("attractions", "weather")

# 条件边：weather → budget_check → high/low
workflow.add_conditional_edges(
    "weather",
    budget_check,
    {
        "high": "high_budget",
        "low": "low_budget"
    }
)

# 酒店节点都通向行程生成
workflow.add_edge("high_budget", "itinerary")
workflow.add_edge("low_budget", "itinerary")

# 行程生成 → 人工确认
workflow.add_edge("itinerary", "human_confirm")

# 人工确认 → 结束
workflow.add_edge("human_confirm", END)

# 编译
app = workflow.compile()

# ========= 6. 测试 =========
if __name__ == "__main__":
    print("="*60)
    print("🏖️ LangGraph 旅游规划 Agent 测试")
    print("="*60)
    
    # 测试用例
    test_input_1 = "我想去北京旅游，3天，预算3000元"
    initial_state_1 = {"messages": [HumanMessage(content=test_input_1)]}
    final_state_1 = app.invoke(initial_state_1)
    print("\n" + "="*60)
    print("最终输出：")
    print(final_state_1["messages"][-1].content)
    
    # 测试用例2：低预算
    # 注意：由于人工确认节点需要输入，如果不想交互可注释掉下面这段
    # 若要测试低预算分支，请复制上面代码并修改输入为预算3000元，同时取消下面注释
    # initial_state_2 = {"messages": [HumanMessage(content="成都3天，预算3000")]}
    # final_state_2 = app.invoke(initial_state_2)
    # print(final_state_2["messages"][-1].content)