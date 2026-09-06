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

# ========= 2. 定义工具（包含一个“坏掉”的工具用于测试） =========

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。输入城市名称（如'北京'），返回天气描述。"""
    weather_db = {
        "北京": "晴天，温度 25°C，湿度 40%",
        "上海": "多云，温度 28°C，湿度 70%",
        "深圳": "阵雨，温度 30°C，湿度 80%",
    }
    return weather_db.get(city, f"未找到 {city} 的天气信息")

@tool
def check_inventory(product_name: str) -> str:
    """查询指定商品的库存数量。输入商品名称，返回库存状态。"""
    inventory_db = {
        "iPhone 15": "库存充足，剩余 156 台",
        "雨伞": "库存一般，剩余 45 把，1000积分可兑换一把",
    }
    return inventory_db.get(product_name, f"未找到 {product_name} 的库存信息")

_CALCULATOR_CALL_COUNT = 0

@tool
def broken_calculator(expression: str) -> str:
    """计算数学表达式。输入表达式如 '1+1'，返回结果。"""
    global _CALCULATOR_CALL_COUNT
    _CALCULATOR_CALL_COUNT += 1
    
    if _CALCULATOR_CALL_COUNT <= 2:
        raise TimeoutError("计算器服务请求超时，请稍后重试")
    elif _CALCULATOR_CALL_COUNT == 3:
        # 第三次返回正确结果
        return f"计算结果：{eval(expression)}"
    else:
        raise RuntimeError("计算器服务内部错误")

tools = [get_weather, check_inventory, broken_calculator]

# ========= 3. 带熔断机制的 Agent 执行器 =========

def run_resilient_agent(user_input: str, max_steps=10, max_consecutive_failures=2):
    """
    带熔断机制的 Agent 执行器。
    
    - max_steps: 总执行步数上限，防止死循环
    - max_consecutive_failures: 同一工具连续失败多少次后熔断
    """
    # ----- 新增：状态跟踪 -----
    available_tools = list(tools)                    # 当前可用的工具
    tools_map = {tool.name: tool for tool in tools}  # 全量工具映射（保留原始）
    tool_failure_count = {}                          # 每个工具当前的连续失败次数
    disabled_tools = set()                           # 被熔断的工具名集合
    
    llm_with_tools = lambda msgs: llm.bind_tools(available_tools).invoke(msgs)
    messages = [HumanMessage(content=user_input)]
    step_log = []  # 记录每一步的执行情况
    
    for step in range(1, max_steps + 1):
        print(f"\n--- 第 {step} 步 ---")
        
        # 调用模型（只使用当前可用的工具）
        try:
            response = llm_with_tools(messages)
        except Exception as e:
            print(f"❌ 模型调用失败: {e}")
            step_log.append(f"步{step}: 模型调用异常 - {e}")
            break
        
        messages.append(response)
        
        # 如果模型直接返回了文本答案（没有工具调用），说明任务完成
        if not response.tool_calls:
            print(f"✅ 任务完成")
            return response.content, step_log
        
        # 逐条处理工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # 如果工具已被熔断，告诉模型不要再用它
            if tool_name in disabled_tools:
                print(f"🚫 工具 {tool_name} 已被熔断，跳过")
                messages.append(ToolMessage(
                    content=f"错误：工具 {tool_name} 已因连续失败被禁用，请使用其他可用工具完成任务。",
                    tool_call_id=tool_call["id"]
                ))
                continue
            
            # ----- 新增：执行工具并跟踪失败 -----
            print(f"🔧 调用: {tool_name}({tool_args})")
            try:
                tool_result = tools_map[tool_name].invoke(tool_args)
                # 成功：重置该工具的失败计数
                tool_failure_count[tool_name] = 0
                print(f"📊 返回: {tool_result}")
                step_log.append(f"步{step}: {tool_name} 成功")
            except Exception as e:
                # 失败：增加计数
                tool_failure_count[tool_name] = tool_failure_count.get(tool_name, 0) + 1
                fail_cnt = tool_failure_count[tool_name]
                tool_result = f"工具执行错误：{str(e)} (连续失败 {fail_cnt} 次)"
                print(f"❌ 失败: {tool_result}")
                step_log.append(f"步{step}: {tool_name} 失败 ({fail_cnt}/{max_consecutive_failures})")
                
                # ----- 新增：熔断判断 -----
                if fail_cnt >= max_consecutive_failures:
                    disabled_tools.add(tool_name)
                    # 从可用工具列表中移除
                    available_tools = [t for t in available_tools if t.name != tool_name]
                    print(f"🚨 熔断！工具 {tool_name} 已连续失败 {fail_cnt} 次，已被禁用")
                    step_log.append(f"步{step}: 🚨 熔断 {tool_name}")
            
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    
    # 达到最大步数仍未完成
    print(f"⚠️ 达到最大步数限制 ({max_steps})")
    step_log.append(f"达到最大步数 {max_steps}，强制终止")
    return "抱歉，任务达到最大执行步数限制，无法完成。请简化您的问题或稍后重试。", step_log

# ========= 4. 测试 =========
if __name__ == "__main__":
    print("=" * 60)
    print("场景：故意调用损坏的计算器，观察熔断机制")
    print("=" * 60)
    
    question = "帮我算一下 123 * 456 等于多少？如果计算器坏了，请告诉我无法计算。"
    print(f"❓ 用户问题：{question}\n")
    
    answer, log = run_resilient_agent(question, max_steps=10, max_consecutive_failures=2)
    
    print(f"\n{'='*60}")
    print(f"💬 最终回答：{answer}")
    print(f"\n📋 执行日志：")
    for entry in log:
        print(f"  {entry}")