from llm_api import call_llm
from .tools import get_weather, get_attractions
import re

REACT_PROMPT   = """你是一个智能助手，可以使用以下工具来获取真实信息：

- get_weather(city): 查询指定城市的天气，返回天气描述。
- get_attractions(city): 查询指定城市的景点，返回景点列表（逗号分隔）。

你需要按以下格式逐步解决问题，每次只输出 Thought 和 Action，**绝对不要自己编造 Observation**。
当你输出 Action 后，系统会自动执行工具并将结果作为 Observation 返回，然后你再继续。

格式：
Thought: <你的思考过程>
Action: <工具名>(<参数>)

=== 示例 ===
用户问题：河北有什么好玩的？天气如何？

Thought: 我需要先查询河北的景点，再查天气。
Action: get_attractions("河北")

（此时系统会返回 Observation，然后你继续）

Thought: 现在我需要查询天气。
Action: get_weather("河北")

（系统再次返回 Observation）

Thought: 我已经获得了景点和天气信息，可以回答用户了。
Final Answer: 河北的景点有...，天气是...，适合...

=== 真实对话开始 ===
用户问题：{question}
"""

def execute_action(action_str):
    """解析 Action 字符串并调用真实工具，返回结果字符串"""
    match = re.match(r'(\w+)\((.*)\)', action_str.strip())
    if not match:
        return "错误：无法解析 Action，格式应为 工具名(参数)，例如 get_weather(北京)"
    
    tool_name, arg = match.group(1), match.group(2).strip().strip('"').strip("'")
    
    if tool_name == "get_weather":
        return get_weather(arg)
    elif tool_name == "get_attractions":
        attractions = get_attractions(arg)
        if isinstance(attractions, list):
            return ", ".join(attractions) if attractions else "未找到景点"
        return str(attractions)
    else:
        return f"错误：未知工具 '{tool_name}'"

def run_react_agent(question, max_steps=10):
    history = REACT_PROMPT.format(question=question)
    
    for step in range(1, max_steps + 1):
        print(f"\n{'='*20} 第 {step} 步 {'='*20}")
        
        # 关键修改：使用 stop 参数让模型在输出 "Observation:" 前停止
        response = call_llm(history, system_prompt="", stop=["Observation:"])
        if not response:
            print("模型调用失败，终止。")
            break
        
        # 去掉末尾可能残留的换行符
        response = response.rstrip()
        print(f"模型输出:\n{response}")
        history += "\n" + response
        
        # 检查是否生成了 Final Answer
        if "Final Answer:" in response:
            print("\n✅ 任务完成，最终答案已生成。")
            break
        
        # 提取 Action
        action_match = re.search(r'Action:\s*(.+)', response)
        if not action_match:
            print("⚠️ 模型未输出 Action，尝试引导...")
            history += "\nObservation: 请按照格式输出 Action。"
            continue
        
        action_str = action_match.group(1).strip()
        print(f"🔧 执行工具调用: {action_str}")
        observation = execute_action(action_str)
        print(f"📊 工具返回: {observation}")
        
        # 将 Observation 追加到对话历史
        history += f"\nObservation: {observation}"
        
    else:
        print("⚠️ 达到最大步数限制，强制退出。")

if __name__ == "__main__":
    question = "北京，上海有哪些好玩的地方？那边的天气适合户外活动吗？"
    run_react_agent(question)