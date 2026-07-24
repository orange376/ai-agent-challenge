# Day 15: 查询改写 — LLM 将模糊口语/情绪化表达改写为专业检索短语，提升召回率
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

SYSTEM_PROMPT = """你是一个查询改写助手。你的任务是把用户口语化、模糊、带有情绪的问题，改写成清晰、专业、适合用于搜索引擎或资料库检索的短语。

规则：
1. 保留用户的核心意图和问题本质
2. 补充缺失的关键专业词汇
3. 去除情绪化表达（如“太垃圾了”、“要疯了”）
4. 如果是完全不明确的短语（如“那个咋弄”），根据上下文合理推断
5. 只输出改写后的查询短语，不要加任何解释"""

def rewrite_query(original_query, temperature=0.3):
    """将模糊的用户查询改写成专业的检索短语"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请改写以下查询：{original_query}"}
            ],
            temperature=temperature,
            max_tokens=200
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten
    except Exception as e:
        print(f"查询改写失败: {e}")
        return original_query  # 降级：返回原始查询

if __name__ == "__main__":
    # 测试用例：覆盖各种模糊场景
    test_queries = [
        "那个咋弄",          # 极度模糊
        "咋又崩了",          # 情绪化+模糊
        "E4 是什么情况",     # 包含专业代码但表述口语化
        "这破玩意儿太垃圾了，我要退",  # 情绪化+退货意图
        "闪退，求解",        # 简短+技术问题
    ]
    
    print("=" * 60)
    print("查询改写测试")
    print("=" * 60)
    
    for original in test_queries:
        rewritten = rewrite_query(original)
        print(f"\n原始查询: {original}")
        print(f"改写后:   {rewritten}")
        print("-" * 40)