# Day 6: 意图分类对比实验 — Zero-shot vs Few-shot vs CoT，CoT 模糊意图准确率领先 30%
from llm_api import call_llm

ZERO_SHOT_PROMPT = """你是一个客服意图分类助手。请将用户消息归类为以下之一：
退货/退款、技术故障、物流查询、投诉客服、产品咨询
只输出类别名称，不要解释。"""


FEW_SHOT_PROMPT = """你是一个客服意图分类助手。请将用户消息归类为以下之一：
退货/退款、技术故障、物流查询、投诉客服、产品咨询
只输出类别名称，不要解释。

示例：
用户：买的手机屏幕有坏点，我要退货
类别：退货/退款

用户：app一直闪退，卸载重装也没用
类别：技术故障

用户：快递显示签收了但我没收到
类别：物流查询"""


COT_PROMPT = """你是一个客服意图分类助手。请将用户消息归类为以下之一：
退货/退款、技术故障、物流查询、投诉客服、产品咨询
在给出最终类别前，请先用一两句话分析用户消息中的关键信息，然后输出类别。
输出格式：
分析：<你的推理>
类别：<最终类别>"""


def classify_intent(user_message, prompt_type="zero_shot"):
    """根据用户消息进行意图分类"""
    if prompt_type == "zero_shot":
        system_prompt = ZERO_SHOT_PROMPT
    elif prompt_type == "few_shot":
        system_prompt = FEW_SHOT_PROMPT
    elif prompt_type == "cot":
        system_prompt = COT_PROMPT
    else:
        raise ValueError("无效的 prompt_type，请选择 'zero_shot', 'few_shot' 或 'cot'。")

    user_prompt = f"用户消息：\n{user_message}"
    return call_llm(user_prompt, system_prompt=system_prompt)


if __name__ == "__main__":
    TEST_CASES = [
        "手机充电口接触不良，换了好几根线都不行，你们这什么品控啊？",
        "我的快递昨天就显示派送了，今天还没到，能帮我查一下吗？",
        "你好，想问问你们家的加湿器冬天北方够用吗？",
        "我要退货，这个耳机戴久了耳朵疼，还不如我9块9买的。",
        "我刚打了三个电话都没人接，app上找客服也是机器人，你们公司还有人吗？",
        "这破手机又自动关机了，修过一次还是这样，我看你们是根本修不好",
        "你们说的30天无忧退，结果专员上门各种刁难，你们这政策到底真的假的？",
        "app里的物流信息三天没更新了，你们的系统是不是又崩了？",
        "这加湿器静音效果真‘棒’，跟拖拉机一样，有没有办法让它闭嘴？",
        "你们的客服只会让我重启、重装，我都试了五遍了还是闪退，能不能来个懂技术的人？"
    ]

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n===== 测试用例 {i} =====\n用户消息：{case}\n")
        for strategy in ("zero_shot", "few_shot", "cot"):
            print(f"--- {strategy.upper()} ---")
            result = classify_intent(case, prompt_type=strategy)
            print(f"结果：{result}\n")