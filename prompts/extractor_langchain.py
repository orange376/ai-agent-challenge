# Day 8: LangChain 版信息抽取 — LCEL 链式调用重写 Day 5 的 extractor
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 加载 .env 中的 API Key
load_dotenv()
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.5,

    max_tokens=500
)

#使用 ChatPromptTemplate 明确区分系统指令和用户输入
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的客服信息抽取助手。你的任务是从用户投诉文本中提取关键信息。
你必须严格遵守以下规则：
1. 只提取文本中明确提到的信息，不要推测
2. 如果某个字段没有提到，值设为 null
3. 必须严格输出 JSON 格式，不要包含任何其他解释文字
4. 强化“质量缺陷”与“用户诉求”的分离（退货只是诉求，应同时标注真实问题类型）
5. product字段仅限购买的商品名/服务名
6. 感谢，感激，表扬等正面反馈的问题类型归为其他
7. 简短文本只做结构化抽取，不要进行情绪分析或其他推测
8. 多问题复合场景下对于紧急程度的推断应谨慎，紧急程度不能为null
9. 当出现反问、感叹、投诉意图时sentiment优先判为愤怒
10.当用户明确表达退货/换新/咨询时，必须并列标注相应 issue_type

输出 JSON 示例：
{{
    "product": ["手机"], 
    "issue_type": ["质量问题", "退货"], 
    "sentiment": "愤怒", "urgency": "高", 
    "keywords": ["屏幕坏了", "退货退款"]
}}
"""
    ),
    ("human", "{text}")
])

#构建链
chain = prompt | llm

def extract_complaint_langchain(text):
    """使用 LangChain 进行信息抽取"""
    result = chain.invoke({"text": text})
    return result.content

if __name__ == "__main__":
    test_cases = [
        "我在你们平台买的手机才用了三天屏幕就坏了，要求立刻退货退款！太让人生气了！",
        "上周买的衣服收到了，颜色和图片有一点色差，不过整体还行，懒得换了。",
        "太失望了，以后再也不买你们的东西了。",
        "退货。",
        "必须表扬一下客服小李，帮我快速处理了退款，态度非常好，退款三分钟就到账了，很满意！",
        "快递员没经过同意直接把包裹放在了驿站，驿站离我家很远，而且包裹外包装有破损，我要投诉这个配送员并要求检查商品是否损坏。",
        "家里有小孩，这个电暖器开了一会外壳就很烫，孩子如果摸到肯定要烫伤，这设计太危险了，我要退货！",
        "可能是我操作不对吧，这个加湿器按照说明书操作还是不出雾，试了好几次都不行，不确定是不是坏了。",
        "如果我买了超过7天但是没超过15天，商品没有使用过包装完好，还能退货吗？"
        "工号28561的客服极其敷衍，我跟她说手机发热严重，她只会回复让我重启试试，这难道不是硬件问题吗？要求投诉这个客服。",
        "买的坚果打开后发现里面有虫子，已经拍了照片和视频，这种食品安全问题你们打算怎么解决？",
        "充值50元话费已经扣款了，但一直没有到账，打了很多次客服都说在核实中，到底要等多久？",
        "之前买的扫地机器人返修过一次，现在又坏了，而且这次充电器也找不到了。我不想再修了，能不能直接换新或者退款？",
        "我在你们商城买了一个冰箱和一个洗衣机，冰箱制冷有问题，洗衣机倒是正常。现在只要求把冰箱退了，洗衣机的发票也一起寄给我。",
        "快递显示签收了但我根本没收到货，打客服电话一直占线，已经三天了，你们到底管不管？"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n===== 测试用例 {i} =====")
        print(f"原文: {case}")
        result = extract_complaint_langchain(case)
        print(f"抽取结果:\n{result}")