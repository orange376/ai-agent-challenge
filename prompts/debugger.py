# Day 5: Bug 排查 Prompt — CoT 三步推理法（分析→定位→修复建议）
from llm_api import call_llm

SYSTEM_PROMPT = """你是一个资深的 Python 后端调试专家。当用户给你一段报错信息和相关代码时，
你必须按照以下步骤进行推理，每一步都要明确输出你的思考过程：

第一步：错误分类
- 判断这是语法错误、运行时错误、还是逻辑错误
- 指出具体的错误类型（如 KeyError, TypeError 等）

第二步：定位根因
- 指出错误发生的具体代码行
- 解释为什么会发生这个错误

第三步：给出解决方案
- 提供至少一种修复方案
- 说明如何验证修复是否生效

请严格按照这个格式输出，在每个步骤前标注"## 第X步"。"""

def debug_error(error_info, code_snippet=""):
    """分析代码错误并给出修复建议"""
    user_prompt = f"报错信息：\n{error_info}\n\n相关代码：\n{code_snippet}"
    return call_llm(user_prompt, system_prompt=SYSTEM_PROMPT)

# 测试
if __name__ == "__main__":
    test_cases = [
    # ========== 类别1：语法错误 ==========
    {
        "error": "SyntaxError: invalid syntax\n  File 'app.py', line 3\n    if x = 5:\n       ^",
        "code": "x = 10\nif x = 5:\n    print('x is 5')",
    },
    {
        "error": "IndentationError: expected an indented block\n  File 'test.py', line 3\n    print('hello')\n    ^",
        "code": "def greet(name):\nprint(f'Hello, {name}')",
    },

    # ========== 类别2：运行时类型错误 ==========
    {
        "error": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "code": "def calculate(x, y):\n    return x + y\n\nresult = calculate(5, '10')",
    },
    {
        "error": "TypeError: 'NoneType' object is not iterable\n  File 'app.py', line 8, in process\n    for item in data:",
        "code": "def get_data():\n    return None\n\ndata = get_data()\nfor item in data:\n    print(item)",
    },

    # ========== 类别3：键/索引错误 ==========
    {
        "error": "KeyError: 'name'\n\nTraceback (most recent call last):\n  File 'app.py', line 12, in <module>\n    print(user['name'])",
        "code": "user = {'username': 'john', 'age': 25}\nprint(user['name'])",
    },
    {
        "error": "IndexError: list index out of range\n  File 'app.py', line 5, in <module>\n    print(items[3])",
        "code": "items = [1, 2, 3]\nprint(items[3])",
    },

    # ========== 类别4：属性/导入错误 ==========
    {
        "error": "AttributeError: 'str' object has no attribute 'append'\n  File 'app.py', line 6, in <module>\n    result.append('world')",
        "code": "result = 'hello'\nresult.append('world')",
    },
    {
        "error": "ModuleNotFoundError: No module named 'pandas'\n  File 'app.py', line 1, in <module>\n    import pandas as pd",
        "code": "import pandas as pd\n\ndf = pd.DataFrame({'A': [1, 2, 3]})\nprint(df.head())",
    },

    # ========== 类别5：值错误 ==========
    {
        "error": "ValueError: invalid literal for int() with base 10: 'abc'\n  File 'app.py', line 4, in <module>\n    num = int(user_input)",
        "code": "user_input = 'abc'\nnum = int(user_input)\nprint(num * 2)",
    },

    # ========== 类别6：逻辑/资源错误 ==========
    {
        "error": "RecursionError: maximum recursion depth exceeded\n  File 'app.py', line 4, in factorial\n    return n * factorial(n-1)",
        "code": "def factorial(n):\n    return n * factorial(n-1)\n\nprint(factorial(5))",
    }
]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n===== Bug 排查案例 {i} =====")
        print(f"错误类型: {case['error']}")
        result = debug_error(case['error'], case['code'])
        print(f"排查过程:\n{result}")