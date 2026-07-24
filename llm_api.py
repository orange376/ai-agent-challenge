# Day 3: DeepSeek API 封装 — 手写 HTTP 请求，stop 参数可控制生成终止
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def call_llm(prompt,system_prompt="你是一个有用的助手",stop=None):

    api_key = os.getenv('DEEPSEEK_API_KEY')
    endpoint = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-v4-flash",     
        "messages": [
            {"role": "system", "content": system_prompt}, ##系统提示，人设
            {"role": "user", "content": prompt}   ##用户输入
        ],
        "temperature": 0.7,    ##创意度
        "max_tokens": 1000,    ##最大输出长度
        "stream": False,  ##是否流式输出
    }
    if stop:
        payload["stop"] = stop  ##设置停止词，模型在输出到停止词时会停止生成
        
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"] ##解析格式受到模型输出格式影响(是否流式输出)
    except requests.exceptions.Timeout:
        print("请求超时")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP 错误：{e}")
        print(f"响应内容：{response.text}")
        return None
    except Exception as e:
        print(f"未知错误：{e}")
        return None
    
if __name__ == "__main__":
    result = call_llm("用一句话解释什么是人工智能")
    print(result)