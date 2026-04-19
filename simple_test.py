#!/usr/bin/env python3
"""简单测试 LLM API 调用"""
import os
import requests

API_KEY = "5888637f-1af6-4beb-b6dc-0b8c1dd20258"
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-1-5-pro-32k-250115"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("测试火山引擎 API...")
print(f"模型: {MODEL}")

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "你好，请简单回复'API调用成功'"}
    ],
    "max_tokens": 50
}

response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)

if response.status_code == 200:
    result = response.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"成功: {content}")
else:
    print(f"失败: {response.text}")
