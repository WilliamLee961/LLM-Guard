# llm_client.py
import json
import os
import re
from typing import Dict, Any, List, Optional

from openai import OpenAI


# =========================
# OpenAI 兼容客户端配置
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://100.87.153.109:8000/v1")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    timeout=OPENAI_TIMEOUT,
)

# 你实际用的模型名字：
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "Qwen")

# =========================
# 基础聊天封装
# =========================

def llm_chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    通用 LLM 对话封装：返回模型生成的字符串。

    - 适配 openai>=1.0.0 的新接口：client.chat.completions.create
    - 不强行依赖 reasoning_content 字段（本地模型未必支持）
    - 如果需要 JSON 输出，可在上层传入 response_format={"type": "json_object"}
    """
    kwargs: Dict[str, Any] = dict(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)

    msg = resp.choices[0].message
    content = msg.content

    # 有些服务可能返回 list（多段），这里统一转 str
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    if content is None:
        # 理论上不应该为 None，打印一下方便调试
        print("LLM 返回 content=None, raw message:", msg)
        return ""

    return content.strip()


# =========================
# JSON 工具函数
# =========================

def _strip_code_fences(text: str) -> str:
    """
    去掉 ``` 和 ```json 等代码块包裹，只保留内部内容。
    """
    text = re.sub(r"```[a-zA-Z0-9_]*\s*", "", text)  # 去掉 ```json\n / ```\n
    text = text.replace("```", "")
    return text


def _extract_json_candidates(text: str) -> List[str]:
    """
    按大括号深度扫描，提取可能的 JSON 子串。
    """
    candidates: List[str] = []
    brace_stack = 0
    start_idx: Optional[int] = None

    for i, ch in enumerate(text):
        if ch == "{":
            if brace_stack == 0:
                start_idx = i
            brace_stack += 1
        elif ch == "}":
            if brace_stack > 0:
                brace_stack -= 1
                if brace_stack == 0 and start_idx is not None:
                    candidates.append(text[start_idx : i + 1])
                    start_idx = None

    return candidates


def llm_chat_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    要求模型“输出 JSON 对象”的封装：

    1. 调用 llm_chat 时加上 response_format={"type": "json_object"}（如果服务端支持）；
    2. 解析时：
       - 去掉 ```json ... ``` 包裹
       - 按大括号深度提取 JSON 子串
       - 尝试逐个 json.loads，取最后一个成功解析的
       - 全部失败则返回 {}
    """
    raw = llm_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    cleaned = raw.strip()
    cleaned = _strip_code_fences(cleaned)

    # 提取 JSON 候选
    candidates = _extract_json_candidates(cleaned)
    last_good: Optional[Dict[str, Any]] = None

    for cand in candidates:
        try:
            obj = json.loads(cand)
            last_good = obj
        except Exception:
            continue

    if last_good is not None:
        return last_good

    # 兜底：直接整体解析一次
    try:
        return json.loads(cleaned)
    except Exception:
        # 最终兜底
        # print("llm_chat_json parse error, raw:", raw)
        return {}
