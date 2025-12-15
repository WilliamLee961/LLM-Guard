# app.py
"""
应用入口封装：

- build_guard_graph:   纯“内容检测”（不回答，只给 router_ctx + risk_profile + pred_violation）
- build_safe_qa_graph: 安全问答（输入/输出双向检测 + 安全代答 / 重写）

这里对两种模式都做了一层简单封装，方便在其它地方 import 使用。
"""

from typing import Dict, Any

from guard_graph import build_guard_graph, GuardState
from safe_qa_graph import build_safe_qa_graph, QAState


# =========================
# 1) 纯检测模式（不回答）
# =========================conda --version

# 默认用 normal 难度，你可以在这里改成 strict / lenient
_guard_app = build_guard_graph(detection_difficulty="normal")


def run_guardrails(text: str, difficulty: str = "normal") -> Dict[str, Any]:
    """
    纯“侧载式安全检测”调用入口。

    参数：
      text        : 待检测文本（用户输入或模型输出）
      difficulty  : 检测难度开关（lenient / normal / strict / any_risk / overall_l2plus）

    返回：
      {
        "router_ctx": {...},
        "risk_profile": {...},
        "pred_violation": 0/1
      }
    """
    # 如果你希望动态调整 difficulty，可以在这里根据参数重建 graph；
    # 为了简单，这里直接用 difficulty 传进 is_predicted_violation 的配置版 guard_graph。
    guard_app = build_guard_graph(detection_difficulty=difficulty)

    state: GuardState = {"text": text}
    result: GuardState = guard_app.invoke(state)

    return {
        "router_ctx": result.get("router_ctx"),
        "risk_profile": result.get("risk_profile"),
        "pred_violation": result.get("pred_violation", 0),
    }


# =========================
# 2) 完整安全问答模式
# =========================

_safe_qa_app = build_safe_qa_graph(detection_difficulty="normal")


def run_safe_qa(user_query: str, difficulty: str = "normal") -> Dict[str, Any]:
    """
    完整安全问答入口（输入/输出双向检测 + RAG + 安全代答/重写）。

    参数：
      user_query  : 用户原始问题
      difficulty  : 检测难度开关（lenient / normal / strict / any_risk / overall_l2plus）

    返回：
      {
        "router_ctx_in": {...},
        "input_risk_profile": {...},
        "input_violation": 0/1,
        "model_answer": "...",          # 主模型原始回答（如果有）
        "router_ctx_out": {...},
        "output_risk_profile": {...},
        "output_violation": 0/1,
        "rag_docs": [...],              # RAG 命中的文档
        "final_answer": "..."           # 最终给用户的安全回答
      }
    """
    safe_app = build_safe_qa_graph(detection_difficulty=difficulty)

    init_state: QAState = {
        "user_query": user_query,
        "difficulty": difficulty,
    }

    result: QAState = safe_app.invoke(init_state)
    return result


if __name__ == "__main__":
    # 简单测试：你可以选模式
    mode = input("选择模式 [1=纯检测 / 2=安全问答]: ").strip() or "1"

    while True:
        q = input("User> ").strip()
        if not q:
            break

        if mode == "1":
            out = run_guardrails(q, difficulty="normal")
            print("router_ctx:", out["router_ctx"])
            print("risk_profile:", out["risk_profile"])
            print("pred_violation:", out["pred_violation"])
        else:
            out = run_safe_qa(q, difficulty="normal")
            print("[router_ctx_in]", out.get("router_ctx_in"))
            print("[input_risk_profile]", out.get("input_risk_profile"))
            print("[input_violation]", out.get("input_violation"))
            print("[model_answer]", out.get("model_answer"))
            print("[output_risk_profile]", out.get("output_risk_profile"))
            print("[output_violation]", out.get("output_violation"))
            print("[rag_docs]", out.get("rag_docs"))
            print("[final_answer]", out.get("final_answer"))

        print("-" * 80)
