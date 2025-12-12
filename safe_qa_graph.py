# safe_qa_graph.py
from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END

from routing_hub import analyze_query
from risk_hub import detect_risk_discrete
from guard_graph import is_predicted_violation
from llm_client import llm_chat
from rag_helper import retrieve_relevant_docs, format_docs_for_prompt


class QAState(TypedDict, total=False):
    """
    安全问答流水线的状态结构：
      - user_query         : 用户原始问题
      - difficulty         : 检测难度模式（lenient/normal/strict/any_risk/overall_l2plus）
      - router_ctx_in      : 输入侧路由分析结果
      - input_risk_profile : 输入侧风险检测结果
      - input_violation    : 输入侧判定是否违规 (0/1)
      - model_answer       : 主模型的原始回答
      - router_ctx_out     : 输出侧路由分析结果
      - output_risk_profile: 输出侧风险检测结果
      - output_violation   : 输出侧判定是否违规 (0/1)
      - rag_docs           : RAG 检索到的文档列表
      - final_answer       : 最终给用户的回答（可能是原回答，也可能是安全代答）
    """
    user_query: str
    difficulty: str

    router_ctx_in: Dict[str, Any]
    input_risk_profile: Dict[str, Any]
    input_violation: int

    model_answer: str
    router_ctx_out: Dict[str, Any]
    output_risk_profile: Dict[str, Any]
    output_violation: int

    rag_docs: List[Dict[str, Any]]
    final_answer: str


# =========================
# 小工具：从风险 profile 中选出核心风险
# =========================

def _guess_focus_risk_from_profiles(
    input_risk: Dict[str, Any],
    output_risk: Dict[str, Any],
) -> List[str]:
    """
    从 input/output 的 risk_levels 里，根据等级挑几个“最核心”的风险标签，
    作为 RAG 的 focus_risk 提示。

    策略：
      - 先看 input_risk，再看 output_risk（谁更高用谁）
      - 只保留 MEDIUM/HIGH 的类别
      - 若一个都没有，就返回空列表（由 RAG 自己根据 router_ctx 选）
    """
    def extract_high_levels(profile: Dict[str, Any]) -> List[str]:
        levels = (profile.get("risk_levels") or {}) if isinstance(profile, dict) else {}
        out: List[str] = []
        for cat, lv in levels.items():
            lv_norm = (lv or "NONE").upper()
            if lv_norm in {"MEDIUM", "HIGH"}:
                out.append(cat)
        return out

    in_high = extract_high_levels(input_risk)
    out_high = extract_high_levels(output_risk)

    # 优先用输入侧，如果没有再用输出侧
    if in_high:
        return list(set(in_high))
    if out_high:
        return list(set(out_high))
    return []


# =========================
# 各个节点（Node）
# =========================

def route_input_node(state: QAState) -> QAState:
    """
    输入侧：围栏路由
    """
    q = state.get("user_query", "")
    router_ctx = analyze_query(q)
    state["router_ctx_in"] = router_ctx
    return state


def risk_input_node_factory(difficulty: str = "normal"):

    def risk_input_node(state: QAState) -> QAState:
        q = state.get("user_query", "")
        router_ctx = state.get("router_ctx_in") or {}

        risk_profile = detect_risk_discrete(
            q,
            router_ctx,
            mode="input",
            use_rag_on_uncertain=False,
        )
        state["input_risk_profile"] = risk_profile

        pred = is_predicted_violation(risk_profile, mode=difficulty)
        state["input_violation"] = pred
        return state

    return risk_input_node


def input_decision_node(state: QAState) -> QAState:
    return state


def main_answer_node(state: QAState) -> QAState:
    """
    主回答模型节点：
      - 对输入侧判定为“不违规”的问题，用主模型正常回答。
    """
    q = state.get("user_query", "")

    system_prompt = (
        "你是一个乐于助人的中文问答助手，需要在保证安全、合法、"
        "符合伦理的前提下，尽可能准确和清晰地回答用户的问题。"
    )
    user_prompt = f"用户问题如下：\n{q}\n\n请用中文回答。"

    answer = llm_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=1024,
    )

    state["model_answer"] = answer
    return state


def route_output_node(state: QAState) -> QAState:
    """
    输出侧：对模型的回答再做一次路由分析。
    """
    ans = state.get("model_answer", "") or ""
    router_ctx = analyze_query(ans)
    state["router_ctx_out"] = router_ctx
    return state


def risk_output_node_factory(difficulty: str = "normal"):

    def risk_output_node(state: QAState) -> QAState:
        ans = state.get("model_answer", "") or ""
        router_ctx = state.get("router_ctx_out") or {}

        risk_profile = detect_risk_discrete(
            ans,
            router_ctx,
            mode="output",
            use_rag_on_uncertain=False,
        )
        state["output_risk_profile"] = risk_profile

        pred = is_predicted_violation(risk_profile, mode=difficulty)
        state["output_violation"] = pred
        return state

    return risk_output_node


def output_decision_node(state: QAState) -> QAState:
    return state


def safe_answer_node(state: QAState) -> QAState:
    """
    安全代答 / 重写节点：

      用于两种情况：
        1) 输入侧已经判为高风险：直接给一个安全版回答（拒绝 + 劝阻 + 建议）
        2) 输出侧检测到风险：在知道原回答的情况下，进行安全重写或替代回答

      这里会用到：
        - 用户原始问题 user_query
        - （可选）主模型原始回答 model_answer
        - 输入/输出风险信息 input_risk_profile / output_risk_profile
        - 输入侧路由 router_ctx_in
        - RAG 文档 rag_docs
    """
    q = state.get("user_query", "")
    orig_answer = state.get("model_answer", "")
    input_risk = state.get("input_risk_profile", {}) or {}
    output_risk = state.get("output_risk_profile", {}) or {}
    router_ctx_in = state.get("router_ctx_in") or {}

    # ---------- RAG：检索安全知识 ----------
    rag_query_parts = [q]
    if orig_answer:
        rag_query_parts.append(orig_answer)
    rag_query = "\n\n".join(rag_query_parts)

    # 关键修改：把 router_ctx + focus_risk 喂给 RAG
    focus_risk = _guess_focus_risk_from_profiles(input_risk, output_risk)
    docs = retrieve_relevant_docs(
        query=rag_query,
        top_k=3,
        router_ctx=router_ctx_in,
        mode="output",
        focus_risk=focus_risk,
        language_hint=router_ctx_in.get("lang"),
    )
    state["rag_docs"] = docs
    docs_text = format_docs_for_prompt(docs)

    # ---------- 构造提示 ----------
    system_prompt = (
        "你是一个内容安全和合规方面的智能助手，需要在保证用户安全、"
        "遵守法律法规和主流伦理价值观的前提下，对用户的问题给出一个"
        "既安全又尽可能有用的回答。"
        "对于违法犯罪、自残自杀、暴力伤害、隐私泄露等高风险请求，"
        "你应该拒绝提供具体操作方法，并进行温和劝阻和风险提示，"
        "可以提供安全的替代方案或求助途径。"
    )

    user_prompt_parts: List[str] = []

    user_prompt_parts.append(f"【用户原始问题】\n{q}\n")

    if orig_answer:
        user_prompt_parts.append(
            "【模型原始回答（可能存在安全风险，供你参考和重写，不要原样输出有害部分）】\n"
            f"{orig_answer}\n"
        )

    user_prompt_parts.append(
        "【输入侧风险检测结果】\n"
        f"{input_risk}\n"
    )
    user_prompt_parts.append(
        "【输出侧风险检测结果】\n"
        f"{output_risk}\n"
    )

    user_prompt_parts.append(
        "【与安全相关的参考文档（RAG 检索结果）】\n"
        f"{docs_text}\n"
    )

    user_prompt_parts.append(
        "请你综合以上信息，给出一个安全、合规、温和且有帮助的回答：\n"
        "  - 不要提供任何违法犯罪、自残自杀、暴力伤害、隐私泄露、"
        "    高风险金融、恐怖主义等具体操作细节；\n"
        "  - 可以解释为什么这些行为有风险；\n"
        "  - 可以建议用户寻求专业帮助（如心理医生、官方渠道）；\n"
        "  - 对于比较正常但有争议的话题，给出中立、负责任的解释。\n"
        "请用中文回答。"
    )

    user_prompt = "\n\n".join(user_prompt_parts)

    safe_answer = llm_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=1024,
    )

    state["final_answer"] = safe_answer
    return state


def finalize_answer_node(state: QAState) -> QAState:
    state["final_answer"] = state.get("model_answer", "") or ""
    return state


# =========================
# 条件路由函数
# =========================

def input_route_fn(state: QAState) -> str:
    if state.get("input_violation", 0) == 1:
        return "safe_answer"
    else:
        return "main_answer"


def output_route_fn(state: QAState) -> str:
    if state.get("output_violation", 0) == 1:
        return "safe_answer"
    else:
        return "finalize_answer"


# =========================
# 构建安全 QA LangGraph
# =========================

def build_safe_qa_graph(detection_difficulty: str = "normal"):
    workflow = StateGraph(QAState)

    workflow.add_node("route_input", route_input_node)
    workflow.add_node("risk_input", risk_input_node_factory(detection_difficulty))
    workflow.add_node("input_decision", input_decision_node)

    workflow.add_node("safe_answer", safe_answer_node)
    workflow.add_node("main_answer", main_answer_node)
    workflow.add_node("route_output", route_output_node)
    workflow.add_node("risk_output", risk_output_node_factory(detection_difficulty))
    workflow.add_node("output_decision", output_decision_node)
    workflow.add_node("finalize_answer", finalize_answer_node)

    workflow.set_entry_point("route_input")

    workflow.add_edge("route_input", "risk_input")
    workflow.add_edge("risk_input", "input_decision")

    workflow.add_conditional_edges(
        "input_decision",
        input_route_fn,
        {
            "safe_answer": "safe_answer",
            "main_answer": "main_answer",
        },
    )

    workflow.add_edge("main_answer", "route_output")
    workflow.add_edge("route_output", "risk_output")
    workflow.add_edge("risk_output", "output_decision")

    workflow.add_conditional_edges(
        "output_decision",
        output_route_fn,
        {
            "safe_answer": "safe_answer",
            "finalize_answer": "finalize_answer",
        },
    )

    workflow.add_edge("safe_answer", END)
    workflow.add_edge("finalize_answer", END)

    app = workflow.compile()
    return app


if __name__ == "__main__":
    graph = build_safe_qa_graph(detection_difficulty="normal")

    while True:
        q = input("User> ").strip()
        if not q:
            break

        init_state: QAState = {
            "user_query": q,
            "difficulty": "normal",
        }
        result: QAState = graph.invoke(init_state)

        print("\n[router_ctx_in]", result.get("router_ctx_in"))
        print("\n[input_risk_profile]", result.get("input_risk_profile"))
        print("\n[input_violation]", result.get("input_violation"))
        print("\n[model_answer]", result.get("model_answer"))
        print("\n[output_risk_profile]", result.get("output_risk_profile"))
        print("\n[output_violation]", result.get("output_violation"))
        print("\n[rag_docs]", result.get("rag_docs"))
        print("\n[final_answer]", result.get("final_answer"))
        print("-" * 80)
