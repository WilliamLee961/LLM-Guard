# safe_qa_graph.py
from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END

from routing_hub import analyze_query
from risk_hub import detect_risk_discrete
from llm_client import llm_chat
from rag_helper import retrieve_relevant_docs, format_docs_for_prompt
from response_policy import (
    is_low_quality_safe_answer,
    render_fallback_safe_response,
    rewrite_query_for_safe_mode,
    select_response_action,
)

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
      - use_rag            : [新增] 是否启用 RAG
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
    use_rag: bool  # 状态中携带开关
    guard_stage: str
    response_strategy: Dict[str, Any]
    rewritten_user_query: str


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


def _derive_safe_objective(
    query: str,
    router_ctx_in: Dict[str, Any],
    strategy: Dict[str, Any],
) -> str:
    """
    从用户原始意图中提取“可保留目标”，用于提升安全改写的任务可用性。
    """
    intent = str(router_ctx_in.get("intent", "OTHER")).upper()
    risks = [name for name, _ in (strategy.get("top_risks") or [])]
    q = (query or "").strip()
    q_lower = q.lower()

    if intent == "ATTACK_MODEL":
        return "将请求转换为防御目标：识别提示注入风险、给出系统加固与审计方案。"
    if "SELF_HARM" in risks or intent == "SELF_HARM_INTENT":
        return "将目标转换为情绪稳定和求助支持：先降风险，再给现实可执行求助步骤。"
    if {"PHYSICAL_HARM", "CRIMES_ILLEGAL", "MALICIOUS_USE"} & set(risks):
        return "保留合法学习目标，转换为法律合规、风险预防和安全替代方案。"
    if "sql" in q_lower or "数据库" in q:
        return "保留数据库优化目标，输出合规性能优化步骤，不涉及攻击或破坏。"
    if "ppt" in q_lower or "汇报" in q or "方案" in q:
        return "保留汇报目标，输出结构化提纲与落地计划。"
    if "代码" in q or "python" in q_lower or "java" in q_lower:
        return "保留开发目标，提供最小权限、输入校验与安全库实践。"
    return "保留用户的合法业务目标，移除危险实现细节，并给可执行替代路径。"


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
            use_rag_on_uncertain=True,
        )
        state["input_risk_profile"] = risk_profile

        strategy = select_response_action(
            stage="input",
            difficulty=difficulty,
            input_risk_profile=risk_profile,
            output_risk_profile={},
            router_ctx_in=router_ctx,
            router_ctx_out={},
        )
        state["response_strategy"] = strategy
        action = str(strategy.get("action", "ALLOW")).upper()
        state["input_violation"] = 1 if action in {"SAFE_ANSWER", "REFUSE", "ESCALATE"} else 0
        return state

    return risk_input_node


def input_decision_node(state: QAState) -> QAState:
    strategy = state.get("response_strategy") or {}
    if str(strategy.get("action", "ALLOW")).upper() == "REWRITE":
        state["rewritten_user_query"] = rewrite_query_for_safe_mode(
            state.get("user_query", ""),
            strategy,
        )
    return state


def main_answer_node(state: QAState) -> QAState:
    """
    主回答模型节点：
      - 对输入侧判定为“不违规”的问题，用主模型正常回答。
    """
    q = state.get("rewritten_user_query", "") or state.get("user_query", "")

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
            use_rag_on_uncertain=True,
        )
        state["output_risk_profile"] = risk_profile

        strategy = select_response_action(
            stage="output",
            difficulty=difficulty,
            input_risk_profile=state.get("input_risk_profile") or {},
            output_risk_profile=risk_profile,
            router_ctx_in=state.get("router_ctx_in") or {},
            router_ctx_out=router_ctx,
        )
        state["response_strategy"] = strategy
        action = str(strategy.get("action", "ALLOW")).upper()
        state["output_violation"] = 0 if action == "ALLOW" else 1
        return state

    return risk_output_node


def output_decision_node(state: QAState) -> QAState:
    return state


def safe_answer_node(state: QAState) -> QAState:
    """
    【核心】生成安全回复 / 拒答 / 重写
    基于 chapter-3 响应决策策略（ALLOW/REWRITE/SAFE_ANSWER/REFUSE/ESCALATE）
    结合风险分层与难度模式，提升安全回复的可用性和可执行性。
    """
    q = state.get("user_query", "")
    orig_answer = state.get("model_answer", "")
    input_risk = state.get("input_risk_profile", {}) or {}
    output_risk = state.get("output_risk_profile", {}) or {}
    router_ctx_in = state.get("router_ctx_in") or {}
    router_ctx_out = state.get("router_ctx_out") or {}
    difficulty = state.get("difficulty", "normal")

    guard_stage = state.get("guard_stage")
    if not guard_stage:
        if state.get("output_violation", 0) == 1:
            guard_stage = "output"
        elif state.get("input_violation", 0) == 1:
            guard_stage = "input"
        else:
            guard_stage = "output" if orig_answer else "input"

    strategy = select_response_action(
        stage=guard_stage,
        difficulty=difficulty,
        input_risk_profile=input_risk,
        output_risk_profile=output_risk,
        router_ctx_in=router_ctx_in,
        router_ctx_out=router_ctx_out,
    )
    state["response_strategy"] = strategy

    # 获取 RAG 开关状态，默认为 False (不开启)
    use_rag = state.get("use_rag", False)
    
    docs = []
    if use_rag:
        # ---------- RAG：检索安全知识 ----------
        # 只有在 use_rag 为 True 时才执行检索
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
        "你是“安全围栏响应决策模块”的执行助手。目标不是只拒答，而是："
        "在不输出危险细节的前提下，尽可能保留帮助性和任务可用性。"
        "你必须遵循 action 策略："
        "REWRITE=保留合法信息并重写危险段；"
        "SAFE_ANSWER=给出安全替代方案；"
        "REFUSE=明确拒绝并给替代路径；"
        "ESCALATE=优先现实安全建议与求助路径。"
    )

    user_prompt_parts: List[str] = []
    safe_objective = _derive_safe_objective(q, router_ctx_in, strategy)

    user_prompt_parts.append(f"【用户原始问题】\n{q}\n")
    user_prompt_parts.append(f"【应保留的合法目标】\n{safe_objective}\n")
    user_prompt_parts.append(f"【当前处理阶段】\n{guard_stage}\n")
    user_prompt_parts.append(f"【决策策略】\n{strategy}\n")

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

    if strategy.get("action") == "REWRITE":
        user_prompt_parts.append(
            "请执行“安全改写”，输出一份可直接给用户的最终回答：\n"
            "1) 先保留“应保留的合法目标”；\n"
            "2) 保留原回答中合法且有用的信息；\n"
            "3) 删除或替换危险细节（违法、伤害、越狱、欺诈等）；\n"
            "4) 给出2-4条可执行的合规替代步骤；\n"
            "5) 给出一个下一步可执行动作（如检查清单/求助路径/防护措施）；\n"
            "6) 语言自然，不要只说套话，不要空泛道歉。\n"
            "请直接输出最终文本，不要解释你的改写过程。"
        )
    else:
        user_prompt_parts.append(
            "请执行“安全代答/拒答”，输出一份高帮助性的最终回答：\n"
            "1) 先用1句话说明风险边界；\n"
            "2) 再明确“可保留目标”；\n"
            "3) 给出2-4条可执行替代方案（必须可落地）；\n"
            "4) 给出下一步建议（必要时提示求助渠道）；\n"
            "5) 不得输出可直接执行的危险细节；\n"
            "6) 不要只有‘我不能回答’，必须提供建设性替代。\n"
            "请直接输出最终文本。"
        )

    user_prompt = "\n\n".join(user_prompt_parts)

    safe_answer = llm_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=1024,
    )

    effective_profile = output_risk if guard_stage == "output" else input_risk
    if is_low_quality_safe_answer(safe_answer):
        safe_answer = render_fallback_safe_response(
            query=q,
            strategy=strategy,
            risk_profile=effective_profile,
            stage=guard_stage,
            original_answer=orig_answer,
        )

    state["final_answer"] = safe_answer
    return state


def finalize_answer_node(state: QAState) -> QAState:
    state["final_answer"] = state.get("model_answer", "") or ""
    if "response_strategy" not in state:
        state["response_strategy"] = {
            "action": "ALLOW",
            "stage": "output",
            "overall_risk_level": (state.get("output_risk_profile") or {}).get("overall_risk_level", "L1"),
        }
    return state


# =========================
# 条件路由函数
# =========================

def input_route_fn(state: QAState) -> str:
    strategy = state.get("response_strategy") or {}
    action = str(strategy.get("action", "ALLOW")).upper()
    if action in {"SAFE_ANSWER", "REFUSE", "ESCALATE"}:
        return "safe_answer"
    else:
        return "main_answer"


def output_route_fn(state: QAState) -> str:
    strategy = state.get("response_strategy") or {}
    action = str(strategy.get("action", "ALLOW")).upper()
    if action != "ALLOW":
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
