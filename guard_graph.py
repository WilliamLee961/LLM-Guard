# guard_graph.py
from typing import Dict, Any, TypedDict

from langgraph.graph import StateGraph, END

from routing_hub import analyze_query
from risk_hub import detect_risk_discrete


class GuardState(TypedDict, total=False):
    """
    LangGraph 的状态结构：
      - text           : 待检测文本（用户 Query 或模型输出）
      - router_ctx     : 围栏路由中枢的分析结果
      - risk_profile   : 风险检测中枢的输出
      - pred_violation : 0/1 二分类结果（0 不违规，1 违规）
    """
    text: str
    router_ctx: Dict[str, Any]
    risk_profile: Dict[str, Any]
    pred_violation: int
    runtime_mode: str


# =========================
# 判违规策略（含难度开关）
# =========================

def is_predicted_violation(
    risk_profile: Dict[str, Any],
    mode: str = "normal",
) -> int:
    """
    根据 risk_hub.detect_risk_discrete 的输出，压成二分类结果：
      - 1: 检测到有风险（预测违规）
      - 0: 未检测到风险（预测不违规）

    mode 为“检测难度开关”：
      - "lenient"        : 宽松模式，只有整体风险较高 / 强维度明显时才判违规
      - "normal"         : 默认模式，兼顾安全与误杀
      - "strict"         : 严格模式，更接近以前的 any_risk（宁可多杀少放）
    同时保留兼容：
      - "any_risk"       : 旧逻辑，只要任意维度 != NONE 就违规
      - "overall_l2plus" : 旧逻辑，只要 overall ∈ {L2,L3,L4} 就违规
    """
    levels = risk_profile.get("risk_levels") or {}
    if not isinstance(levels, dict):
        levels = {}

    overall = (risk_profile.get("overall_risk_level") or "L1").upper()

    # --- 旧模式兼容 ---
    if mode == "overall_l2plus":
        return 1 if overall in ("L2", "L3", "L4") else 0

    if mode == "any_risk":
        for v in levels.values():
            if (v or "").upper() != "NONE":
                return 1
        return 0

    # --- 新模式：lenient / normal / strict ---
    # 强风险维度：一旦出现一般都比较严重
    STRONG_DIMS = {
        "PHYSICAL_HARM",
        "SELF_HARM",
        "CRIMES_ILLEGAL",
        "MALICIOUS_USE",
        "PRIVACY_PROPERTY",
        "FRAUD_DECEPTION",
        "FINANCIAL_RISK",
        "ATTACK_MODEL",
        "SEXUAL_CONTENT",        # 可视需要保留或移除
    }

    # 等级转数值
    level2num = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def get_level_num(cat: str) -> int:
        v = (levels.get(cat) or "NONE").upper()
        return level2num.get(v, 0)

    # 软风险维度 = 除强风险以外的所有
    soft_dims = [c for c in levels.keys() if c not in STRONG_DIMS]

    def any_dim_ge(dims, threshold: int) -> bool:
        return any(get_level_num(d) >= threshold for d in dims)

    # 不认识的 mode 当 normal 处理
    if mode not in ("lenient", "normal", "strict"):
        mode = "normal"

    # ========== lenient：宽松模式 ==========
    if mode == "lenient":
        # L4：无条件违规
        if overall == "L4":
            return 1

        # 强维度 HIGH（3）直接违规
        if any_dim_ge(STRONG_DIMS, 3):
            return 1

        # L3：强维度 >= LOW，或者软维度 >= MEDIUM 才违规
        if overall == "L3":
            if any_dim_ge(STRONG_DIMS, 1):
                return 1
            if any_dim_ge(soft_dims, 2):
                return 1
            return 0

        # L2：只在强维度 >= MEDIUM 时才违规
        if overall == "L2":
            if any_dim_ge(STRONG_DIMS, 2):
                return 1
            return 0

        # L1：默认不违规（除非极端强维度已上面捕获）
        return 0

    # ========== normal：默认模式 ==========
    if mode == "normal":
        # L4：必违规
        if overall == "L4":
            return 1

        # 任何强维度 MEDIUM/HIGH 直接违规
        if any_dim_ge(STRONG_DIMS, 2):
            return 1

        # overall 在 L2/L3 时：
        if overall in ("L2", "L3"):
            # 强维度 LOW 也算违规（例如轻度自残倾向、轻度违法意图）
            if any_dim_ge(STRONG_DIMS, 1):
                return 1
            # 软维度 MEDIUM/HIGH 也算违规（例如明显虚假信息、明显伦理问题）
            if any_dim_ge(soft_dims, 2):
                return 1

        # 其余情况不违规
        return 0

    # ========== strict：严格模式 ==========
    if mode == "strict":
        # 只要强维度有任何非 NONE 就违规
        if any_dim_ge(STRONG_DIMS, 1):
            return 1
        # overall 在 L2/L3/L4 时，只要软维度 LOW 以上就违规
        if overall in ("L2", "L3", "L4"):
            if any_dim_ge(soft_dims, 1):
                return 1
        return 0


# =========================
# 各个节点（Node）函数
# =========================

def route_node_factory(runtime_mode: str = "accurate"):
    def route_node(state: GuardState) -> GuardState:
        """
        围栏路由节点：
          输入：state["text"]
          输出：state["router_ctx"]
        """
        text = state.get("text", "")
        router_ctx = analyze_query(text, mode=runtime_mode)
        state["router_ctx"] = router_ctx
        state["runtime_mode"] = runtime_mode
        return state

    return route_node


def risk_node_factory(difficulty: str = "normal", runtime_mode: str = "accurate"):
    """
    工厂函数，返回真正的 risk_node，
    这样我们可以把 detection difficulty 作为配置注入。
    """

    def risk_node(state: GuardState) -> GuardState:
        """
        风险检测节点：
          输入：state["text"], state["router_ctx"]
          输出：state["risk_profile"], state["pred_violation"]
        """
        text = state.get("text", "")
        router_ctx = state.get("router_ctx") or {}

        risk_profile = detect_risk_discrete(
            text,
            router_ctx,
            mode="input",
            runtime_mode=runtime_mode,
        )
        state["risk_profile"] = risk_profile

        pred = is_predicted_violation(risk_profile, mode=difficulty)
        state["pred_violation"] = pred

        return state

    return risk_node


# =========================
# 构建 LangGraph 工作流
# =========================

def build_guard_graph(detection_difficulty: str = "normal", runtime_mode: str = "accurate"):
    """
    构建一个两节点的 LangGraph：
      entry: route  →  risk  →  END

    detection_difficulty:
      - "lenient" / "normal" / "strict"
      - 或兼容老模式："any_risk" / "overall_l2plus"
    """
    workflow = StateGraph(GuardState)

    # 添加节点
    workflow.add_node("route", route_node_factory(runtime_mode=runtime_mode))
    workflow.add_node(
        "risk",
        risk_node_factory(difficulty=detection_difficulty, runtime_mode=runtime_mode),
    )

    # 指定入口节点
    workflow.set_entry_point("route")

    # 配置边：route -> risk -> END
    workflow.add_edge("route", "risk")
    workflow.add_edge("risk", END)

    # 编译成可调用的 graph
    app = workflow.compile()
    return app


if __name__ == "__main__":
    # 简单交互测试
    graph = build_guard_graph(detection_difficulty="normal")

    while True:
        q = input("User> ").strip()
        if not q:
            break
        state: GuardState = {"text": q}
        result = graph.invoke(state)
        print("router_ctx:", result.get("router_ctx"))
        print("risk_profile:", result.get("risk_profile"))
        print("pred_violation:", result.get("pred_violation"))
        print("-" * 80)
