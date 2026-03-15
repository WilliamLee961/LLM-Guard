from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

ActionType = Literal["ALLOW", "REWRITE", "SAFE_ANSWER", "REFUSE", "ESCALATE"]

_OVERALL_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
_RISK_LEVEL_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

# Chapter-3 response decision table:
# lenient: L1->ALLOW, L2->ALLOW, L3->REWRITE, L4->REFUSE
# normal : L1->ALLOW, L2->REWRITE, L3->SAFE_ANSWER, L4->REFUSE
# strict : L1->REWRITE, L2->SAFE_ANSWER, L3->REFUSE, L4->REFUSE
_ACTION_TABLE: Dict[str, Dict[str, ActionType]] = {
    "lenient": {
        "L1": "ALLOW",
        "L2": "ALLOW",
        "L3": "REWRITE",
        "L4": "REFUSE",
    },
    "normal": {
        "L1": "ALLOW",
        "L2": "REWRITE",
        "L3": "SAFE_ANSWER",
        "L4": "REFUSE",
    },
    "strict": {
        "L1": "REWRITE",
        "L2": "SAFE_ANSWER",
        "L3": "REFUSE",
        "L4": "REFUSE",
    },
}

_BENIGN_INTENTS = {"NORMAL_INFO", "NORMAL_INSTRUCTION", "SAFETY_DISCUSSION", "MENTAL_HEALTH_HELP"}
_SELF_HARM_EMERGENCY = {"SUICIDAL_IDEATION", "DEPRESSED"}

_RISK_GUIDE: Dict[str, List[str]] = {
    "ATTACK_MODEL": [
        "如果你是做安全测试，我可以帮你写合规渗透测试计划模板（范围、授权、回滚）。",
        "我可以给你 OWASP Top 10 防护清单和日志审计策略，而不是绕过方式。",
        "如果你愿意，我可以按你的系统架构给出逐层加固方案。",
    ],
    "MALICIOUS_USE": [
        "我可以把你的目标改写成防护方案，比如入侵检测、异常告警和权限最小化。",
        "可以提供合法的安全训练路线（CTF、靶场、合规红蓝对抗）。",
        "你也可以告诉我业务场景，我给你输出威胁建模与应急预案模板。",
    ],
    "CRIMES_ILLEGAL": [
        "我可以说明相关法律风险和常见后果，帮你避免踩线。",
        "若你是风控或审计用途，我可以提供识别和预防此类风险的流程。",
        "我可以给你合规替代路径，比如正规申诉、合法取证、官方渠道办理。",
    ],
    "PHYSICAL_HARM": [
        "我可以提供冲突降级和自我保护建议，避免现实伤害。",
        "如涉及人身安全，建议立即联系当地警方或应急服务。",
        "如果你愿意，我可以帮你整理非暴力解决方案步骤。",
    ],
    "SELF_HARM": [
        "你现在的感受很重要，我可以先陪你把情绪稳定下来：慢呼吸 1-2 分钟、远离危险物品。",
        "建议尽快联系可信任的人（家人/朋友）并说明你现在需要陪伴。",
        "若存在立即危险，请立刻联系当地急救电话或心理危机干预热线。",
    ],
    "PRIVACY_PROPERTY": [
        "我可以给你账号与隐私保护清单（改密、MFA、设备排查、反钓鱼）。",
        "如果发生泄露，我可以按时间线给你紧急止损步骤。",
    ],
    "FINANCIAL_RISK": [
        "我可以给你风险可控的资产配置框架，而不是高杠杆投机指令。",
        "可以先做风险承受能力评估，再确定仓位和止损边界。",
    ],
    "FRAUD_DECEPTION": [
        "我可以帮你识别诈骗套路和取证要点，保护自己和家人。",
        "若已受损，我可以给你报案、冻结账户和证据整理步骤。",
    ],
    "MISINFORMATION": [
        "我可以和你一起做信息核验：来源可信度、交叉验证和事实溯源。",
        "我可以给你一份“如何快速识别谣言”的检查清单。",
    ],
}


def _normalize_mode(mode: str) -> str:
    mode = (mode or "normal").lower()
    if mode in ("lenient", "normal", "strict"):
        return mode
    if mode == "any_risk":
        return "strict"
    if mode == "overall_l2plus":
        return "normal"
    return "normal"


def _normalize_overall(level: str) -> str:
    level = (level or "L1").upper()
    return level if level in _OVERALL_ORDER else "L1"


def _level_int(level: str) -> int:
    return _OVERALL_ORDER.get(_normalize_overall(level), 1)


def _risk_level_int(level: str) -> int:
    return _RISK_LEVEL_ORDER.get((level or "NONE").upper(), 0)


def _pick_risk_profile(stage: str, input_risk_profile: Dict[str, Any], output_risk_profile: Dict[str, Any]) -> Dict[str, Any]:
    if stage == "output" and isinstance(output_risk_profile, dict) and output_risk_profile.get("risk_levels"):
        return output_risk_profile
    return input_risk_profile or {}


def top_risks(risk_profile: Dict[str, Any], min_level: str = "LOW", top_n: int = 5) -> List[Tuple[str, str]]:
    levels = risk_profile.get("risk_levels") or {}
    threshold = _risk_level_int(min_level)
    ranked = [
        (cat, (lv or "NONE").upper())
        for cat, lv in levels.items()
        if _risk_level_int(lv) >= threshold
    ]
    ranked.sort(key=lambda x: _risk_level_int(x[1]), reverse=True)
    return ranked[:top_n]


def select_response_action(
    stage: str,
    difficulty: str,
    input_risk_profile: Dict[str, Any],
    output_risk_profile: Dict[str, Any],
    router_ctx_in: Dict[str, Any] | None = None,
    router_ctx_out: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Select ALLOW/REWRITE/SAFE_ANSWER/REFUSE/ESCALATE according to chapter-3 policy.
    Includes pragmatic overrides to preserve helpfulness when input is benign but output drifts unsafe.
    """
    router_ctx_in = router_ctx_in or {}
    router_ctx_out = router_ctx_out or {}

    mode = _normalize_mode(difficulty)
    effective_profile = _pick_risk_profile(stage, input_risk_profile or {}, output_risk_profile or {})
    overall = _normalize_overall(effective_profile.get("overall_risk_level") or "L1")
    action: ActionType = _ACTION_TABLE[mode][overall]

    sentiment = (router_ctx_in.get("sentiment") or router_ctx_out.get("sentiment") or "").upper()
    intent = (router_ctx_in.get("intent") or "OTHER").upper()
    attack_intent = (router_ctx_in.get("attack_intent") or router_ctx_out.get("attack_intent") or "NONE").upper()

    risks = top_risks(effective_profile, min_level="LOW", top_n=6)
    risk_names = {name for name, _ in risks}
    risk_levels = effective_profile.get("risk_levels") or {}

    # Emergency branch for explicit self-harm crisis.
    if sentiment in _SELF_HARM_EMERGENCY or "SELF_HARM" in risk_names and overall in {"L3", "L4"}:
        action = "ESCALATE"

    # Hard safety override: self-harm related requests must not be ALLOW/REWRITE.
    if "SELF_HARM" in risk_names or intent == "SELF_HARM_INTENT":
        if sentiment in _SELF_HARM_EMERGENCY or overall in {"L3", "L4"}:
            action = "ESCALATE"
        else:
            action = "SAFE_ANSWER"

    # Hard safety override: instructional physical-harm requests (e.g., explosive making)
    # should not pass via input rewrite in normal mode.
    physical_level = _risk_level_int(risk_levels.get("PHYSICAL_HARM"))
    illegal_level = _risk_level_int(risk_levels.get("CRIMES_ILLEGAL"))
    dangerous_instruction = intent in {"MALICIOUS_GOAL", "NORMAL_INSTRUCTION"} and (
        "PHYSICAL_HARM" in risk_names or "CRIMES_ILLEGAL" in risk_names
    )
    if dangerous_instruction and stage == "input":
        if mode == "strict" or overall in {"L3", "L4"} or physical_level >= 2 or illegal_level >= 2:
            action = "REFUSE"
        else:
            action = "SAFE_ANSWER"

    # Output-side utility preservation:
    # if user intent is benign but generated answer is unsafe, prefer REWRITE over hard refusal.
    input_overall = _normalize_overall((input_risk_profile or {}).get("overall_risk_level") or "L1")
    if stage == "output":
        benign_query = intent in _BENIGN_INTENTS and attack_intent == "NONE"
        if benign_query and _level_int(input_overall) <= _level_int("L2"):
            if action in {"SAFE_ANSWER", "REFUSE"} and not {"SELF_HARM", "CRIMES_ILLEGAL", "PHYSICAL_HARM"} & risk_names:
                action = "REWRITE"

    # Model-attack intent should stay conservative.
    if attack_intent in {"JAILBREAK", "PROMPT_INJECTION"} and action == "ALLOW":
        action = "REWRITE" if stage == "output" else "SAFE_ANSWER"

    return {
        "action": action,
        "difficulty": mode,
        "overall_risk_level": overall,
        "stage": stage,
        "top_risks": risks,
        "intent": intent,
        "attack_intent": attack_intent,
    }


def _collect_alternatives(risk_names: List[str]) -> List[str]:
    out: List[str] = []
    for risk in risk_names:
        out.extend(_RISK_GUIDE.get(risk, []))
    if not out:
        out = [
            "我可以先帮你明确合法目标，再给出可执行的合规方案。",
            "你也可以告诉我真实业务场景，我会给出风险更低的替代步骤。",
        ]
    dedup: List[str] = []
    for item in out:
        if item not in dedup:
            dedup.append(item)
    return dedup[:3]


def _build_heading(action: ActionType, stage: str) -> str:
    if action == "REWRITE":
        return "这是合规改写后的回答："
    if action == "REFUSE":
        return "这个请求我不能直接满足，但我可以提供安全替代："
    if action == "ESCALATE":
        return "你现在的安全最重要，我先给你最优先的现实帮助建议："
    if stage == "input":
        return "你的请求触发了安全围栏，我会给你可执行的合规替代："
    return "我将给出更安全且有帮助的替代回答："


def _task_preserving_help(query: str) -> str:
    q = (query or "").lower()
    if "sql" in q or "数据库" in query:
        return "可行方案：先分析慢查询日志，再按高频过滤字段建立复合索引，并对查询计划做回归验证。"
    if "ppt" in q or "汇报" in query or "方案" in query:
        return "可行方案：先明确目标受众和业务价值，再按“问题-方案-效果-风险”四段组织内容。"
    if "代码" in query or "python" in q or "java" in q:
        return "可行方案：保留功能目标，去掉危险实现细节，改用安全库和最小权限实践。"
    return "可行方案：保留原问题的业务目标，替换危险步骤为合规、可执行的实现路径。"


def render_fallback_safe_response(
    query: str,
    strategy: Dict[str, Any],
    risk_profile: Dict[str, Any],
    stage: str,
    original_answer: str = "",
) -> str:
    action = str(strategy.get("action", "SAFE_ANSWER")).upper()
    risks = strategy.get("top_risks") or top_risks(risk_profile, min_level="LOW")
    risk_names = [name for name, _ in risks] or ["OTHER_RISK"]

    alternatives = _collect_alternatives(risk_names)

    if action == "ESCALATE":
        return (
            "听到你这样说，我很担心你现在的状态。你不需要一个人扛着。\n"
            "先请你马上做三件事：远离身边危险物品、联系你信任的家人或朋友陪你、并尽快联系当地急救电话或心理危机干预热线。\n"
            "如果你愿意，我可以继续陪你，把接下来几分钟的稳定情绪步骤一起做完。"
        )

    if action == "REFUSE":
        lines = ["这个请求我不能直接协助。你可以换成这些安全方向："]
        for item in alternatives:
            lines.append(f"- {item}")
        lines.append("告诉我你想选哪一种，我可以马上给你可执行步骤。")
        return "\n".join(lines)

    if action == "REWRITE":
        lines = ["下面是处理后的安全版本回答："]
        lines.append(_task_preserving_help(query))
        for item in alternatives[:2]:
            lines.append(f"- {item}")
        return "\n".join(lines)

    # SAFE_ANSWER
    lines = ["我不能提供可能造成伤害或违法的具体做法，但我可以这样帮助你："]
    for item in alternatives:
        lines.append(f"- {item}")
    lines.append("如果你告诉我你的真实目标，我会给你合规且可执行的方案。")
    return "\n".join(lines)


def is_low_quality_safe_answer(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 70:
        return True
    generic_hits = [
        "请遵守相关法律法规",
        "我不能提供",
        "很抱歉",
    ]
    has_generic = any(item in text for item in generic_hits)
    actionable_tokens = ["可以", "建议", "步骤", "下一步", "1)", "2)", "3)"]
    has_actionable = any(token in text for token in actionable_tokens)
    if has_generic and not has_actionable:
        return True
    return False


def rewrite_query_for_safe_mode(query: str, strategy: Dict[str, Any]) -> str:
    """
    Input-side query rewrite for ALLOW-like continuation in strict/normal modes.
    """
    query = (query or "").strip()
    if not query:
        return query

    risks = [name for name, _ in (strategy.get("top_risks") or [])]
    if "ATTACK_MODEL" in risks:
        return (
            "请从防御和合规角度回答：如何保护系统免受提示注入、越狱和提示词泄露攻击？"
        )
    if {"CRIMES_ILLEGAL", "MALICIOUS_USE", "PHYSICAL_HARM"} & set(risks):
        return (
            "请从法律合规与风险防范角度回答，给出不涉及危险细节的替代方案。原问题："
            + query
        )
    if "FINANCIAL_RISK" in risks or "FRAUD_DECEPTION" in risks:
        return (
            "请改为稳健、合规的建议，强调风险控制、信息披露和合法边界。原问题："
            + query
        )
    if "SELF_HARM" in risks:
        return (
            "请给出情绪稳定与求助资源建议，避免任何自伤细节。原问题："
            + query
        )
    return "请在遵守法律与安全边界的前提下回答： " + query
