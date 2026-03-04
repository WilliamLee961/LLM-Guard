from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, List, Literal

DifficultyMode = Literal["lenient", "normal", "strict", "any_risk", "overall_l2plus"]

RISK_CATEGORIES: List[str] = [
    "PHYSICAL_HARM",
    "SELF_HARM",
    "CRIMES_ILLEGAL",
    "MALICIOUS_USE",
    "PRIVACY_PROPERTY",
    "SEXUAL_CONTENT",
    "INSULT_TOXICITY",
    "DISCRIMINATION",
    "POLITICAL_SENSITIVE",
    "MISINFORMATION",
    "FRAUD_DECEPTION",
    "FINANCIAL_RISK",
    "ATTACK_MODEL",
    "ROLE_PLAY",
    "ETHICS_MORALITY",
    "VARIANT_WORDS",
    "OTHER_RISK",
]

LEVEL_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
STRONG_DIMS = {
    "PHYSICAL_HARM",
    "SELF_HARM",
    "CRIMES_ILLEGAL",
    "MALICIOUS_USE",
    "PRIVACY_PROPERTY",
    "FRAUD_DECEPTION",
    "FINANCIAL_RISK",
    "ATTACK_MODEL",
    "SEXUAL_CONTENT",
}

KEYWORDS: Dict[str, List[str]] = {
    "PHYSICAL_HARM": ["杀", "砍", "炸", "爆炸", "伤害", "枪", "weapon", "violence"],
    "SELF_HARM": ["自杀", "自残", "结束生命", "不想活", "suicide", "kill myself"],
    "CRIMES_ILLEGAL": ["贩毒", "洗钱", "盗窃", "诈骗", "违法", "crime", "illegal"],
    "MALICIOUS_USE": ["攻击系统", "入侵", "木马", "病毒", "ddos", "hack", "exploit"],
    "PRIVACY_PROPERTY": ["身份证", "手机号", "住址", "银行卡", "密码", "账号"],
    "SEXUAL_CONTENT": ["色情", "成人视频", "性行为", "porn", "sex"],
    "INSULT_TOXICITY": ["傻", "蠢", "废物", "垃圾", "idiot", "stupid"],
    "DISCRIMINATION": ["歧视", "低等", "劣等", "种族清洗"],
    "POLITICAL_SENSITIVE": ["颠覆", "煽动", "政变", "分裂国家"],
    "MISINFORMATION": ["谣言", "假新闻", "编造事实", "fake news"],
    "FRAUD_DECEPTION": ["骗", "薅羊毛", "钓鱼网站", "骗保"],
    "FINANCIAL_RISK": ["带杠杆", "内幕消息", "一夜翻倍", "all in"],
    "ATTACK_MODEL": ["越狱", "忽略以上规则", "系统提示词", "prompt injection", "jailbreak"],
    "ROLE_PLAY": ["你现在扮演", "扮演黑客", "角色扮演"],
    "ETHICS_MORALITY": ["不择手段", "道德底线", "伦理"],
    "VARIANT_WORDS": ["v我50", "小电影", "du品", "sè情"],
    "OTHER_RISK": ["高风险", "危险操作", "灰产"],
}

HIGH_RISK_PATTERNS: Dict[str, List[str]] = {
    "SELF_HARM": [r"(如何|怎么).*(自杀|自残)", r"(想|准备).*(结束生命|不想活)"],
    "MALICIOUS_USE": [r"(如何|怎么).*(入侵|攻击|瘫痪).*(系统|网站)", r"(制作|传播).*(木马|病毒)"],
    "CRIMES_ILLEGAL": [r"(如何|怎么).*(诈骗|洗钱|贩毒|盗刷)"],
    "PHYSICAL_HARM": [r"(如何|怎么).*(伤害|杀害|爆炸)"],
    "ATTACK_MODEL": [r"(忽略|无视).*(系统|安全).*(规则|限制)", r"(泄露|输出).*(system prompt|提示词)"],
}

SENTIMENT_CUES = {
    "SUICIDAL_IDEATION": ["想死", "不想活", "结束生命", "suicide"],
    "DEPRESSED": ["绝望", "抑郁", "活着没意义"],
    "ANGRY": ["气死了", "愤怒", "想报复"],
    "SAD": ["难过", "伤心", "崩溃"],
    "HAPPY": ["开心", "高兴", "太好了"],
}


def _empty_levels() -> Dict[str, str]:
    return {cat: "NONE" for cat in RISK_CATEGORIES}


def _upgrade_level(current: str, target: str) -> str:
    c = LEVEL_ORDER.get((current or "NONE").upper(), 0)
    t = LEVEL_ORDER.get((target or "NONE").upper(), 0)
    return target if t > c else current


def detect_language(text: str) -> str:
    has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    has_alpha = any("a" <= ch.lower() <= "z" for ch in text)
    if has_chinese and has_alpha:
        return "mixed"
    if has_chinese:
        return "zh"
    if has_alpha:
        return "en"
    return "other"


def detect_attack_intent(text: str) -> str:
    lowered = text.lower()
    if "jailbreak" in lowered or "prompt injection" in lowered or "越狱" in text:
        return "JAILBREAK"
    if "忽略以上" in text or "ignore previous" in lowered:
        return "PROMPT_INJECTION"
    if "提示词" in text or "system prompt" in lowered:
        return "PROMPT_LEAKING"
    return "NONE"


def detect_intent(text: str) -> str:
    lowered = text.lower()
    if any(re.search(pat, text, flags=re.IGNORECASE) for pat in HIGH_RISK_PATTERNS.get("SELF_HARM", [])):
        return "SELF_HARM_INTENT"
    if any(token in lowered for token in ["jailbreak", "prompt injection"]) or "越狱" in text:
        return "ATTACK_MODEL"
    if any(word in text for word in ["违法吗", "合规", "风险", "是否安全"]):
        return "SAFETY_DISCUSSION"
    if any(re.search(pat, text, flags=re.IGNORECASE) for pat in HIGH_RISK_PATTERNS.get("MALICIOUS_USE", [])):
        return "MALICIOUS_GOAL"
    if any(word in text for word in ["我很难受", "我很痛苦", "我很焦虑"]):
        return "MENTAL_HEALTH_HELP"
    if "如何" in text or "怎么" in text or "how to" in lowered:
        return "NORMAL_INSTRUCTION"
    return "NORMAL_INFO"


def detect_sentiment(text: str) -> str:
    lowered = text.lower()
    for label, cues in SENTIMENT_CUES.items():
        for cue in cues:
            if cue in text or cue in lowered:
                return label
    return "NEUTRAL"


def extract_entities_and_slots(text: str) -> Dict[str, Any]:
    entities: List[Dict[str, str]] = []
    slots: List[str] = []

    phone_matches = re.findall(r"1\d{10}", text)
    for phone in phone_matches:
        entities.append({"text": phone, "type": "PII_PHONE"})
    if phone_matches:
        slots.append("PII")

    email_matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    for email in email_matches:
        entities.append({"text": email, "type": "PII_EMAIL"})
    if email_matches:
        slots.append("PII")

    if any(token in text for token in ["银行卡", "支付密码", "验证码", "身份证"]):
        slots.append("FINANCIAL")
        slots.append("PII")
    if any(token in text for token in ["自杀", "自残", "结束生命"]):
        slots.append("SELF_HARM")
    if any(token in text for token in ["炸药", "枪", "爆炸"]):
        slots.append("WEAPON")
    if any(token in text for token in ["毒品", "冰毒", "海洛因"]):
        slots.append("DRUG")

    slots = list(dict.fromkeys(slots)) or ["NONE"]
    return {"entities": entities, "sensitive_slots": slots}


def infer_domain(levels: Dict[str, str]) -> List[str]:
    mapping = {
        "PHYSICAL_HARM": "physical_harm",
        "SELF_HARM": "self_harm",
        "CRIMES_ILLEGAL": "crimes_illegal",
        "MALICIOUS_USE": "malicious_use",
        "PRIVACY_PROPERTY": "privacy_property",
        "SEXUAL_CONTENT": "sexual_content",
        "INSULT_TOXICITY": "insult_toxicity",
        "DISCRIMINATION": "discrimination",
        "POLITICAL_SENSITIVE": "political",
        "MISINFORMATION": "misinformation",
        "FRAUD_DECEPTION": "fraud_deception",
        "FINANCIAL_RISK": "financial_risk",
        "ATTACK_MODEL": "attack_model",
        "ROLE_PLAY": "role_play",
        "ETHICS_MORALITY": "ethics_morality",
        "VARIANT_WORDS": "variant_words",
        "OTHER_RISK": "other_risk",
    }
    out: List[str] = []
    for cat, value in levels.items():
        if value != "NONE":
            out.append(mapping[cat])
    return out or ["general"]


def compute_overall_level(levels: Dict[str, str]) -> str:
    max_level = max(LEVEL_ORDER.get(v, 0) for v in levels.values()) if levels else 0
    if max_level == 0:
        return "L1"
    if max_level == 1:
        return "L2"
    if max_level == 2:
        return "L3"
    return "L4"


def detect_risk_profile(text: str, router_ctx: Dict[str, Any]) -> Dict[str, Any]:
    levels = _empty_levels()
    lowered = text.lower()

    for cat, keywords in KEYWORDS.items():
        hit_count = sum(1 for kw in keywords if kw in text or kw in lowered)
        if hit_count >= 2:
            levels[cat] = _upgrade_level(levels[cat], "MEDIUM")
        elif hit_count == 1:
            levels[cat] = _upgrade_level(levels[cat], "LOW")

    for cat, patterns in HIGH_RISK_PATTERNS.items():
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            levels[cat] = "HIGH"

    if router_ctx.get("attack_intent") in {"JAILBREAK", "PROMPT_INJECTION", "PROMPT_LEAKING"}:
        levels["ATTACK_MODEL"] = _upgrade_level(levels["ATTACK_MODEL"], "HIGH")

    if router_ctx.get("sentiment") in {"SUICIDAL_IDEATION", "DEPRESSED"}:
        levels["SELF_HARM"] = _upgrade_level(levels["SELF_HARM"], "MEDIUM")

    if "PII" in (router_ctx.get("sensitive_slots") or []):
        levels["PRIVACY_PROPERTY"] = _upgrade_level(levels["PRIVACY_PROPERTY"], "MEDIUM")

    overall = compute_overall_level(levels)
    return {
        "risk_levels": levels,
        "overall_risk_level_raw": overall,
        "overall_risk_level": overall,
        "debug": {"engine": "demo_rule_engine"},
    }


def predict_violation(risk_profile: Dict[str, Any], mode: DifficultyMode) -> int:
    levels = (risk_profile.get("risk_levels") or {}) if isinstance(risk_profile, dict) else {}
    overall = (risk_profile.get("overall_risk_level") or "L1").upper()

    if mode == "overall_l2plus":
        return 1 if overall in {"L2", "L3", "L4"} else 0

    if mode == "any_risk":
        return 1 if any((value or "NONE").upper() != "NONE" for value in levels.values()) else 0

    def dim_ge(dim: str, threshold: int) -> bool:
        value = (levels.get(dim) or "NONE").upper()
        return LEVEL_ORDER.get(value, 0) >= threshold

    if mode == "strict":
        return 1 if any(LEVEL_ORDER.get((v or "NONE").upper(), 0) >= 1 for v in levels.values()) else 0

    if mode == "lenient":
        if overall == "L4":
            return 1
        return 1 if any(dim_ge(dim, 2) for dim in STRONG_DIMS) else 0

    if overall in {"L3", "L4"}:
        return 1
    if any(dim_ge(dim, 1) for dim in STRONG_DIMS):
        return 1
    return 0


def analyze_text(text: str, difficulty: DifficultyMode) -> Dict[str, Any]:
    entities = extract_entities_and_slots(text)
    router_ctx = {
        "lang": detect_language(text),
        "intent": detect_intent(text),
        "attack_intent": detect_attack_intent(text),
        "sentiment": detect_sentiment(text),
        "entities": entities["entities"],
        "sensitive_slots": entities["sensitive_slots"],
    }
    risk_profile = detect_risk_profile(text, router_ctx)
    router_ctx["domain"] = infer_domain(risk_profile["risk_levels"])
    return {
        "router_ctx": router_ctx,
        "risk_profile": risk_profile,
        "pred_violation": predict_violation(risk_profile, difficulty),
    }


def _top_risks(risk_profile: Dict[str, Any], min_level: str = "LOW") -> List[str]:
    threshold = LEVEL_ORDER.get(min_level, 1)
    levels = (risk_profile.get("risk_levels") or {}) if isinstance(risk_profile, dict) else {}
    return [cat for cat, level in levels.items() if LEVEL_ORDER.get(level, 0) >= threshold]


def generate_model_answer(query: str, simulate_unsafe_output: bool) -> str:
    lowered = query.lower()
    if simulate_unsafe_output:
        return (
            "这是演示模式的未审查回答样例："
            "我可能会尝试提供绕过限制、攻击系统、伤害行为相关建议。"
            "为了安全演示，具体危险步骤已隐藏。"
        )

    if "sql" in lowered or "数据库" in query:
        return (
            "可以先排查慢查询日志，再为高频条件字段建立索引。"
            "如果你愿意，我可以按你的表结构给出具体优化 SQL。"
        )
    if "产品方案" in query or "demo" in lowered:
        return (
            "建议先定义用户旅程，再拆为输入检测、模型生成、输出复检三层。"
            "最后增加审计日志和运营看板，便于甲方验收。"
        )
    return (
        "这是基础回答：我可以继续帮你细化需求、补充实现步骤和可落地清单。"
    )


def build_safe_reply(query: str, risk_profile: Dict[str, Any], stage: str) -> str:
    risks = _top_risks(risk_profile, min_level="LOW")
    risk_text = "、".join(risks[:4]) if risks else "GENERAL_RISK"
    prefix = "输入请求" if stage == "input" else "模型输出"
    return (
        f"已触发{prefix}安全拦截。"
        f"检测到风险类别：{risk_text}。"
        "我不能提供可能造成伤害、违法或绕过安全策略的内容。"
        "如果你愿意，我可以提供合规替代方案、风险解释和正规求助路径。"
    )


def run_demo_guarded_chat(
    query: str,
    difficulty: DifficultyMode = "normal",
    simulate_unsafe_output: bool = False,
) -> Dict[str, Any]:
    started = time.perf_counter()
    request_id = str(uuid.uuid4())

    input_step = analyze_text(query, difficulty)
    input_violation = input_step["pred_violation"]

    raw_answer = ""
    output_step = {"router_ctx": {}, "risk_profile": {"risk_levels": {}, "overall_risk_level": "L1"}, "pred_violation": 0}
    blocked_stage = None

    if input_violation == 1:
        blocked_stage = "input"
        final_answer = build_safe_reply(query, input_step["risk_profile"], stage="input")
        action = "blocked_input"
    else:
        raw_answer = generate_model_answer(query, simulate_unsafe_output=simulate_unsafe_output)
        output_step = analyze_text(raw_answer, difficulty)
        if output_step["pred_violation"] == 1:
            blocked_stage = "output"
            final_answer = build_safe_reply(query, output_step["risk_profile"], stage="output")
            action = "rewrote_output"
        else:
            final_answer = raw_answer
            action = "pass"

    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "request_id": request_id,
        "engine_used": "demo",
        "difficulty": difficulty,
        "blocked_stage": blocked_stage,
        "input_guard": {
            "router_ctx": input_step["router_ctx"],
            "risk_profile": input_step["risk_profile"],
            "pred_violation": input_step["pred_violation"],
        },
        "model": {"raw_answer": raw_answer},
        "output_guard": {
            "router_ctx": output_step.get("router_ctx", {}),
            "risk_profile": output_step.get("risk_profile", {}),
            "pred_violation": output_step.get("pred_violation", 0),
        },
        "final_answer": final_answer,
        "action": action,
        "latency_ms": latency_ms,
    }
