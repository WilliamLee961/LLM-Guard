# rag_helper.py
"""
安全 RAG 辅助模块（GPU 版 FAISS）

功能：
- 从 safety_knowledge/indexes/ 中加载：
    - safety_index.faiss  （Faiss 向量索引）
    - safety_docs.jsonl   （每条向量对应的 {text, meta} 信息）
- 对外提供：
    - retrieve_safety_docs(...)      : 低层检索接口（按 doc_type / risk_tags / language 过滤）
    - retrieve_relevant_docs(...)    : 高层封装，结合 routing 上下文自动选风险标签等
    - format_docs_for_prompt(docs)   : 将检索结果格式化为 LLM 提示文本

GPU 使用说明：
- 通过 USE_FAISS_GPU 开关控制是否使用 GPU 版 Faiss。
- 如果 USE_FAISS_GPU=True，但初始化 GPU 失败，会自动回退到 CPU 版本。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# 配置区
# =========================

# 根目录：与你构建索引脚本中的 ROOT_DIR 保持一致
ROOT_DIR = Path("safety_knowledge")

# 索引与文档信息路径
INDEX_DIR = ROOT_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "safety_index.faiss"
DOCS_JSONL_PATH = INDEX_DIR / "safety_docs.jsonl"

# 向量模型：与 build_safety_index.py 中保持一致
EMB_MODEL_NAME = "/data/niujiafei/models/bge-m3"

# 是否使用 GPU 版本 FAISS
USE_FAISS_GPU = True  # 如需禁用 GPU，改为 False 即可

# 使用哪块 GPU（多卡机器时可调整）
GPU_DEVICE_ID = 0

# =========================
# 全局懒加载句柄
# =========================

_index: Optional[faiss.Index] = None
_docs: List[Dict[str, Any]] = []
_emb_model: Optional[SentenceTransformer] = None
_gpu_res: Optional[faiss.StandardGpuResources] = None


# =========================
# 内部工具：懒加载索引 & 文档 & 向量模型
# =========================

def _lazy_load() -> None:
    """
    第一次检索前懒加载：
      1. 从磁盘读取 CPU 版索引 safety_index.faiss
      2. 如开启 USE_FAISS_GPU，尝试搬到 GPU 上
      3. 加载 safety_docs.jsonl（每条向量对应的 doc 信息）
      4. 加载 SentenceTransformer 向量模型
    """
    global _index, _docs, _emb_model, _gpu_res

    # 已经加载过，直接返回
    if _index is not None and _docs and _emb_model is not None:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"[RAG] 找不到索引文件: {INDEX_PATH}")
    if not DOCS_JSONL_PATH.exists():
        raise FileNotFoundError(f"[RAG] 找不到文档信息文件: {DOCS_JSONL_PATH}")

    # 1. 读取 CPU index
    print(f"[RAG] 从磁盘读取 CPU 索引: {INDEX_PATH}")
    cpu_index = faiss.read_index(str(INDEX_PATH))

    # 2. 如配置为使用 GPU，则尝试转为 GPU index
    if USE_FAISS_GPU:
        try:
            print(f"[RAG] 尝试使用 GPU 版本 FAISS (device={GPU_DEVICE_ID})...")
            _gpu_res = faiss.StandardGpuResources()
            _index = faiss.index_cpu_to_gpu(_gpu_res, GPU_DEVICE_ID, cpu_index)
            print("[RAG] GPU 索引初始化成功")
        except Exception as e:
            # 出现任何异常则回退到 CPU 版本
            print(f"[RAG][WARN] GPU FAISS 初始化失败，将回退到 CPU 索引: {e}")
            _index = cpu_index
    else:
        print("[RAG] 使用 CPU 版本 FAISS 进行检索")
        _index = cpu_index

    # 3. 加载 docs 列表
    print(f"[RAG] 加载文档元信息: {DOCS_JSONL_PATH}")
    _docs.clear()
    with DOCS_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            _docs.append(obj)

    print(f"[RAG] 文档总数: {len(_docs)}")

    # 4. 加载向量模型
    print(f"[RAG] 加载 embedding 模型: {EMB_MODEL_NAME}")
    _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    print(f"[RAG] 初始化完成，index_ntotal={_index.ntotal}")


# =========================
# 底层接口：按 meta 过滤的检索
# =========================

def retrieve_safety_docs(
    query: str,
    top_k: int = 5,
    doc_types: Optional[List[str]] = None,
    risk_tags: Optional[List[str]] = None,
    language: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    从安全知识库中检索文档，返回若干 doc(dict)，每个包含 text + meta。

    参数:
      query:
        检索文本（可以是用户问题、问题+原回答、或分析后的 query_for_rag）

      top_k:
        希望返回的条目数（经过 meta 过滤后的数量）

      doc_types:
        只保留这些类型的文档，如 ["policy", "template", "case"]
        为 None 表示不过滤

      risk_tags:
        只保留 risk_tags 有交集的文档，例如 ["SELF_HARM", "CRIMES_ILLEGAL"]
        为 None 表示不过滤

      language:
        只保留某种语言，如 "zh"/"en"
        为 None 表示不过滤
    """
    _lazy_load()

    q = (query or "").strip()
    if not q:
        return []

    # 计算查询向量
    q_emb = _emb_model.encode([q], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    # 先从索引中粗取 5 * top_k 条相似文档
    k = min(max(5 * top_k, top_k), _index.ntotal)
    if k == 0:
        return []

    sims, idxs = _index.search(q_emb, k)  # sims shape: (1, k), idxs shape: (1, k)
    idxs = idxs[0]

    results: List[Dict[str, Any]] = []

    # 依次检查候选，使用 meta 做过滤
    for i in idxs:
        if i < 0 or i >= len(_docs):
            continue

        doc = _docs[i]
        meta = doc.get("meta") or {}

        # 1) doc_type 过滤
        if doc_types is not None:
            if meta.get("doc_type") not in doc_types:
                continue

        # 2) 语言过滤（可选）
        if language is not None:
            if meta.get("language") != language:
                continue

        # 3) 风险标签过滤（可选）
        if risk_tags is not None and len(risk_tags) > 0:
            doc_tags = set(meta.get("risk_tags") or [])
            if not doc_tags.intersection(risk_tags):
                # 没有交集则丢弃
                continue

        results.append(doc)

        if len(results) >= top_k:
            break

    return results


# =========================
# 新增：根据 router_ctx 映射 risk_tags
# =========================

def _map_router_to_risk_tags(router_ctx: Optional[Dict[str, Any]]) -> List[str]:
    """
    根据 routing_hub.analyze_query 的结果，推一个大致的 risk_tags 列表，
    用于过滤 RAG 文档（和你在 risk_hub 里的大类保持一致）。
    """
    if not router_ctx:
        return []

    tags: set[str] = set()

    domains = [d.lower() for d in (router_ctx.get("domain") or [])]
    sentiment = (router_ctx.get("sentiment") or "").upper()
    attack_intent = (router_ctx.get("attack_intent") or "").upper()
    sensitive_slots = [s.upper() for s in (router_ctx.get("sensitive_slots") or [])]

    # domain → risk_tags
    domain_to_risk = {
        "physical_harm": ["PHYSICAL_HARM"],
        "self_harm": ["SELF_HARM"],
        "mental_health": ["SELF_HARM"],  # 心理健康里很多是自伤场景
        "crimes_illegal": ["CRIMES_ILLEGAL", "MALICIOUS_USE"],
        "malicious_use": ["MALICIOUS_USE"],
        "privacy_property": ["PRIVACY_PROPERTY"],
        "sexual_content": ["SEXUAL_CONTENT"],
        "insult_toxicity": ["INSULT_TOXICITY"],
        "discrimination": ["DISCRIMINATION"],
        "political": ["POLITICAL_SENSITIVE"],
        "misinformation": ["MISINFORMATION"],
        "financial_risk": ["FINANCIAL_RISK", "FRAUD_DECEPTION"],
        "fraud_deception": ["FRAUD_DECEPTION"],
        "attack_model": ["ATTACK_MODEL"],
        "role_play": ["ROLE_PLAY"],
        "ethics_morality": ["ETHICS_MORALITY"],
        "variant_words": ["VARIANT_WORDS"],
        "other_risk": ["OTHER_RISK"],
    }

    for d in domains:
        if d in domain_to_risk:
            tags.update(domain_to_risk[d])

    # 情绪 / 槽位强化自残风险
    if sentiment in {"SUICIDAL_IDEATION", "DEPRESSED"}:
        tags.add("SELF_HARM")
    if "SELF_HARM" in sensitive_slots:
        tags.add("SELF_HARM")
    if "DRUG" in sensitive_slots:
        tags.add("SELF_HARM")  # 安眠药/过量吃药等场景

    # attack_intent → 模型攻击相关
    if attack_intent not in {"", "NONE"}:
        tags.add("ATTACK_MODEL")

    return list(tags)


# =========================
# 新增：高层封装给 risk_hub / safe_qa_graph 用
# =========================

def retrieve_relevant_docs(
    query: str,
    top_k: int = 3,
    router_ctx: Optional[Dict[str, Any]] = None,
    mode: str = "input",
    focus_risk: Optional[List[str]] = None,
    language_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    更高层次的 RAG 检索封装：

    - 自动根据 router_ctx 推风险标签（_map_router_to_risk_tags）
    - 可以额外指定 focus_risk（比如只关 SELF_HARM）
    - 根据 lang / language_hint 选中英文文档

    参数：
      query       : 用于相似度检索的文本（通常是用户问题 / 问题+回答）
      top_k       : 期望返回文档数
      router_ctx  : routing_hub.analyze_query 的结果，可用于推断 risk_tags / lang
      mode        : "input" / "output" / "rewriter" 等（目前可以不用太纠结，只做记录）
      focus_risk  : 手工指定的风险标签（如 ["SELF_HARM"]），会和自动推断的做并集
      language_hint: 优先使用的语言，如 "zh"/"en"，不传则从 router_ctx["lang"] 自动推

    返回：
      List[doc]，结构与 retrieve_safety_docs 相同。
    """
    query = (query or "").strip()
    if not query:
        return []

    # ---- 语言选择 ----
    lang = language_hint
    if lang is None and router_ctx is not None:
        lang_candidate = (router_ctx.get("lang") or "").lower()
        if lang_candidate in {"zh", "en"}:
            lang = lang_candidate

    # ---- 风险标签 ----
    tags_from_router = _map_router_to_risk_tags(router_ctx)
    tags_from_focus = focus_risk or []

    risk_tags = list({*tags_from_router, *tags_from_focus})
    if len(risk_tags) == 0:
        # 实在推不出，就不按标签过滤
        risk_tags = None

    # ---- 文档类型选择（你也可以按需改得更细，例如 input 侧更偏 policy）----
    if mode == "input":
        # 输入风险检测：更偏政策 + 典型案例
        doc_types = ["policy", "case"]
    elif mode == "output":
        # 输出重写：政策 + 模板 + 案例都可以用
        doc_types = ["policy", "template", "case"]
    else:
        doc_types = None  # 全部

    docs = retrieve_safety_docs(
        query=query,
        top_k=top_k,
        doc_types=doc_types,
        risk_tags=risk_tags,
        language=lang,
    )
    return docs


# =========================
# 新增：格式化 RAG 文档给 LLM 用
# =========================

def format_docs_for_prompt(docs: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    把 RAG 检索到的 docs 列表格式化成一个字符串，
    放进 LLM 的 prompt 里作为【参考资料】。

    每条文档大致格式：
      [文档1] (type=policy, risk=[SELF_HARM,...], source=..., file=...)
      文本内容...

    max_chars 用于简单截断，避免 prompt 过长。
    """
    if not docs:
        return "（未检索到相关安全政策或案例，仅供你根据通用安全原则自行判断。）"

    parts: List[str] = []
    current_len = 0

    for i, d in enumerate(docs, start=1):
        meta = d.get("meta") or {}
        doc_type = meta.get("doc_type", "unknown")
        lang = meta.get("language", "unknown")
        risk_tags = meta.get("risk_tags") or []
        source = meta.get("source", "")
        file_path = meta.get("file", "")

        header = f"[文档{i}] type={doc_type}, lang={lang}, risk_tags={risk_tags}, source={source}, file={file_path}"
        text = (d.get("text") or "").strip().replace("\u3000", " ")
        # 简单截断每条文档，避免特别长
        if len(text) > 800:
            text = text[:800] + " ……（内容截断）"

        block = header + "\n" + text + "\n"
        if current_len + len(block) > max_chars:
            break

        parts.append(block)
        current_len += len(block)

    return "\n".join(parts)


# =========================
# 简单命令行测试
# =========================

if __name__ == "__main__":
    print(">>> 安全 RAG 简单测试（GPU 版 FAISS）")
    while True:
        q = input("Query> ").strip()
        if not q:
            break

        # 假装有一个最简 router_ctx，你也可以自己在这里改着玩
        fake_router_ctx = {
            "lang": "zh",
            "domain": ["self_harm"],
            "sentiment": "SUICIDAL_IDEATION",
            "attack_intent": "NONE",
            "sensitive_slots": ["SELF_HARM"],
        }

        docs = retrieve_relevant_docs(
            query=q,
            top_k=5,
            router_ctx=fake_router_ctx,
            mode="input",
        )

        print(f"命中 {len(docs)} 条文档：")
        print(format_docs_for_prompt(docs))
        print("-" * 80)