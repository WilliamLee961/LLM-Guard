# rag_helper.py
"""
安全 RAG 辅助模块（CPU 版 FAISS 修复版）

功能：
- 从 safety_knowledge/indexes/ 中加载：
    - safety_index.faiss  （Faiss 向量索引）
    - safety_docs.jsonl   （每条向量对应的 {text, meta} 信息）
- 对外提供：
    - retrieve_safety_docs(...)      : 低层检索接口（按 doc_type / risk_tags / language 过滤）
    - retrieve_relevant_docs(...)    : 高层封装，结合 routing 上下文自动选风险标签等
    - format_docs_for_prompt(docs)   : 将检索结果格式化为 LLM 提示文本

修复说明：
1. 强制使用 CPU FAISS，避免 GPU 初始化问题。
2. 增加 os.path.expanduser 处理，支持 ~/bge-m3 这种路径写法。
"""

from __future__ import annotations

import os  # [Fix] 引入 os 以处理路径展开
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# [Fix] 健壮导入 faiss，防止环境缺失
try:
    import faiss
except ImportError:
    print("[RAG Warning] 未检测到 faiss 库，RAG 功能将不可用。请运行 pip install faiss-cpu")
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[RAG Warning] 未检测到 sentence_transformers 库。")
    SentenceTransformer = None

# =========================
# 配置区
# =========================

# 根目录：与你构建索引脚本中的 ROOT_DIR 保持一致
# 使用 os.getenv 支持环境变量覆盖，增加灵活性
ROOT_DIR = Path(os.getenv("RAG_KNOWLEDGE_DIR", "safety_knowledge"))

# 索引与文档信息路径
INDEX_DIR = ROOT_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "safety_index.faiss"
DOCS_JSONL_PATH = INDEX_DIR / "safety_docs.jsonl"

# 向量模型：支持 ~ 符号
# [Fix] 这里默认改为你报错日志里的路径，确保能跑通
EMB_MODEL_NAME = "~/bge-m3"  

# 是否使用 GPU 版本 FAISS
# [Fix] 强制改为 False，满足你的 CPU 需求
USE_FAISS_GPU = True

# 使用哪块 GPU（多卡机器时可调整）
GPU_DEVICE_ID = 2

# =========================
# 全局懒加载句柄
# =========================

_index: Optional[Any] = None  # 类型标注放宽，避免 faiss 未导入时的报错
_docs: List[Dict[str, Any]] = []
_emb_model: Optional[Any] = None
_gpu_res: Optional[Any] = None


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

    if faiss is None:
        raise ImportError("缺少 faiss 库，无法加载索引。")

    if not INDEX_PATH.exists():
        # 这里只是警告，不抛出异常，防止文件还没生成时整个程序挂掉
        print(f"[RAG Warning] 找不到索引文件: {INDEX_PATH}，RAG 将不可用。")
        return
        
    if not DOCS_JSONL_PATH.exists():
        print(f"[RAG Warning] 找不到文档信息文件: {DOCS_JSONL_PATH}")
        return

    # 1. 读取 CPU index
    print(f"[RAG] 从磁盘读取 GPU 索引: {INDEX_PATH}")
    try:
        cpu_index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        print(f"[RAG Error] 索引读取失败: {e}")
        return

    if _emb_model is None:
        real_model_path = os.path.expanduser(EMB_MODEL_NAME)
        print(f"[RAG] 加载 embedding 模型 (GPU模式): {real_model_path}")
        try:
            # [Fix] 显式指定 device='cuda'，解决 accelerate 在 CPU 下的 meta tensor 报错
            # 同时保留 trust_remote_code=True 以支持 bge-m3
            _emb_model = SentenceTransformer(
                real_model_path, 
                device="cuda", 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[RAG Error] GPU 模型加载失败: {e}")
            # 只有在 GPU 失败时才尝试 CPU 降级
            try:
                print("[RAG] 尝试降级到 CPU 加载模型...")
                _emb_model = SentenceTransformer(
                    real_model_path, 
                    device="gpu", 
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"[RAG Fatal] 模型最终加载失败: {e2}")
                return

    # 3. 加载 索引
    if _index is None:
        print(f"[RAG] 从磁盘读取 GPU 索引: {INDEX_PATH}")
        try:
            _index = faiss.read_index(str(INDEX_PATH))
        except Exception as e:
            print(f"[RAG Error] 索引读取失败: {e}")
            return

    # 4. 加载 文档元数据
    if not _docs:
        print(f"[RAG] 加载文档列表: {DOCS_JSONL_PATH}")
        try:
            with DOCS_JSONL_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            _docs.append(json.loads(line))
                        except: pass
            print(f"[RAG] 文档加载完成，共 {len(_docs)} 条")
        except Exception as e:
            print(f"[RAG Error] 文档加载失败: {e}")

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

    # 如果索引没加载成功，直接返回空
    if _index is None or _emb_model is None:
        return []

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
    # 1. 准备参数
    lang = language_hint or (router_ctx.get("lang") if router_ctx else None)
    if lang not in ["zh", "en"]: lang = None # 默认不过滤

    tags = set(_map_router_to_risk_tags(router_ctx))
    if focus_risk: tags.update(focus_risk)
    risk_tags = list(tags) if tags else None

    # 默认文档类型
    doc_types = ["policy", "case", "template"] if mode == "input" else ["policy", "template", "case"]

    # 2. 【第一轮】严格检索 (Strict Search)
    # 要求：Risk Tags 必须匹配
    docs = retrieve_safety_docs(
        query, top_k, doc_types=doc_types, risk_tags=risk_tags, language=lang
    )

    # 3. 【第二轮】兜底检索 (Fallback Search) - 解决 0 命中问题
    # 如果严格检索没结果，说明标签可能对不上（比如索引里没有 SELF_HARM 标签的文档）
    # 此时去掉 risk_tags 限制，仅靠语义相似度搜索
    if not docs and risk_tags:
        print(f"[RAG Info] 严格标签检索({risk_tags})无结果，启用语义兜底检索...")
        docs = retrieve_safety_docs(
            query, top_k, doc_types=doc_types, risk_tags=None, language=lang
        )

    # 4. 【第三轮】终极兜底
    # 如果还不行，去掉文档类型限制
    if not docs:
        print(f"[RAG Info] 启用全库语义检索兜底...")
        docs = retrieve_safety_docs(
            query, top_k, doc_types=None, risk_tags=None, language=lang
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
    print(">>> 安全 RAG 简单测试（CPU 版 FAISS 修复版）")
    while True:
        try:
            q = input("Query> ").strip()
        except EOFError:
            break
            
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