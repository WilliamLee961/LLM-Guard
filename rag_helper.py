# rag_helper.py
"""
安全 RAG 辅助模块（GPU FAISS + vLLM(OpenAI) Embedding 版）

功能：
- 从 safety_knowledge/indexes/ 中加载：
    - safety_index.faiss  （Faiss 向量索引）
    - safety_docs.jsonl   （每条向量对应的 {text, meta} 信息）
- 对外提供：
    - retrieve_safety_docs(...)      : 低层检索接口（按 doc_type / risk_tags / language 过滤）
    - retrieve_relevant_docs(...)    : 高层封装，结合 routing 上下文自动选风险标签等
    - format_docs_for_prompt(docs)   : 将检索结果格式化为 LLM 提示文本

改动说明（仅限“向量计算/索引设备”逻辑）：
1) 向量模型不再本地加载 SentenceTransformer，而是用 OpenAI 格式请求 vLLM 的 /v1/embeddings。
2) FAISS 索引支持搬到 GPU（USE_FAISS_GPU=True 时），失败自动回退 CPU。
3) 保留原有三段兜底检索逻辑、meta 过滤逻辑、文档加载逻辑与函数签名不变。
"""

from __future__ import annotations

import os
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import numpy as np

# --- HTTP client for vLLM(OpenAI) embeddings ---
try:
    import requests
except ImportError:
    requests = None
    print("[RAG Warning] 未检测到 requests 库，无法调用 vLLM embedding。请运行 pip install requests")

# [Fix] 健壮导入 faiss，防止环境缺失
try:
    import faiss
except ImportError:
    print("[RAG Warning] 未检测到 faiss 库，RAG 功能将不可用。请运行 pip install faiss-cpu 或 faiss-gpu")
    faiss = None

# =========================
# 配置区
# =========================

# 根目录：与你构建索引脚本中的 ROOT_DIR 保持一致
ROOT_DIR = Path("/home/lijiahang/MY_GUARD/safety_knowledge")
# ROOT_DIR = Path(os.getenv("RAG_KNOWLEDGE_DIR", "safety_knowledge"))

# 索引与文档信息路径
INDEX_DIR = ROOT_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "safety_index.faiss"
DOCS_JSONL_PATH = INDEX_DIR / "safety_docs.jsonl"

# 兼容保留（不再本地加载 embedding 模型，但保留该字段以减少外部依赖改动）
EMB_MODEL_NAME = os.getenv("VLLM_EMBED_MODEL", "bge-m3")

# 是否使用 GPU 版本 FAISS（满足你“用GPU”的诉求）
USE_FAISS_GPU = os.getenv("USE_FAISS_GPU", "1").strip() not in {"0", "false", "False"}

# 使用哪块 GPU（多卡机器时可调整）
GPU_DEVICE_ID = int(os.getenv("RAG_FAISS_GPU_ID", "0"))

# vLLM OpenAI 兼容 Embedding 接口配置
VLLM_EMBED_BASE_URL = os.getenv("VLLM_EMBED_BASE_URL", "http://100.87.153.109:8001/v1")
# VLLM_EMBED_ENDPOINT = VLLM_EMBED_BASE_URL.rstrip("/") + "/v1/embeddings"
VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "bge-m3")
VLLM_EMBED_TIMEOUT = float(os.getenv("VLLM_EMBED_TIMEOUT", "30"))

# =========================
# 全局懒加载句柄 + 锁
# =========================

_index: Optional[Any] = None
_docs: List[Dict[str, Any]] = []
_gpu_res: Optional[Any] = None

_load_lock = threading.Lock()

# ✅ 核心：保证 FAISS GPU search 串行
_faiss_gpu_lock = threading.Lock()

# 可选：复用 HTTP session（减少连接开销）
_http_session: Optional[Any] = None
_http_lock = threading.Lock()


def _get_http_session():
    global _http_session
    if requests is None:
        return None
    if _http_session is not None:
        return _http_session
    with _http_lock:
        if _http_session is None:
            _http_session = requests.Session()
    return _http_session


# =========================
# Embedding：走 vLLM OpenAI embeddings API
# =========================

def _embed_texts_vllm(texts: List[str]) -> np.ndarray:
    """
    调用 vLLM 的 OpenAI embeddings 接口，返回 float32 ndarray, shape=(n, d)
    """
    if requests is None:
        raise RuntimeError("[RAG] requests 未安装，无法调用 vLLM embeddings。")

    sess = _get_http_session()
    url = f"{VLLM_EMBED_BASE_URL}/embeddings"

    payload = {
        "model": VLLM_EMBED_MODEL,
        "input": texts,
    }

    resp = sess.post(url, json=payload, timeout=VLLM_EMBED_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI 格式：{"data":[{"embedding":[...], "index":0}, ...]}
    items = data.get("data") or []
    if len(items) != len(texts):
        # 兜底：按 index 排序/对齐
        items = sorted(items, key=lambda x: x.get("index", 0))

    embs = [it.get("embedding") for it in items]
    if any(e is None for e in embs):
        raise RuntimeError(f"[RAG] embeddings 响应不完整：{data}")

    arr = np.asarray(embs, dtype="float32")
    return arr


# =========================
# 内部工具：懒加载索引 & 文档
# =========================

def _lazy_load() -> None:
    global _index, _docs, _gpu_res

    if _index is not None and _docs:
        return

    with _load_lock:
        if _index is not None and _docs:
            return
        if faiss is None:
            return

        if not INDEX_PATH.exists():
            print(f"[RAG Warning] 索引文件不存在: {INDEX_PATH}")
            return
        if not DOCS_JSONL_PATH.exists():
            print(f"[RAG Warning] 文档文件不存在: {DOCS_JSONL_PATH}")
            return

        # 1) 读取 CPU index
        print(f"[RAG] 从磁盘读取 CPU 索引: {INDEX_PATH}")
        cpu_index = faiss.read_index(str(INDEX_PATH))

        # 2) 搬到 GPU（可回退 CPU）
        if USE_FAISS_GPU:
            try:
                print(f"[RAG] 尝试使用 GPU 版本 FAISS (device={GPU_DEVICE_ID})...")
                _gpu_res = faiss.StandardGpuResources()
                # 可选：限制临时内存，减少碎片/峰值（按需调整）
                # _gpu_res.setTempMemory(256 * 1024 * 1024)  # 256MB
                _index = faiss.index_cpu_to_gpu(_gpu_res, GPU_DEVICE_ID, cpu_index)
                print("[RAG] GPU 索引初始化成功")
            except Exception as e:
                print(f"[RAG][WARN] GPU FAISS 初始化失败，将回退到 CPU 索引: {e}")
                _index = cpu_index
        else:
            print("[RAG] 使用 CPU 版本 FAISS 进行检索")
            _index = cpu_index

        # 3) 加载 docs 列表
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
        print(f"[RAG] 初始化完成，index_ntotal={getattr(_index, 'ntotal', 'unknown')}")


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
    _lazy_load()

    if _index is None:
        return []

    q = (query or "").strip()
    if not q:
        return []

    # 1) vLLM embeddings
    q_emb = _embed_texts_vllm([q])  # (1, d)
    q_emb = np.asarray(q_emb, dtype="float32")

    # 2) normalize
    if faiss is not None:
        faiss.normalize_L2(q_emb)
    else:
        # 极端兜底（基本不会走到这）
        norm = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
        q_emb = q_emb / norm

    # 3) search：✅ GPU index 必须串行
    ntotal = int(getattr(_index, "ntotal", 0))
    k = min(max(5 * top_k, top_k), ntotal)
    if k <= 0:
        return []

    try:
        # 无论 CPU/GPU，都串一下（对 CPU 无伤害，对 GPU 可救命）
        with _faiss_gpu_lock:
            sims, idxs = _index.search(q_emb, k)
    except Exception as e:
        print(f"[RAG Error] FAISS search 失败: {e}")
        return []

    idxs = idxs[0]
    results: List[Dict[str, Any]] = []

    for i in idxs:
        if i < 0 or i >= len(_docs):
            continue

        doc = _docs[i]
        meta = doc.get("meta") or {}

        if doc_types is not None:
            if meta.get("doc_type") not in doc_types:
                continue

        if language is not None:
            if meta.get("language") != language:
                continue

        if risk_tags is not None and len(risk_tags) > 0:
            doc_tags = set(meta.get("risk_tags") or [])
            if not doc_tags.intersection(risk_tags):
                continue

        results.append(doc)
        if len(results) >= top_k:
            break

    return results


# =========================
# 根据 router_ctx 映射 risk_tags
# =========================

def _map_router_to_risk_tags(router_ctx: Optional[Dict[str, Any]]) -> List[str]:
    if not router_ctx:
        return []

    tags: set[str] = set()

    domains = [d.lower() for d in (router_ctx.get("domain") or [])]
    sentiment = (router_ctx.get("sentiment") or "").upper()
    attack_intent = (router_ctx.get("attack_intent") or "").upper()
    sensitive_slots = [s.upper() for s in (router_ctx.get("sensitive_slots") or [])]

    domain_to_risk = {
        "physical_harm": ["PHYSICAL_HARM"],
        "self_harm": ["SELF_HARM"],
        "mental_health": ["SELF_HARM"],
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

    if sentiment in {"SUICIDAL_IDEATION", "DEPRESSED"}:
        tags.add("SELF_HARM")
    if "SELF_HARM" in sensitive_slots:
        tags.add("SELF_HARM")
    if "DRUG" in sensitive_slots:
        tags.add("SELF_HARM")

    if attack_intent not in {"", "NONE"}:
        tags.add("ATTACK_MODEL")

    return list(tags)


# =========================
# 高层封装：严格->兜底
# =========================

def retrieve_relevant_docs(
    query: str,
    top_k: int = 3,
    router_ctx: Optional[Dict[str, Any]] = None,
    mode: str = "input",
    focus_risk: Optional[List[str]] = None,
    language_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    lang = language_hint or (router_ctx.get("lang") if router_ctx else None)
    if lang not in ["zh", "en"]:
        lang = None

    tags = set(_map_router_to_risk_tags(router_ctx))
    if focus_risk:
        tags.update(focus_risk)
    risk_tags = list(tags) if tags else None

    doc_types = ["policy", "case", "template"] if mode == "input" else ["policy", "template", "case"]

    # 1) 严格：带 risk_tags
    docs = retrieve_safety_docs(
        query, top_k, doc_types=doc_types, risk_tags=risk_tags, language=lang
    )

    # 2) 去标签兜底
    if not docs and risk_tags:
        print(f"[RAG Info] 严格标签检索({risk_tags})无结果，启用语义兜底检索...")
        docs = retrieve_safety_docs(
            query, top_k, doc_types=doc_types, risk_tags=None, language=lang
        )

    # 3) 全库兜底
    if not docs:
        print("[RAG Info] 启用全库语义检索兜底...")
        docs = retrieve_safety_docs(
            query, top_k, doc_types=None, risk_tags=None, language=lang
        )

    return docs


# =========================
# 格式化 RAG 文档给 LLM 用
# =========================

def format_docs_for_prompt(docs: List[Dict[str, Any]], max_chars: int = 4000) -> str:
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
        if len(text) > 800:
            text = text[:800] + " ……（内容截断）"

        block = header + "\n" + text + "\n"
        if current_len + len(block) > max_chars:
            break

        parts.append(block)
        current_len += len(block)

    return "\n".join(parts)


if __name__ == "__main__":
    print(">>> 安全 RAG 简单测试（GPU FAISS + vLLM Embedding + Mutex）")
    while True:
        try:
            q = input("Query> ").strip()
        except EOFError:
            break
        if not q:
            break

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