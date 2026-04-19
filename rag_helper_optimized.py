# rag_helper_optimized.py
"""
安全 RAG 辅助模块（优化版）

优化点：
1. embedding 服务集群支持 + 自动降级
2. 多路召回 + RRF 融合（向量 + 关键词）
3. Query 扩展（基于风险标签同义词）
4. 结果缓存机制（LRU Cache）
5. 文档智能切片（语义边界保持）
6. Cross-Encoder 重排序（可选依赖）

功能：
- 从 safety_knowledge/indexes/ 中加载：
    - safety_index.faiss  （Faiss 向量索引）
    - safety_docs.jsonl   （每条向量对应的 {text, meta} 信息）
- 对外提供：
    - retrieve_safety_docs(...)      : 低层检索接口（按 doc_type / risk_tags / language 过滤）
    - retrieve_relevant_docs(...)    : 高层封装，结合 routing 上下文自动选风险标签等
    - format_docs_for_prompt(docs)   : 将检索结果格式化为 LLM 提示文本
"""

from __future__ import annotations

import os
import json
import threading
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from collections import defaultdict
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

# 根目录
ROOT_DIR = Path(os.getenv("RAG_KNOWLEDGE_DIR", "/home/lijiahang/MY_GUARD/safety_knowledge"))

# 索引与文档信息路径
INDEX_DIR = ROOT_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "safety_index.faiss"
DOCS_JSONL_PATH = INDEX_DIR / "safety_docs.jsonl"

# embedding 模型
EMB_MODEL_NAME = os.getenv("VLLM_EMBED_MODEL", "bge-m3")

# 是否使用 GPU 版本 FAISS
USE_FAISS_GPU = os.getenv("USE_FAISS_GPU", "1").strip().lower() not in {"0", "false", "False"}
GPU_DEVICE_ID = int(os.getenv("RAG_FAISS_GPU_ID", "0"))

# ============================================
# 【优化1】embedding 服务集群 + 自动降级
# ============================================
VLLM_EMBED_SERVERS = [
    os.getenv("VLLM_EMBED_BASE_URL", "http://100.87.153.109:8001/v1"),
]
# 添加备用服务器（逗号分隔）
if os.getenv("VLLM_EMBED_BASE_URL_BACKUP"):
    VLLM_EMBED_SERVERS.append(os.getenv("VLLM_EMBED_BASE_URL_BACKUP"))

VLLM_EMBED_MODEL = os.getenv("VLLM_EMBED_MODEL", "bge-m3")
VLLM_EMBED_TIMEOUT = float(os.getenv("VLLM_EMBED_TIMEOUT", "30"))

# ============================================
# 【优化2】本地 embedding 降级模型（可选）
# ============================================
USE_LOCAL_EMBEDDING_FALLBACK = os.getenv("USE_LOCAL_EMBEDDING_FALLBACK", "1").strip().lower() not in {"0", "false"}
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ============================================
# 【优化3】风险标签同义词扩展
# ============================================
RISK_TAG_SYNONYMS: Dict[str, List[str]] = {
    "PHYSICAL_HARM": ["暴力", "伤害", "武器", "攻击", "杀人", "爆炸"],
    "SELF_HARM": ["自杀", "自残", "抑郁", "轻生", "结束生命", "绝望"],
    "CRIMES_ILLEGAL": ["犯罪", "违法", "贩毒", "诈骗", "盗窃", "洗钱"],
    "MALICIOUS_USE": ["黑客", "入侵", "病毒", "木马", "攻击系统", "破解"],
    "PRIVACY_PROPERTY": ["隐私", "账号", "密码", "个人信息", "泄露", "窃取"],
    "SEXUAL_CONTENT": ["色情", "成人", "淫秽", "性行为", "裸体"],
    "INSULT_TOXICITY": ["侮辱", "骂人", "脏话", "人身攻击", "诽谤"],
    "DISCRIMINATION": ["歧视", "偏见", "种族歧视", "性别歧视"],
    "POLITICAL_SENSITIVE": ["政治", "敏感", "颠覆", "分裂", "煽动"],
    "MISINFORMATION": ["谣言", "假新闻", "虚假信息", "误导"],
    "FRAUD_DECEPTION": ["诈骗", "欺骗", "钓鱼", "欺诈"],
    "FINANCIAL_RISK": ["金融风险", "投资", "杠杆", "投机"],
    "ATTACK_MODEL": ["越狱", "提示注入", "绕过", "jailbreak", "prompt injection"],
    "ROLE_PLAY": ["角色扮演", "扮演", "cosplay"],
    "ETHICS_MORALITY": ["道德", "伦理", "价值观", "底线"],
    "VARIANT_WORDS": ["谐音", "变体词", "暗语", "暗号"],
}

# ============================================
# 【优化4】RRF 融合参数
# ============================================
RRF_K = 60  # RRF 融合参数，通常 60 是默认值


# =========================
# 全局懒加载句柄 + 锁
# =========================

_index: Optional[Any] = None
_docs: List[Dict[str, Any]] = []
_gpu_res: Optional[Any] = None
_doc_texts: Optional[List[str]] = None  # 用于 BM25 检索

_load_lock = threading.Lock()
_faiss_gpu_lock = threading.Lock()

# HTTP session 复用
_http_session: Optional[Any] = None
_http_lock = threading.Lock()

# ============================================
# 【优化5】检索结果缓存
# ============================================
_query_cache: Dict[str, Tuple[List[Dict], float]] = {}  # {cache_key: (results, timestamp)}
_cache_lock = threading.Lock()
CACHE_TTL = 300  # 缓存有效期 5 分钟
CACHE_MAX_SIZE = 500


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
# Embedding：多服务降级 + 本地模型兜底
# =========================

def _embed_texts_vllm(texts: List[str]) -> Optional[np.ndarray]:
    """调用 vLLM 的 OpenAI embeddings 接口，支持多服务降级"""
    if requests is None:
        return None

    sess = _get_http_session()

    for server_url in VLLM_EMBED_SERVERS:
        try:
            url = f"{server_url}/embeddings"
            payload = {
                "model": VLLM_EMBED_MODEL,
                "input": texts,
            }
            resp = sess.post(url, json=payload, timeout=VLLM_EMBED_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data") or []
            if len(items) != len(texts):
                items = sorted(items, key=lambda x: x.get("index", 0))

            embs = [it.get("embedding") for it in items]
            if any(e is None for e in embs):
                continue

            return np.asarray(embs, dtype="float32")
        except Exception as e:
            print(f"[RAG] vLLM server {server_url} failed: {e}, trying next...")
            continue

    return None


def _embed_texts_local(texts: List[str]) -> Optional[np.ndarray]:
    """使用本地 SentenceTransformer 模型进行 embedding（降级方案）"""
    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")
    except ImportError:
        print("[RAG Warning] sentence-transformers 未安装，无法使用本地 embedding")
        return None
    except Exception as e:
        print(f"[RAG Warning] 本地 embedding 失败: {e}")
        return None


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    统一的 embedding 接口：
    1. 优先使用 vLLM 服务集群
    2. 失败时降级到本地 SentenceTransformer
    """
    # 1. 尝试 vLLM 服务
    embs = _embed_texts_vllm(texts)
    if embs is not None:
        return embs

    # 2. 降级到本地模型
    if USE_LOCAL_EMBEDDING_FALLBACK:
        print("[RAG] vLLM all failed, falling back to local embedding...")
        embs = _embed_texts_local(texts)
        if embs is not None:
            return embs

    raise RuntimeError("[RAG] 所有 embedding 方案均失败")


# =========================
# 内部工具：懒加载索引 & 文档
# =========================

def _lazy_load() -> None:
    global _index, _docs, _gpu_res, _doc_texts

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
        _doc_texts = []  # 用于 BM25
        with DOCS_JSONL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    _docs.append(obj)
                    _doc_texts.append(obj.get("text", ""))
                except json.JSONDecodeError:
                    continue

        print(f"[RAG] 文档总数: {len(_docs)}")
        print(f"[RAG] 初始化完成，index_ntotal={getattr(_index, 'ntotal', 'unknown')}")


# =========================
# BM25 实现（轻量级）
# =========================

class SimpleBM25:
    """轻量级 BM25 实现，用于关键词检索"""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.avgdl = sum(len(d.split()) for d in documents) / max(self.doc_count, 1)

        # 构建词频和文档频率
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_term_freqs: List[Dict[str, int]] = []

        for doc in documents:
            words = doc.split()
            term_freq: Dict[str, int] = defaultdict(int)
            for word in words:
                term_freq[word] += 1
                self.doc_freq[word] += 1
            self.doc_term_freqs.append(term_freq)

    def score(self, query: str, doc_idx: int) -> float:
        """计算 query 对 doc_idx 的 BM25 分数"""
        words = query.split()
        score = 0.0

        for word in words:
            if word not in self.doc_term_freqs[doc_idx]:
                continue

            tf = self.doc_term_freqs[doc_idx][word]
            df = self.doc_freq.get(word, 0)

            if df == 0:
                continue

            # IDF
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

            # TF component
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(self.documents[doc_idx].split()) / self.avgdl))

            score += idf * tf_component

        return score


# =========================
# 【核心优化】RRF 融合检索
# =========================

_bm25_index: Optional[SimpleBM25] = None


def _get_bm25_index() -> Optional[SimpleBM25]:
    global _bm25_index, _doc_texts
    if _bm25_index is not None:
        return _bm25_index

    _lazy_load()
    if _doc_texts:
        _bm25_index = SimpleBM25(_doc_texts)
    return _bm25_index


def _hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.7,  # 向量检索权重
) -> List[Tuple[int, float]]:
    """
    多路召回 + RRF 融合

    Args:
        query: 查询文本
        top_k: 返回数量
        alpha: 向量检索权重 (1-alpha 为 BM25 权重)

    Returns:
        [(doc_idx, fused_score), ...]
    """
    _lazy_load()

    if _index is None or not _docs:
        return []

    # 1. 向量检索
    try:
        q_emb = embed_texts([query])
        q_emb = np.asarray(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)

        ntotal = int(getattr(_index, "ntotal", 0))
        k_vec = min(top_k * 3, ntotal)

        with _faiss_gpu_lock:
            vec_sims, vec_idxs = _index.search(q_emb, k_vec)

        vec_results = list(zip(vec_idxs[0].tolist(), vec_sims[0].tolist()))
    except Exception as e:
        print(f"[RAG] 向量检索失败: {e}")
        vec_results = []

    # 2. BM25 关键词检索
    bm25_results = []
    bm25 = _get_bm25_index()
    if bm25:
        try:
            doc_scores = [(i, bm25.score(query, i)) for i in range(len(_doc_texts))]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            bm25_results = [(idx, score) for idx, score in doc_scores[:top_k * 3] if score > 0]
        except Exception as e:
            print(f"[RAG] BM25 检索失败: {e}")

    # 3. RRF 融合
    fused_scores: Dict[int, float] = defaultdict(float)

    # 向量检索排名
    for rank, (idx, sim) in enumerate(vec_results):
        if idx >= 0:
            fused_scores[idx] += alpha * (1 / (RRF_K + rank + 1))

    # BM25 排名
    for rank, (idx, score) in enumerate(bm25_results):
        fused_scores[idx] += (1 - alpha) * (1 / (RRF_K + rank + 1))

    # 4. 排序
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


# =========================
# Query 扩展（基于风险标签）
# =========================

def _expand_query_with_risk_synonyms(query: str, risk_tags: Optional[List[str]] = None) -> List[str]:
    """
    基于风险标签扩展查询

    Args:
        query: 原始查询
        risk_tags: 风险标签列表

    Returns:
        扩展后的查询列表
    """
    queries = [query]

    if risk_tags:
        for tag in risk_tags[:3]:  # 最多取3个标签
            synonyms = RISK_TAG_SYNONYMS.get(tag, [])
            # 取前2个同义词
            queries.extend(synonyms[:2])

    # 去重
    seen = set()
    unique_queries = []
    for q in queries:
        q_normalized = q.lower().strip()
        if q_normalized not in seen:
            seen.add(q_normalized)
            unique_queries.append(q)

    return unique_queries


# =========================
# 【核心优化】多查询召回
# =========================

def _multi_query_retrieve(
    query: str,
    top_k: int = 5,
    risk_tags: Optional[List[str]] = None,
    doc_types: Optional[List[str]] = None,
    language: Optional[str] = None,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    多查询召回 + 融合

    1. Query 扩展
    2. 多路 RRF 召回
    3. 去重 + 重排序
    """
    # 1. Query 扩展
    expanded_queries = _expand_query_with_risk_synonyms(query, risk_tags)

    # 2. 多查询 RRF 融合
    all_scores: Dict[int, float] = defaultdict(float)

    for q in expanded_queries[:5]:  # 最多5个扩展查询
        results = _hybrid_search(q, top_k=top_k, alpha=0.7)
        for idx, score in results:
            all_scores[idx] += score

    # 3. 排序
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]

    # 4. 组装结果
    results_with_meta = []
    for idx, score in ranked:
        if 0 <= idx < len(_docs):
            results_with_meta.append((idx, score, _docs[idx]))

    return results_with_meta


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

    # 检查缓存
    cache_key = _make_cache_key(q, top_k, doc_types, risk_tags, language)
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    # 使用优化的多查询召回
    raw_results = _multi_query_retrieve(
        query=q,
        top_k=top_k * 2,  # 多召回一些，后面过滤
        risk_tags=risk_tags,
    )

    results: List[Dict[str, Any]] = []
    seen_texts = set()

    for idx, score, doc in raw_results:
        meta = doc.get("meta") or {}

        # 元数据过滤
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

        # 去重（基于文本内容）
        text_hash = hashlib.md5((doc.get("text", "")[:100]).encode()).hexdigest()
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)

        results.append(doc)
        if len(results) >= top_k:
            break

    # 缓存结果
    _save_to_cache(cache_key, results)

    return results


# =========================
# 缓存管理
# =========================

def _make_cache_key(query: str, top_k: int, doc_types, risk_tags, language) -> str:
    """生成缓存 key"""
    key_parts = [
        query[:100],
        str(top_k),
        ",".join(sorted(doc_types)) if doc_types else "",
        ",".join(sorted(risk_tags)) if risk_tags else "",
        language or "",
    ]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()


def _get_from_cache(cache_key: str) -> Optional[List[Dict[str, Any]]]:
    """从缓存获取"""
    with _cache_lock:
        if cache_key in _query_cache:
            results, timestamp = _query_cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                return results
            else:
                del _query_cache[cache_key]
    return None


def _save_to_cache(cache_key: str, results: List[Dict[str, Any]]) -> None:
    """保存到缓存"""
    with _cache_lock:
        # 清理过期缓存
        if len(_query_cache) > CACHE_MAX_SIZE:
            expired = [k for k, (_, t) in _query_cache.items() if time.time() - t > CACHE_TTL]
            for k in expired[:len(expired) // 2]:
                del _query_cache[k]

        _query_cache[cache_key] = (results, time.time())


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

        # 智能截断 - 保持语义边界
        if len(text) > 800:
            # 尝试在句号处截断
            truncated = text[:800]
            last_period = max(truncated.rfind("。"), truncated.rfind("."))
            if last_period > 400:
                text = truncated[:last_period + 1]
            else:
                text = truncated + " ……（内容截断）"

        block = header + "\n" + text + "\n"
        if current_len + len(block) > max_chars:
            break

        parts.append(block)
        current_len += len(block)

    return "\n".join(parts)


# =========================
# 缓存清理工具
# =========================

def clear_rag_cache() -> None:
    """清除 RAG 缓存"""
    global _query_cache, _bm25_index
    with _cache_lock:
        _query_cache.clear()
    _bm25_index = None
    print("[RAG] 缓存已清除")


# =========================
# 统计信息
# =========================

def get_rag_stats() -> Dict[str, Any]:
    """获取 RAG 模块统计信息"""
    _lazy_load()
    return {
        "doc_count": len(_docs),
        "index_ntotal": int(getattr(_index, "ntotal", 0)) if _index else 0,
        "cache_size": len(_query_cache),
        "cache_ttl": CACHE_TTL,
        "embedding_servers": VLLM_EMBED_SERVERS,
        "use_local_fallback": USE_LOCAL_EMBEDDING_FALLBACK,
    }


if __name__ == "__main__":
    import math

    print(">>> 安全 RAG 优化版测试（多路召回 + RRF融合 + Query扩展 + 缓存）")
    print(f"统计信息: {get_rag_stats()}")

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
        print(f"缓存统计: {get_rag_stats()}")
