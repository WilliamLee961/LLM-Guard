# build_safety_index.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ========================
# 路径配置（按你的目录来）
# ========================

ROOT_DIR = Path("safety_knowledge")

POLICY_DIR = ROOT_DIR / "policies"
TEMPLATE_FILE = ROOT_DIR / "templates" / "答复模版.jsonl"
CASE_FILE = ROOT_DIR / "cases" / "经典案例.jsonl"

INDEX_DIR = ROOT_DIR / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "safety_index.faiss"
DOCS_JSONL_PATH = INDEX_DIR / "safety_docs.jsonl"

# 多语种 embedding 模型（支持中英）
EMB_MODEL_NAME = "/data/niujiafei/models/bge-m3"


# ========================
# 小工具：简单文本切片
# ========================

def chunk_text(text: str, max_len: int = 400, overlap: int = 80) -> List[str]:
    """
    简单按字符数切分文本（可以后面再换成按句子）。
    max_len: 每段最大长度
    overlap: 相邻两段重叠长度
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_len, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


# ========================
# 1) policies 加载 & 自动生成 meta
# ========================

def guess_risk_tags_from_policy_path(path: Path) -> List[str]:
    """
    根据政策文件所在目录的中文名字，粗略猜一个 risk_tags。
    目的是减少手工标注，不追求特别精细。
    """
    # 相对 policies 的路径，例如：国家法律/刑法/xxx.md
    rel = path.relative_to(POLICY_DIR)
    parts = rel.parts  # ('国家法律', '刑法', 'xxx.md') 或 ('使用政策', 'openAI.md')

    # 顶层类别：国家法律 / 使用政策 / 政策
    top = parts[0]

    # 默认：policy 属于 OTHER_RISK（泛安全政策）
    tags: List[str] = ["OTHER_RISK"]

    if top == "国家法律":
        # 二级目录：刑法 / 民法典 / 经济法 等
        if len(parts) >= 2:
            sub = parts[1]
            if "刑法" in sub or "治安" in sub or "司法解释" in sub:
                tags = ["CRIMES_ILLEGAL"]
            elif "民法" in sub:
                tags = ["PRIVACY_PROPERTY"]
            elif "经济法" in sub:
                tags = ["FINANCIAL_RISK"]
            elif "行政法" in sub or "行政法规" in sub:
                tags = ["CRIMES_ILLEGAL", "OTHER_RISK"]
            else:
                # 其他就让它保持 OTHER_RISK
                tags = ["OTHER_RISK"]
    elif top == "使用政策":
        # 各家大模型使用政策，和模型攻击、恶意使用、隐私等都相关
        tags = ["MALICIOUS_USE", "ATTACK_MODEL", "OTHER_RISK"]
    elif top == "政策":
        # “生成式人工智能服务管理暂行办法”等
        tags = ["MALICIOUS_USE", "MISINFORMATION", "PRIVACY_PROPERTY", "OTHER_RISK"]

    return tags


def load_policies() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not POLICY_DIR.exists():
        return docs

    for path in POLICY_DIR.rglob("*.md"):
        body = path.read_text(encoding="utf-8", errors="ignore")
        if not body.strip():
            continue

        rel = path.relative_to(POLICY_DIR)  # 例如 "国家法律/刑法/xxx.md"
        language = "zh"  # 这些基本都是中文
        risk_tags = guess_risk_tags_from_policy_path(path)

        chunks = chunk_text(body, max_len=400, overlap=80)
        for i, ch in enumerate(chunks):
            docs.append({
                "id": f"policy::{rel.as_posix()}#{i}",
                "text": ch,
                "meta": {
                    "doc_type": "policy",
                    "language": language,
                    "risk_tags": risk_tags,
                    "file": str(path),
                    "chunk_id": i,
                },
            })

    return docs


# ========================
# 2) 模板 jsonl：答复模版.jsonl
# ========================

def load_templates() -> List[Dict[str, Any]]:
    """
    约定答复模版.jsonl 每行类似：
    {
      "id": "self_harm_zh_001",
      "language": "zh",
      "risk_tags": ["SELF_HARM", "MENTAL_HEALTH"],
      "scene": "用户明确表达想自杀或严重自伤",
      "template": "真正给用户看的话术..."
    }
    """
    docs: List[Dict[str, Any]] = []
    if not TEMPLATE_FILE.exists():
        return docs

    with TEMPLATE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            template_text = (obj.get("template") or "").strip()
            if not template_text:
                continue

            language = obj.get("language") or "zh"
            risk_tags = obj.get("risk_tags") or []
            scene = obj.get("scene") or ""
            base_id = obj.get("id") or TEMPLATE_FILE.stem

            docs.append({
                "id": f"template::{base_id}",
                "text": template_text,
                "meta": {
                    "doc_type": "template",
                    "language": language,
                    "risk_tags": risk_tags,
                    "scene": scene,
                    "source": str(TEMPLATE_FILE),
                },
            })

    return docs


# ========================
# 3) 案例 jsonl：经典案例.jsonl
# ========================

def load_cases() -> List[Dict[str, Any]]:
    """
    建议 经典案例.jsonl 每行格式类似：
    {
      "id": "case_0001",
      "language": "zh",
      "risk_tags": ["CRIMES_ILLEGAL"],
      "question": "用户危险提问...",
      "safe_answer": "一个安全/合规的示例回答",
      "explain": "可选：为什么要这样回答"
    }

    如果你现在还没有这么多字段，也没关系，下面代码做了兜底处理。
    """
    docs: List[Dict[str, Any]] = []
    if not CASE_FILE.exists():
        return docs

    with CASE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            q = (obj.get("question") or "").strip()
            a = (obj.get("safe_answer") or "").strip()
            if not q and not a:
                continue

            language = obj.get("language") or "zh"
            risk_tags = obj.get("risk_tags") or []
            explain = obj.get("explain") or ""
            base_id = obj.get("id") or CASE_FILE.stem

            # 文本：把“问 + 安全答复”拼在一起，作为检索内容
            full_text = f"【用户问题】{q}\n【安全答复示例】{a}"
            docs.append({
                "id": f"case::{base_id}",
                "text": full_text,
                "meta": {
                    "doc_type": "case",
                    "language": language,
                    "risk_tags": risk_tags,
                    "explain": explain,
                    "source": str(CASE_FILE),
                },
            })

    return docs


# ========================
# 主流程：编码 + 建索引
# ========================

def main():
    print("[INFO] 开始加载安全知识文档...")

    docs_policies = load_policies()
    docs_templates = load_templates()
    docs_cases = load_cases()

    docs = docs_policies + docs_templates + docs_cases

    print(f"[INFO] 政策文档 chunk 数: {len(docs_policies)}")
    print(f"[INFO] 答复模版数: {len(docs_templates)}")
    print(f"[INFO] 经典案例数: {len(docs_cases)}")
    print(f"[INFO] 总文档数: {len(docs)}")

    if not docs:
        print("[WARN] 没有加载到任何文档，请检查目录结构。")
        return

    texts = [d["text"] for d in docs]

    print(f"[INFO] 加载 embedding 模型: {EMB_MODEL_NAME}")
    model = SentenceTransformer(EMB_MODEL_NAME)

    print("[INFO] 开始编码文本为向量...")
    embs = model.encode(texts, batch_size=32, show_progress_bar=True)
    embs = np.asarray(embs, dtype="float32")
    faiss.normalize_L2(embs)

    dim = embs.shape[1]
    print(f"[INFO] 向量维度: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    print(f"[INFO] 索引中向量数: {index.ntotal}")

    print(f"[INFO] 保存索引到: {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))

    print(f"[INFO] 保存文档元信息到: {DOCS_JSONL_PATH}")
    with DOCS_JSONL_PATH.open("w", encoding="utf-8") as f_out:
        for d in docs:
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("[INFO] 安全知识索引构建完成。")


if __name__ == "__main__":
    main()