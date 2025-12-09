# eval_binary_risk_mt.py
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from routing_hub import analyze_query
from risk_hub import detect_risk_discrete


#######################################
# 配置区（按你的数据情况修改）
#######################################

# jsonl 文件路径
JSONL_PATH = "../safety_en_14k_final.jsonl"

# 文本字段名
TEXT_FIELD = "question"      # 或 "prompt"

# 标签字段名
LABEL_FIELD = "label"

# 真实标签到 0/1 的映射：
#   1 表示违规
#   0 表示不违规
GOLD_LABEL_MAPPING = {
    "不违规": 0,
    "违规": 1,
}

# 判断“有风险”的策略：
#   MODE = "any_risk"      : 只要有任何风险大类 != NONE，就算预测违规
#   MODE = "overall_l2plus": overall_risk_level 在 L2/L3/L4 才算预测违规
DETECTION_MODE = "any_risk"

# 并发线程数（根据你调用 LLM 的吞吐调整）
MAX_CONCURRENT_REQUESTS = 32


#######################################
# 工具函数
#######################################

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """逐行读取 jsonl 文件，返回列表[dict, ...]"""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def is_true_violation(raw_label: str) -> int:
    """
    把数据里的 label 映射成 0/1：
      - 违规    -> 1
      - 不违规  -> 0
    """
    raw_label = (raw_label or "").strip()
    if raw_label not in GOLD_LABEL_MAPPING:
        raise ValueError(f"真实标签 {raw_label!r} 不在 GOLD_LABEL_MAPPING 中，请修改映射。")
    return GOLD_LABEL_MAPPING[raw_label]


def is_predicted_violation(risk_profile: Dict[str, Any]) -> int:
    """
    根据 risk_hub.detect_risk_discrete 的输出，压成二分类结果：
      - 1: 检测到有风险（预测违规）
      - 0: 未检测到风险（预测不违规）
    """
    levels = risk_profile.get("risk_levels") or {}
    overall = (risk_profile.get("overall_risk_level") or "L1").upper()

    if DETECTION_MODE == "overall_l2plus":
        return 1 if overall in ("L2", "L3", "L4") else 0

    # 默认 any_risk：只要有任何一个大类不是 NONE，就算检测到风险
    for v in levels.values():
        if (v or "").upper() != "NONE":
            return 1
    return 0


def predict_for_one(text: str) -> int:
    """
    对单条文本：
      1) 调用 routing_hub.analyze_query
      2) 调用 risk_hub.detect_risk_discrete
      3) 压成二分类预测标签 0/1
    """
    # 1. 围栏路由
    router_ctx = analyze_query(text)

    # 2. 风险检测
    risk_profile = detect_risk_discrete(text, router_ctx, mode="input")

    # 3. 压成二分类
    pred = is_predicted_violation(risk_profile)
    return pred


#######################################
# 主流程（批量并行评估）
#######################################

def main():
    jsonl_path = Path(JSONL_PATH)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到 jsonl 文件：{jsonl_path}")

    print("加载数据...")
    records = read_jsonl(str(jsonl_path))
    print(f"共读取 {len(records)} 条样本")

    texts: List[str] = []
    y_true: List[int] = []

    # 先把文本和真实标签整理出来
    for idx, item in enumerate(records):
        if TEXT_FIELD not in item:
            raise KeyError(f"第 {idx} 条记录中找不到字段 '{TEXT_FIELD}'，请检查 TEXT_FIELD 设置")
        if LABEL_FIELD not in item:
            raise KeyError(f"第 {idx} 条记录中找不到字段 '{LABEL_FIELD}'，请检查 LABEL_FIELD 设置")

        text = (item[TEXT_FIELD] or "").strip()
        gold_raw = (item[LABEL_FIELD] or "").strip()

        if not text:
            # 空文本可以选择跳过
            continue

        gold = is_true_violation(gold_raw)

        texts.append(text)
        y_true.append(gold)

    print(f"有效样本数：{len(texts)}")
    print("开始批量并行评估（调用 routing_hub + risk_hub）...")

    y_pred: List[int] = [0] * len(texts)

    # 用线程池并发请求
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(predict_for_one, texts[i]): i
            for i in range(len(texts))
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                pred = future.result()
            except Exception as e:
                # 这里根据需要决定是记录错误还是给默认值
                print(f"\n样本 {i} 推理失败: {e}")
                print(future)
                pred = 0  # 默认当作不违规，避免中断整个评估
            y_pred[i] = pred

    # ========== 输出分类报告和 ACC ==========
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    print("\n====== 二分类报告 (0: 不违规, 1: 违规) ======")
    print(
        classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=["不违规", "违规"],
            digits=4,
        )
    )

    acc = accuracy_score(y_true_arr, y_pred_arr)
    print(f"整体准确率 ACC: {acc:.4f}")


if __name__ == "__main__":
    main()