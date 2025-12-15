# eval_binary_risk_mt.py
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from guard_graph import build_guard_graph, GuardState  # 注意：从 guard_graph 导入


#######################################
# 配置区（按你的数据情况修改）
#######################################

# jsonl 文件路径
JSONL_PATH = "../typical_safety_scenarios_final.jsonl"

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

# 检测难度（新的开关）：
#   - "lenient"        : 宽松（误杀少、可能多放）
#   - "normal"         : 默认
#   - "strict"         : 严格（宁可多杀少放）
#   - 兼容老模式：
#       "any_risk"       : 任意非 NONE 即违规
#       "overall_l2plus" : overall ∈ {L2,L3,L4} 即违规
DETECTION_MODE = "strict"

# 并发线程数（根据你调用 LLM 的吞吐调整）
MAX_CONCURRENT_REQUESTS = 64

# 错误样本保存路径（jsonl）
ERROR_OUTPUT_PATH = JSONL_PATH.split('/')[-1].replace('.json','_error.jsonl')


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


def predict_for_one(text: str, graph) -> Dict[str, Any]:
    """
    使用 LangGraph 图对单条文本做安全检测：
      输入：
        - text: 待检测文本
        - graph: build_guard_graph(...) 得到的 app

      输出：
        {
          "pred": 0/1,
          "router_ctx": {...},
          "risk_profile": {...}
        }
    """
    init_state: GuardState = {"text": text}
    result_state: GuardState = graph.invoke(init_state)

    pred = int(result_state.get("pred_violation", 0))
    router_ctx = result_state.get("router_ctx")
    risk_profile = result_state.get("risk_profile")

    return {
        "pred": pred,
        "router_ctx": router_ctx,
        "risk_profile": risk_profile,
    }


#######################################
# 主流程（批量并行评估）
#######################################

def main():
    jsonl_path = Path(JSONL_PATH)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到 jsonl 文件：{jsonl_path}")

    print("加载数据...")
    records = read_jsonl(str(jsonl_path))   # 按需裁剪
    print(f"共读取 {len(records)} 条样本")

    texts: List[str] = []
    y_true: List[int] = []
    eval_records: List[Dict[str, Any]] = []  # 与 texts / y_true 对齐的记录

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

        # 保存一份与评估对齐的记录（可附加索引便于追溯）
        rec = dict(item)  # 浅拷贝，避免直接改原 records
        rec["_eval_idx"] = idx          # 原始行号
        rec["_eval_text"] = text        # 实际用于评估的文本
        eval_records.append(rec)

    print(f"有效样本数：{len(texts)}")

    # ===== 构建 LangGraph Guardrails 工作流 =====
    print(f"构建 LangGraph 工作流，DETECTION_MODE = {DETECTION_MODE!r} ...")
    graph = build_guard_graph(detection_difficulty=DETECTION_MODE)

    print("开始批量并行评估（通过 LangGraph 调用 routing_hub + risk_hub）...")

    y_pred: List[int] = [0] * len(texts)
    router_ctx_list: List[Dict[str, Any]] = [None] * len(texts)
    risk_profile_list: List[Dict[str, Any]] = [None] * len(texts)

    # 用线程池并发请求
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(predict_for_one, texts[i], graph): i
            for i in range(len(texts))
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                result = future.result()
                pred = result["pred"]
                router_ctx_list[i] = result.get("router_ctx")
                risk_profile_list[i] = result.get("risk_profile")
            except Exception as e:
                # 这里根据需要决定是记录错误还是给默认值
                print(f"\n样本 {i} 推理失败: {e}")
                print(future)
                pred = 0  # 默认当作不违规，避免中断整个评估
                router_ctx_list[i] = None
                risk_profile_list[i] = None

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

    # ========== 保存判错样本（包含模型推理信息） ==========
    error_indices = np.where(y_true_arr != y_pred_arr)[0]
    print(f"共有 {len(error_indices)} 条样本预测错误，将保存到: {ERROR_OUTPUT_PATH}")

    with open(ERROR_OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for i in error_indices:
            rec = dict(eval_records[i])  # 再拷一份，避免污染内存里的结构
            rec["_gold_label"] = int(y_true_arr[i])   # 0/1 真实
            rec["_pred_label"] = int(y_pred_arr[i])   # 0/1 预测

            if y_true_arr[i] == 1 and y_pred_arr[i] == 0:
                rec["_error_type"] = "FN"  # 漏检：真实违规 -> 预测不违规
            elif y_true_arr[i] == 0 and y_pred_arr[i] == 1:
                rec["_error_type"] = "FP"  # 误杀：真实不违规 -> 预测违规
            else:
                rec["_error_type"] = "OTHER"

            # 把 LangGraph 图里各节点的中间结果也写进去
            rec["_router_ctx"] = router_ctx_list[i]
            rec["_risk_profile"] = risk_profile_list[i]

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[INFO] 错误样本已保存。")


if __name__ == "__main__":
    main()