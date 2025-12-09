# eval_safe_qa_binary_mt.py
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from safe_qa_graph import build_safe_qa_graph, QAState


#######################################
# 配置区（按你的数据情况修改）
#######################################

# jsonl 文件路径
JSONL_PATH = "../ChineseSafe.jsonl"

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

# 检测难度（传给 safe_qa_graph / is_predicted_violation）：
#   - "lenient"        : 宽松（误杀少、可能多放）
#   - "normal"         : 默认
#   - "strict"         : 严格（宁可多杀少放）
#   - 兼容老模式：
#       "any_risk"       : 任意非 NONE 即违规
#       "overall_l2plus" : overall ∈ {L2,L3,L4} 即违规
DETECTION_MODE = "normal"

# 评估时怎么用输入/输出侧的结果压成一个二分类：
#   - "input_only"       : 只看输入侧 input_violation
#   - "output_only"      : 只看输出侧 output_violation（注意：模型回答要跑一遍）
#   - "input_or_output"  : 只要输入 or 输出任一侧判违规，就算违规
AGG_MODE = "input_or_output"

# 并发线程数（根据你调用 LLM 的吞吐调整）
MAX_CONCURRENT_REQUESTS = 32

# 错误样本保存路径（jsonl）
ERROR_OUTPUT_PATH = "./eval_safe_qa_errors.jsonl"

# 是否裁剪样本数量（例如只评估前 2000 条），None 表示用全部
MAX_SAMPLES = None    # 比如想只跑 2000，可以改成 2000


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
    return data[:2000]


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


def aggregate_prediction_from_state(state: QAState, agg_mode: str = "input_or_output") -> int:
    """
    从 safe_qa_graph 的状态中抽取一个二分类预测标签 0/1。

    agg_mode:
      - "input_only"      : 只看 input_violation
      - "output_only"     : 只看 output_violation
      - "input_or_output" : 只要任一为 1 就算违规
    """
    iv = int(state.get("input_violation", 0) or 0)
    ov = int(state.get("output_violation", 0) or 0)

    if agg_mode == "input_only":
        return iv
    elif agg_mode == "output_only":
        return ov
    else:
        # 默认 input_or_output
        return 1 if (iv == 1 or ov == 1) else 0


def predict_for_one(text: str, graph, agg_mode: str = "input_or_output") -> Dict[str, Any]:
    """
    使用 safe_qa_graph 对单条文本做完整流程：
      输入：
        - text: 用户问题
        - graph: build_safe_qa_graph(...) 得到的 app
      输出：
        {
          "pred": 0/1,
          "state": QAState  # 包含 router_ctx_in/out, risk_profile, final_answer 等
        }
    """
    init_state: QAState = {
        "user_query": text,
        "difficulty": DETECTION_MODE,  # 虽然现在 graph 中没用到这个字段，但保留以防后续扩展
    }
    result_state: QAState = graph.invoke(init_state)

    pred = aggregate_prediction_from_state(result_state, agg_mode=agg_mode)

    return {
        "pred": pred,
        "state": result_state,
    }


#######################################
# 主流程（批量并行评估）
#######################################

def main():
    jsonl_path = Path(JSONL_PATH)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到 jsonl 文件：{jsonl_path}")

    print("加载数据...")
    records = read_jsonl(str(jsonl_path))
    if MAX_SAMPLES is not None:
        records = records[:MAX_SAMPLES]

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

    # ===== 构建安全 QA LangGraph 工作流 =====
    print(f"构建 Safe QA LangGraph，DETECTION_MODE = {DETECTION_MODE!r}, AGG_MODE = {AGG_MODE!r} ...")
    graph = build_safe_qa_graph(detection_difficulty=DETECTION_MODE)

    print("开始批量并行评估（调用 safe_qa_graph 全流程）...")

    y_pred: List[int] = [0] * len(texts)
    # 为了后面排错保存信息，把完整 state 也存起来
    states: List[QAState] = [None] * len(texts)

    # 用线程池并发请求
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(predict_for_one, texts[i], graph, AGG_MODE): i
            for i in range(len(texts))
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                result = future.result()
                pred = int(result["pred"])
                state = result["state"]
            except Exception as e:
                # 这里根据需要决定是记录错误还是给默认值
                print(f"\n样本 {i} 推理失败: {e}")
                print(future)
                pred = 0  # 默认当作不违规，避免中断整个评估
                state = {}  # 空 state

            y_pred[i] = pred
            states[i] = state

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

            # 把图里的中间结果也写进去，方便分析
            st = states[i] or {}
            rec["_router_ctx_in"] = st.get("router_ctx_in")
            rec["_input_risk_profile"] = st.get("input_risk_profile")
            rec["_input_violation"] = st.get("input_violation")

            rec["_model_answer"] = st.get("model_answer")
            rec["_router_ctx_out"] = st.get("router_ctx_out")
            rec["_output_risk_profile"] = st.get("output_risk_profile")
            rec["_output_violation"] = st.get("output_violation")

            rec["_final_answer"] = st.get("final_answer")
            rec["_rag_docs"] = st.get("rag_docs")

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[INFO] 错误样本已保存。")


if __name__ == "__main__":
    main()