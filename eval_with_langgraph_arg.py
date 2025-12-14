import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd  # [新增] 用于处理数据和导出 CSV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
from safety_guard import SafetyGuard
from guard_graph import build_guard_graph, GuardState

#######################################
# 配置区
#######################################

DEFAULT_JSONL_PATH = ".datasets/typical_safety_scenarios_final.jsonl"
DEFAULT_DETECT_MODE = "normal"
MAX_CONCURRENT_REQUESTS = 50

# 默认字段名
DEFAULT_LABEL_FIELD = "label"
DEFAULT_RAW_FIELD = "response_with_qwen"
DEFAULT_GUARD_FIELD = "response_with_qwen_after_guard"

# 标签映射
GOLD_LABEL_MAPPING = {
    "不违规": 0,
    "违规": 1,
}

#######################################
# 命令行参数解析
#######################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="多维度评估 GuardGraph: 对比 Guard 前后的安全性变化"
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default=DEFAULT_JSONL_PATH,
        help="包含 label, raw_answer, guard_answer 的 jsonl 文件路径",
    )
    parser.add_argument(
        "--raw_field",
        type=str,
        default=DEFAULT_RAW_FIELD,
        help=f"原始模型回答的字段名 (默认: {DEFAULT_RAW_FIELD})",
    )
    parser.add_argument(
        "--guard_field",
        type=str,
        default=DEFAULT_GUARD_FIELD,
        help=f"Guard 处理后的回答字段名 (默认: {DEFAULT_GUARD_FIELD})",
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default=DEFAULT_LABEL_FIELD,
        help=f"真实标签字段名 (默认: {DEFAULT_LABEL_FIELD})",
    )
    parser.add_argument(
        "--detect_mode",
        type=str,
        default=DEFAULT_DETECT_MODE,
        choices=["lenient", "normal", "strict", "any_risk", "overall_l2plus"],
        help="检测难度模式 (用于 GuardGraph 的判断标准)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help="并发线程数",
    )
    return parser.parse_args()


#######################################
# 工具函数
#######################################

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def is_true_violation(raw_label: str) -> int:
    raw_label = (raw_label or "").strip()
    return GOLD_LABEL_MAPPING.get(raw_label, 0) # 默认 0 容错


def predict_for_one(text: str, graph) -> int:
    """
    仅返回预测结果 (0/1)，用于快速评估
    """
    if not text:
        return 0
    try:
        init_state: GuardState = {"text": text}
        result_state = graph.invoke(init_state)
        return int(result_state.get("pred_violation", 0))
    except Exception:
        return 0


#######################################
# 核心逻辑：计算指标并返回结构化数据
#######################################

def compute_metrics(y_true, y_pred, report_type, description, mode):
    """
    计算各项指标并返回一个字典（用于生成 CSV 的一行）以及文本报告字符串
    """
    # 1. 计算混淆矩阵 (labels=[0, 1] 确保矩阵一定是 2x2，即使数据里只有一种类别)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 2. 计算分类报告 (获取精确率、召回率、F1)
    # output_dict=True 让它返回字典而不是字符串
    # target_names 不指定，默认 key 为 '0', '1'
    report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    
    acc = accuracy_score(y_true, y_pred)
    
    # 3. 构造一行数据 (遵循你的 Excel 格式要求)
    row = {
        "Report Type": report_type,
        "Description": description,
        "Mode": mode,
        "Accuracy": round(acc, 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        
        # Unsafe (Class 1) Metrics
        "Unsafe_Precision": round(report.get("1", {}).get("precision", 0), 4),
        "Unsafe_Recall":    round(report.get("1", {}).get("recall", 0), 4),
        "Unsafe_F1":        round(report.get("1", {}).get("f1-score", 0), 4),
        
        # Safe (Class 0) Metrics
        "Safe_Precision":   round(report.get("0", {}).get("precision", 0), 4),
        "Safe_Recall":      round(report.get("0", {}).get("recall", 0), 4),
        "Safe_F1":          round(report.get("0", {}).get("f1-score", 0), 4),
    }
    
    # 4. 生成用于打印的可读字符串 (保留原来的 report.txt 功能)
    cm_str = (
        f"Confusion Matrix:\n"
        f"                Pred 0    Pred 1\n"
        f"True 0 (不违规)   {tn:<8}  {fp:<8}\n"
        f"True 1 (违规)     {fn:<8}  {tp:<8}\n"
    )
    
    text_report = (
        f"====== {report_type} ======\n"
        f"Description: {description}\n"
        f"{cm_str}\n"
        f"{classification_report(y_true, y_pred, target_names=['不违规(0)', '违规(1)'], digits=4, zero_division=0)}\n"
        f"Accuracy: {acc:.4f}\n"
        f"--------------------------------------------------\n\n"
    )
    
    return row, text_report


#######################################
# 主流程
#######################################

def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl_path)
    
    # 输出文件路径 (带时间戳)
    base_name = jsonl_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_txt_path = jsonl_path.parent / f"{base_name}_{args.detect_mode}_eval_{timestamp}.txt"
    report_csv_path = jsonl_path.parent / f"{base_name}_{args.detect_mode}_eval_{timestamp}.csv"
    
    print(f"加载数据: {jsonl_path}")
    print(f"检测模式: {args.detect_mode}")
    print(f"字段映射: Raw='{args.raw_field}' | Guard='{args.guard_field}' | Label='{args.label_field}'")

    records = read_jsonl(str(jsonl_path))
    print(f"总数据量: {len(records)}")

    # 准备数据容器
    y_true_question = []   # Ground Truth (Question Label)
    raw_answers = []       # 原始回答
    guard_answers = []     # Guard 后回答
    
    valid_count = 0

    for idx, item in enumerate(records):
        # 1. 检查 Label
        if args.label_field not in item:
            continue
        
        # 2. 检查 Raw Answer
        if args.raw_field not in item:
            continue
            
        # 3. 检查 Guard Answer (如果没有，默认用 Raw 填充，视作未改动)
        guard_text = item.get(args.guard_field, item[args.raw_field])

        # 解析 Label
        q_label_str = item[args.label_field]
        y_true_question.append(is_true_violation(q_label_str))
        
        raw_answers.append(item[args.raw_field])
        guard_answers.append(guard_text)
        
        valid_count += 1

    print(f"有效评估样本: {valid_count}")

    # ===== 构建 Graph =====
    print(f"构建 GuardGraph (Mode: {args.detect_mode})...")
    graph = build_guard_graph(detection_difficulty=args.detect_mode)

    # ===== 批量预测 =====
    # 我们需要跑两轮预测：一轮针对 Raw Answer，一轮针对 Guard Answer
    
    y_pred_raw = [0] * valid_count
    y_pred_guard = [0] * valid_count

    print("正在评估...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交 Raw Answer 任务
        futures_raw = {executor.submit(predict_for_one, txt, graph): i for i, txt in enumerate(raw_answers)}
        for f in tqdm(as_completed(futures_raw), total=valid_count, desc="Eval Raw Answer"):
            idx = futures_raw[f]
            y_pred_raw[idx] = f.result()

        # 提交 Guard Answer 任务
        futures_guard = {executor.submit(predict_for_one, txt, graph): i for i, txt in enumerate(guard_answers)}
        for f in tqdm(as_completed(futures_guard), total=valid_count, desc="Eval Guard Answer"):
            idx = futures_guard[f]
            y_pred_guard[idx] = f.result()

    # ===== 生成报告 (文本 + CSV) =====
    
    full_text_report = f"评估报告 | Mode: {args.detect_mode} | File: {jsonl_path.name}\n\n"
    csv_rows = []

    # Report 1: Question Label vs Raw Answer Pred
    # 衡量：原始模型的回答是否安全（相对于问题标签）
    row1, txt1 = compute_metrics(
        y_true_question, y_pred_raw, 
        "Report 1: Question Label (GT) vs Raw Answer (Pred)",
        "原始模型回答安全性评估 (基线)",
        args.detect_mode
    )
    csv_rows.append(row1)
    full_text_report += txt1

    # Report 2: Question Label vs Guard Answer Pred
    # 衡量：加上 Guard 后，最终输出的安全性
    row2, txt2 = compute_metrics(
        y_true_question, y_pred_guard, 
        "Report 2: Question Label (GT) vs Guard Answer (Pred)",
        "Guard处理后回答安全性评估 (最终效果)",
        args.detect_mode
    )
    csv_rows.append(row2)
    full_text_report += txt2

    # Report 3: Raw Answer Pred vs Guard Answer Pred
    # 衡量：清洗效果（有多少 Raw=1 变成了 Guard=0）
    # 注意：这里 Raw Answer 充当 "True" (基准)，Guard Answer 充当 "Pred" (变化)
    row3, txt3 = compute_metrics(
        y_pred_raw, y_pred_guard, 
        "Report 3: Raw Answer (Pred) vs Guard Answer (Pred)",
        "处理前后变化对比 (清洗幅度)",
        args.detect_mode
    )
    csv_rows.append(row3)
    full_text_report += txt3

    print("\n" + full_text_report)
    
    # 保存文本报告
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text_report)
    print(f"[INFO] 文本报告已保存至: {report_txt_path}")

    # 保存 CSV 报告
    df = pd.DataFrame(csv_rows)
    # 强制调整列顺序以符合要求
    cols = [
        "Report Type", "Description", "Mode", "Accuracy",
        "TN", "FP", "FN", "TP",
        "Unsafe_Precision", "Unsafe_Recall", "Unsafe_F1",
        "Safe_Precision", "Safe_Recall", "Safe_F1"
    ]
    # 过滤掉 dataframe 中不存在的列（如果有的话），防止报错
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    
    df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] CSV 表格已保存至: {report_csv_path}")

if __name__ == "__main__":
    main()